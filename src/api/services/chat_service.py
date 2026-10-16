"""Server-side AI chat session manager.

Holds conversation contexts in-memory, delegates AI inference to the
configured adapter (Ollama/OpenAI/Anthropic/Gemini), and persists
sessions to disk for cross-process sharing.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.shared.python.app_state import get_state_logger
from src.shared.python.core.contracts import precondition
from src.shared.python.core.error_utils import InvalidRequestError
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from src.shared.python.ai.adapters.base import BaseAgentAdapter
    from src.shared.python.ai.types import ConversationContext

logger = get_logger(__name__)

UTC = timezone.utc


class ChatService:
    """Server-side chat session manager.

    Manages conversation contexts in-memory with TTL eviction,
    persists to ~/.golf_modeling_suite/chat_sessions/ on each message,
    and delegates AI inference to the configured adapter.
    """

    MAX_SESSIONS = 50
    SESSION_TTL_SECONDS = 7200  # 2 hours
    PERSIST_DIR = Path.home() / ".golf_modeling_suite" / "chat_sessions"

    def __init__(
        self,
        app_state_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        if app_state_provider is not None and not callable(app_state_provider):
            raise TypeError(
                "app_state_provider must be a callable returning a dict, or None"
            )
        self._app_state_provider = app_state_provider
        self._sessions: OrderedDict[str, ConversationContext] = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self._adapter: BaseAgentAdapter | None = None
        # Human-readable reason when no chat backend could be loaded.
        self._backend_error: str | None = None
        self._lock = threading.Lock()

        # Pinned Tools registrars type-check against the canonical Tools
        # ``ToolRegistry``; UD's ``src.shared.python.ai`` copy is a distinct
        # class, so build the registry from the canonical spelling (UD #12893).
        from shared.python.ai.sample_tools import register_golf_suite_tools
        from shared.python.ai.tool_registry import ToolRegistry

        self._tool_registry = ToolRegistry()
        register_golf_suite_tools(self._tool_registry)
        self._register_canonical_core_retrieval_tool()
        self._tool_declarations = self._build_tool_declarations()

        self._load_adapter()

    def _register_canonical_core_retrieval_tool(self) -> None:
        """Register the bounded Canonical Core docs/schema Q&A tool."""
        from shared.python.ai.tool_registry import ToolCategory
        from src.shared.python.canonical_core import answer_canonical_core_question

        self._tool_registry.register(
            name="answer_canonical_core_question",
            description=(
                "Answer Canonical Core setup questions from the bounded local "
                "docs/schema index. Returns deterministic text with source "
                "citations; does not write files or run commands."
            ),
            category=ToolCategory.EDUCATIONAL,
            expertise_level=1,
        )(answer_canonical_core_question)

    def _build_tool_declarations(self) -> list[Any]:
        from src.shared.python.ai.adapters.base import ToolDeclaration

        tool_declarations = []
        for t in self._tool_registry.list_tools():
            props = {p.name: p.to_json_schema() for p in t.parameters}
            reqs = [p.name for p in t.parameters if p.required]
            tool_declarations.append(
                ToolDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=props,
                    required=reqs,
                )
            )
        return tool_declarations

    def _load_adapter(self) -> None:
        """Load AI adapter from persisted user settings."""
        try:
            from src.shared.python.ai.gui.settings_dialog import (
                AIProvider,
                AISettings,
                get_api_key,
            )

            settings = AISettings.load()

            if settings.provider == AIProvider.OLLAMA:
                from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

                self._adapter = OllamaAdapter(
                    host=settings.ollama_host,
                    model=settings.model,
                )
            elif settings.provider == AIProvider.OPENAI:
                api_key = get_api_key(AIProvider.OPENAI)
                if api_key:
                    from src.shared.python.ai.adapters.openai_adapter import (
                        OpenAIAdapter,
                    )

                    self._adapter = OpenAIAdapter(api_key=api_key, model=settings.model)
            elif settings.provider == AIProvider.ANTHROPIC:
                api_key = get_api_key(AIProvider.ANTHROPIC)
                if api_key:
                    from src.shared.python.ai.adapters.anthropic_adapter import (
                        AnthropicAdapter,
                    )

                    self._adapter = AnthropicAdapter(
                        api_key=api_key, model=settings.model
                    )
            elif settings.provider == AIProvider.GEMINI:
                api_key = get_api_key(AIProvider.GEMINI)
                if api_key:
                    from src.shared.python.ai.adapters.gemini_adapter import (
                        GeminiAdapter,
                    )

                    self._adapter = GeminiAdapter(api_key=api_key, model=settings.model)

            if self._adapter:
                logger.info("ChatService loaded adapter: %s", settings.provider.name)
            else:
                logger.warning(
                    "ChatService: no adapter configured, falling back to Ollama"
                )
                self._fallback_to_ollama()
        except ImportError as e:
            logger.warning(
                "ChatService: failed to load settings (%s), falling back to Ollama", e
            )
            self._fallback_to_ollama()

    def _fallback_to_ollama(self) -> None:
        """Fall back to default Ollama adapter.

        On failure the service has no usable chat backend; this surfaces a
        clear ``_backend_error`` state (see :attr:`adapter_available` and
        :attr:`backend_error`) rather than degrading silently.
        """
        try:
            from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

            self._adapter = OllamaAdapter()
            self._backend_error = None
            logger.info("ChatService using default OllamaAdapter")
        except ImportError:
            self._adapter = None
            self._backend_error = (
                "No chat backend available: failed to import the fallback "
                "Ollama adapter. Configure an AI provider in Settings > AI."
            )
            logger.exception("ChatService: could not create fallback adapter")

    @property
    def adapter_available(self) -> bool:
        """Whether a usable chat backend adapter is loaded."""
        return self._adapter is not None

    @property
    def backend_error(self) -> str | None:
        """Reason no chat backend is available, or ``None`` when one is."""
        return self._backend_error

    def _build_app_state_message(self) -> Any | None:
        """Build a system :class:`Message` containing the current app state.

        Returns ``None`` when no provider is configured or when the provider
        raises an exception (degraded gracefully — chat continues).

        Returns:
            A ``Message(role="system", ...)`` instance, or ``None``.
        """
        if self._app_state_provider is None:
            return None
        try:
            state = self._app_state_provider()
            content = "Current application state:\n" + json.dumps(
                state, indent=2, default=str
            )
            from src.shared.python.ai.types import Message

            return Message(role="system", content=content)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ChatService: app_state_provider raised %s — skipping state injection",
                exc,
            )
            return None

    def get_or_create_session(self, session_id: str | None) -> ConversationContext:
        """Return existing session or create a new one."""
        from src.shared.python.ai.types import ConversationContext

        with self._lock:
            self._cleanup_expired()

            if session_id and session_id in self._sessions:
                self._timestamps[session_id] = time.monotonic()
                return self._sessions[session_id]

            # Try loading from disk
            if session_id:
                ctx = self._load_session(session_id)
                if ctx:
                    self._sessions[session_id] = ctx
                    self._timestamps[session_id] = time.monotonic()
                    return ctx

            # Create new session
            ctx = ConversationContext()
            self._sessions[ctx.session_id] = ctx
            self._timestamps[ctx.session_id] = time.monotonic()

            # Evict oldest if adding pushed us over max
            while len(self._sessions) > self.MAX_SESSIONS:
                oldest_sid, _ = self._sessions.popitem(last=False)
                self._timestamps.pop(oldest_sid, None)

            logger.info("ChatService: created session %s", ctx.session_id)
            return ctx

    @precondition(
        lambda self, session_id, message, engine_context=None: (
            session_id is not None and len(session_id) > 0
        ),
        "Session ID must be a non-empty string",
    )
    @precondition(
        lambda self, session_id, message, engine_context=None: (
            message is not None and len(message) > 0
        ),
        "Message must be a non-empty string",
    )
    def add_user_message(
        self,
        session_id: str,
        message: str,
        engine_context: str | None = None,
    ) -> str:
        """Add a user message to the session and return a message ID."""
        with self._lock:
            ctx = self._sessions.get(session_id)
            if not ctx:
                raise InvalidRequestError(f"Session {session_id} not found")

            # Prepend engine context hint if provided
            content = message
            if engine_context:
                ctx.metadata["last_engine"] = engine_context

            ctx.add_user_message(content)
            self._persist_session(session_id)

        get_state_logger().log_event(
            "chat.message_sent",
            {"session_id": session_id, "role": "user"},
        )
        return str(uuid.uuid4().hex[:12])

    @precondition(
        lambda self, session_id: session_id is not None and len(session_id) > 0,
        "Session ID must be a non-empty string",
    )
    async def stream_response(self, session_id: str) -> AsyncIterator[Any]:  # noqa: C901
        """Stream AI response chunks for the latest user message.

        Runs the synchronous adapter in a thread pool executor.
        """
        if not self._adapter:
            if self._backend_error:
                yield self._backend_error
            else:
                yield (
                    "I'm not connected to an AI provider. Please configure "
                    "one in the launcher Settings > AI."
                )
            return

        with self._lock:
            ctx = self._sessions.get(session_id)
            if not ctx:
                yield "Session not found."
                return

        temp_ctx = self._build_adapter_context(ctx)

        import queue

        chunk_queue: queue.Queue[Any] = queue.Queue()

        # Cancellation token (#6981): set by the async consumer's ``finally``
        # on client disconnect / generator close. The worker checks it in its
        # loops and exits promptly instead of pulling from the adapter and
        # persisting messages for an abandoned session, which previously left
        # the daemon thread orphaned and contending for ``self._lock``.
        stop_event = threading.Event()
        temp_ctx.metadata["stop_event"] = stop_event

        worker = _StreamWorker(
            service=self,
            ctx=ctx,
            temp_ctx=temp_ctx,
            session_id=session_id,
            chunk_queue=chunk_queue,
            stop_event=stop_event,
        )

        thread = threading.Thread(
            target=worker.run,
            name=f"ChatService-StreamWorker-{session_id}",
            daemon=True,
        )
        thread.start()

        # The ``finally`` below runs on normal completion, on consumer error,
        # and crucially when the client disconnects -- closing this async
        # generator (``aclose``) raises ``GeneratorExit`` at the ``yield``.
        # In every case we set the stop flag so the worker exits and then
        # join it (bounded), so no orphaned daemon thread is left behind
        # (#6981).
        try:
            while True:
                try:
                    item = await asyncio.to_thread(chunk_queue.get, timeout=60.0)
                except queue.Empty:
                    yield {
                        "type": "error",
                        "detail": "AI provider connection timed out. Please check that your AI backend (e.g. Ollama or API key configuration) is running and reachable.",
                    }
                    break
                except (FileNotFoundError, OSError):
                    break
                if item is None:
                    break
                yield item
        finally:
            stop_event.set()
            # ``stop_event`` is only observed between adapter chunks. A
            # provider parked inside a socket read (Ollama's
            # ``iter_lines()``, the OpenAI SDK stream iterator) never gets
            # there, so escalate by closing the provider stream itself --
            # otherwise the join below expires and the worker leaks (#9509).
            self._cancel_adapter_stream()
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(
                    "ChatService stream worker for session %s did not exit "
                    "within 5s of cancellation; provider stream may not be "
                    "interruptible",
                    session_id,
                )

    def _build_adapter_context(self, ctx: ConversationContext) -> ConversationContext:
        """Copy ``ctx`` for the adapter, prepending engine/app-state system hints."""
        from src.shared.python.ai.types import ConversationContext, Message

        temp_messages = list(ctx.messages)
        engine = ctx.metadata.get("last_engine")
        if engine:
            system_hint = (
                f"The user is currently working in the {engine} physics engine. "
                "Tailor your responses to that context when relevant."
            )
            temp_messages.insert(0, Message(role="system", content=system_hint))

        # Inject app state context if a provider is configured
        app_state_msg = self._build_app_state_message()
        if app_state_msg is not None:
            temp_messages.insert(0, app_state_msg)

        return ConversationContext(
            session_id=ctx.session_id,
            messages=temp_messages,
            user_expertise=ctx.user_expertise,
            metadata=ctx.metadata,
        )

    def _cancel_adapter_stream(self) -> None:
        """Force-close the adapter's in-flight stream, if it supports it.

        Adapters opt in via ``BaseAgentAdapter.cancel_stream``. Providers
        without an interruptible transport (CLI-backed adapters, test
        stubs) simply report that there was nothing to cancel.

        Postcondition: never raises -- this runs on the teardown path of an
        already-failing or already-abandoned stream.
        """
        adapter = self._adapter
        if adapter is None:
            return
        cancel = getattr(adapter, "cancel_stream", None)
        if not callable(cancel):
            return
        try:
            cancel()
        except (OSError, RuntimeError, ValueError, AttributeError) as exc:
            logger.debug("ChatService: adapter cancel_stream failed: %s", exc)

    def refresh_models(self) -> dict[str, Any]:
        """Poll the configured provider for available chat models.

        Returns a payload matching the ``ChatModelListResponse`` contract
        (Tools issue #2547 / PR #2566): a list of ``{"name", "provider",
        "display_name"}`` entries plus an ISO-8601 ``refreshed_at``
        timestamp. Failure to reach the provider is logged and yields an
        empty list — the chat session itself stays alive.
        """
        models: list[dict[str, Any]] = []
        provider_id = "unknown"
        adapter = self._adapter
        if adapter is not None:
            provider_id = type(adapter).__name__.replace("Adapter", "").lower()
            list_models = getattr(adapter, "list_available_models", None)
            if callable(list_models):
                try:
                    raw_models = list_models()
                except Exception as exc:  # noqa: BLE001
                    # Adapter may raise transport errors (AIConnectionError,
                    # OSError, httpx errors). Degrade gracefully so the chat
                    # session doesn't drop.
                    logger.warning(
                        "ChatService.refresh_models: %s.list_available_models "
                        "failed: %s",
                        type(adapter).__name__,
                        exc,
                    )
                    raw_models = []
                for entry in raw_models:
                    if isinstance(entry, str):
                        models.append(
                            {
                                "name": entry,
                                "provider": provider_id,
                                "display_name": None,
                            }
                        )
                    elif isinstance(entry, dict):
                        name = str(entry.get("name", ""))
                        if name:
                            models.append(
                                {
                                    "name": name,
                                    "provider": str(entry.get("provider", provider_id)),
                                    "display_name": entry.get("display_name"),
                                }
                            )
        return {
            "models": models,
            "refreshed_at": datetime.now(UTC).isoformat(),
        }

    async def run_codemap_rebuild(self) -> dict[str, Any]:
        """Run the in-tree codemap rebuild and return a status payload.

        Tools issue #2549 / PR #2567: handler for the chat ``index_codebase``
        WebSocket action. Wraps the existing
        ``src.shared.python.codemap.indexer.rebuild`` pathway and returns
        a dict shaped like ``ChatIndexStatusResponse``. Runs in a worker
        thread so the WebSocket loop stays responsive.
        """

        def _rebuild_in_thread() -> dict[str, Any]:
            try:
                from src.shared.python.codemap import discover_repo_root
                from src.shared.python.codemap.indexer import rebuild

                repo_root = discover_repo_root()
                stats = rebuild(repo_root)
                return {
                    "state": "complete",
                    "files_parsed": stats.files_parsed,
                    "symbols_inserted": stats.symbols_inserted,
                    "duration_seconds": float(stats.elapsed_s),
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001
                # Indexer can raise from import failures, sqlite errors, or
                # filesystem permission issues — treat all as soft errors
                # so the chat session stays alive.
                logger.warning(
                    "ChatService.run_codemap_rebuild failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                return {
                    "state": "error",
                    "files_parsed": 0,
                    "symbols_inserted": 0,
                    "duration_seconds": None,
                    "error": str(exc),
                }

        return await asyncio.to_thread(_rebuild_in_thread)

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return message history for a session."""
        with self._lock:
            ctx = self._sessions.get(session_id)
            if not ctx:
                return []
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in ctx.messages
            ]

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all active sessions."""
        with self._lock:
            result = []
            for sid, ctx in self._sessions.items():
                engines = []
                if ctx.metadata.get("last_engine"):
                    engines.append(ctx.metadata["last_engine"])
                result.append(
                    {
                        "session_id": sid,
                        "message_count": len(ctx.messages),
                        "created_at": (
                            ctx.messages[0].timestamp.isoformat()
                            if ctx.messages
                            else ""
                        ),
                        "last_active": (
                            ctx.messages[-1].timestamp.isoformat()
                            if ctx.messages
                            else ""
                        ),
                        "engine_contexts": engines,
                    }
                )
            return result

    def _persist_session(self, session_id: str) -> None:
        """Save session to disk."""
        ctx = self._sessions.get(session_id)
        if not ctx:
            return
        try:
            self.PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            path = self.PERSIST_DIR / f"{session_id}.json"
            ctx.save_to_file(path)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Failed to persist session %s: %s", session_id, e)

    def _load_session(self, session_id: str) -> ConversationContext | None:
        """Load session from disk if it exists."""
        from src.shared.python.ai.types import ConversationContext

        path = self.PERSIST_DIR / f"{session_id}.json"
        if path.exists():
            try:
                return ConversationContext.load_from_file(path)
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("Failed to load session %s: %s", session_id, e)
        return None

    def _cleanup_expired(self) -> None:
        """Evict sessions exceeding TTL or max count."""
        now = time.monotonic()
        expired = [
            sid
            for sid, ts in self._timestamps.items()
            if now - ts > self.SESSION_TTL_SECONDS
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._timestamps.pop(sid, None)

        # Evict oldest if over max
        while len(self._sessions) > self.MAX_SESSIONS:
            oldest_sid, _ = self._sessions.popitem(last=False)
            self._timestamps.pop(oldest_sid, None)


class _StreamWorker:
    """Thread body that pumps adapter output (and tool calls) into a queue.

    Lifted out of the former ``ChatService.stream_response`` closure; all
    collaborators are passed explicitly. ``service`` is read lazily (adapter,
    tool registry, lock, persistence) so late rebinding by callers/tests keeps
    working exactly as it did with the closure.
    """

    def __init__(
        self,
        *,
        service: ChatService,
        ctx: ConversationContext,
        temp_ctx: ConversationContext,
        session_id: str,
        chunk_queue: Any,
        stop_event: threading.Event,
    ) -> None:
        if not session_id:
            raise ValueError("session_id must be a non-empty string")
        self._service = service
        self._ctx = ctx
        self._temp_ctx = temp_ctx
        self._session_id = session_id
        self._queue = chunk_queue
        self._stop = stop_event

    def run(self) -> None:
        """Thread target: stream turns until done, always emit the sentinel."""
        try:
            while not self._stop.is_set():
                if not self._run_turn():
                    break
        # Worker thread must survive any error to send the sentinel and avoid
        # hanging the queue consumer.
        except Exception as e:  # noqa: BLE001
            self._queue.put(f"\n[Error: {e}]")
            get_state_logger().log_exception(e, context="ChatService.stream_response")
        finally:
            self._queue.put(None)  # Sentinel

    def _run_turn(self) -> bool:
        """Run one adapter turn; return True if another turn should follow."""
        current_response, tool_calls_accumulator = self._pull_adapter_chunks()

        if self._stop.is_set():
            # Abandoned mid-stream: do not take the lock to persist
            # messages for a session whose client has gone away.
            return False

        complete_response = "".join(current_response)
        tool_calls = self._build_tool_calls(tool_calls_accumulator)

        if complete_response or tool_calls:
            with self._service._lock:
                self._ctx.add_assistant_message(
                    complete_response, tool_calls=tool_calls
                )
                self._temp_ctx.add_assistant_message(
                    complete_response, tool_calls=tool_calls
                )
                self._service._persist_session(self._session_id)

        get_state_logger().log_event(
            "chat.response_received",
            {"session_id": self._session_id, "chars": len(complete_response)},
        )

        if not tool_calls:
            return False

        self._run_tool_calls(tool_calls)
        return True

    def _pull_adapter_chunks(self) -> tuple[list[str], dict[int, dict[str, Any]]]:
        """Stream one adapter turn into the queue; collect text and tool deltas."""
        current_response: list[str] = []
        tool_calls_accumulator: dict[int, dict[str, Any]] = {}

        for chunk in self._service._adapter.stream_response(  # type: ignore[union-attr]
            "",  # message already in context
            self._temp_ctx,
            self._service._tool_declarations,
        ):
            if self._stop.is_set():
                # Consumer disconnected: stop pulling and let the
                # adapter generator close via its own ``finally``.
                break
            if chunk.content:
                self._queue.put(chunk.content)
                current_response.append(chunk.content)
            if chunk.tool_call_delta:
                _accumulate_tool_call_delta(
                    tool_calls_accumulator, chunk.tool_call_delta
                )
        return current_response, tool_calls_accumulator

    @staticmethod
    def _build_tool_calls(accumulator: dict[int, dict[str, Any]]) -> list[Any]:
        """Convert accumulated tool-call deltas into ``ToolCall`` objects."""
        from src.shared.python.ai.types import ToolCall

        tool_calls = []
        for tc_data in accumulator.values():
            try:
                args = json.loads(tc_data["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"raw": tc_data["arguments"]}
            tool_calls.append(
                ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=args)
            )
        return tool_calls

    def _record_tool_result(self, tool_call_id: str, result_str: str) -> None:
        with self._service._lock:
            self._ctx.add_tool_result(tool_call_id, result_str)
            self._temp_ctx.add_tool_result(tool_call_id, result_str)
            self._service._persist_session(self._session_id)

    def _run_tool_calls(self, tool_calls: list[Any]) -> None:
        """Execute each requested tool, emitting progress events to the queue."""
        for tc in tool_calls:
            if self._stop.is_set():
                self._record_tool_result(
                    tc.id,
                    "Tool execution canceled because the stream client disconnected.",
                )
                continue
            self._queue.put({"type": "tool_call_started", "tool": tc.name})
            result_str, success = self._execute_tool(tc)
            self._queue.put(
                {
                    "type": "tool_call_result" if success else "tool_error",
                    "tool": tc.name,
                    "detail": result_str,
                }
            )
            self._record_tool_result(tc.id, result_str)

    def _execute_tool(self, tc: Any) -> tuple[str, bool]:
        """Run one tool with the configured timeout; return (text, success)."""
        import concurrent.futures

        from src.shared.python.ai.config import get_tool_timeout

        try:
            timeout_sec = get_tool_timeout()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._service._tool_registry.execute, tc.name, tc.arguments
                )
                tool_result = future.result(timeout=timeout_sec)
            result_str = (
                str(tool_result.result)
                if tool_result.success
                else str(tool_result.error)
            )
            return result_str, tool_result.success
        except concurrent.futures.TimeoutError:
            return f"Tool execution timed out after {timeout_sec}s", False
        # Tool execution may raise arbitrary exceptions from user-defined
        # tools; catch-all prevents a single tool from crashing the session.
        except Exception as e:  # noqa: BLE001
            return str(e), False


def _accumulate_tool_call_delta(
    accumulator: dict[int, dict[str, Any]], delta: dict[str, Any]
) -> None:
    """Merge one streamed ``tool_call_delta`` payload into ``accumulator``."""
    for tc in delta.get("tool_calls", []):
        idx = tc.get("index", 0)
        function = tc.get("function", {})
        if idx not in accumulator:
            accumulator[idx] = {
                "id": tc.get("id"),
                "name": function.get("name", ""),
                "arguments": function.get("arguments", ""),
            }
        elif function.get("arguments"):
            accumulator[idx]["arguments"] += function["arguments"]
