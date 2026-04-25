"""Server-side AI chat session manager.

Holds conversation contexts in-memory, delegates AI inference to the
configured adapter (Ollama/OpenAI/Anthropic/Gemini), and persists
sessions to disk for cross-process sharing.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from datetime import timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.shared.python.ai.adapters.base import ToolDeclaration
from src.shared.python.ai.exceptions import ToolExecutionError
from src.shared.python.ai.rag.simple_rag import SimpleRAGStore
from src.shared.python.ai.tool_registry import ToolRegistry
from src.shared.python.core.contracts import precondition
from src.shared.python.core.error_utils import InvalidRequestError
from src.shared.python.logging_pkg.logging_config import get_logger

# Tool-call runtime guards (issue #3162).
TOOL_CALL_TIMEOUT_S = 30.0
MAX_TOOL_CALLS_PER_TURN = 5

# Chunks beginning with this marker carry a JSON-encoded structured event
# ({"type": ..., ...}). Downstream (chat_ws) unpacks and forwards them as
# typed WebSocket frames; plain text chunks are emitted as "chunk" frames.
EVENT_CHUNK_PREFIX = "\x00EVENT\x00"

if TYPE_CHECKING:
    from src.shared.python.ai.adapters.base import BaseAgentAdapter
    from src.shared.python.ai.types import ConversationContext

logger = get_logger(__name__)

UTC = timezone.utc


class ChatService:
    """Server-side chat session manager.

    Manages conversation contexts in-memory with TTL eviction,
    persists to ~/.upstream_modeling_suite/chat_sessions/ on each message,
    and delegates AI inference to the configured adapter.
    """

    MAX_SESSIONS = 50
    SESSION_TTL_SECONDS = 7200  # 2 hours
    PERSIST_DIR = Path.home() / ".upstream_modeling_suite" / "chat_sessions"

    def __init__(self, rag_store: SimpleRAGStore | None = None) -> None:
        self._sessions: OrderedDict[str, ConversationContext] = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self._adapter: BaseAgentAdapter | None = None
        self._lock = threading.Lock()
        self._tool_registry = ToolRegistry()
        self._rag_store: SimpleRAGStore | None = rag_store
        self._load_tool_registry()
        self._load_adapter()

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

    def _load_tool_registry(self) -> None:
        """Register all Golf Suite tools with the registry."""
        try:
            from src.shared.python.ai.sample_tools import register_golf_suite_tools

            register_golf_suite_tools(self._tool_registry)
            logger.info("ChatService: registered %d tools", len(self._tool_registry))
        except (ImportError, RuntimeError) as e:
            logger.warning("ChatService: could not load tool registry: %s", e)

    def _get_tool_declarations(self) -> list[ToolDeclaration]:
        """Convert registry tools to ToolDeclaration objects for adapters."""
        declarations: list[ToolDeclaration] = []
        for tool in self._tool_registry.list_tools():
            props: dict[str, Any] = {}
            required: list[str] = []
            for param in tool.parameters:
                props[param.name] = param.to_json_schema()
                if param.required:
                    required.append(param.name)
            declarations.append(
                ToolDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=props,
                    required=required,
                )
            )
        return declarations

    def _fallback_to_ollama(self) -> None:
        """Fall back to default Ollama adapter."""
        try:
            from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter

            self._adapter = OllamaAdapter()
            logger.info("ChatService using default OllamaAdapter")
        except ImportError as e:
            logger.error("ChatService: could not create fallback adapter: %s", e)

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
            return str(uuid.uuid4().hex[:12])

    @precondition(
        lambda self, session_id: session_id is not None and len(session_id) > 0,
        "Session ID must be a non-empty string",
    )
    async def stream_response(self, session_id: str) -> AsyncIterator[str]:
        """Stream AI response chunks for the latest user message.

        Runs the synchronous adapter in a thread pool executor.
        """
        if not self._adapter:
            yield "I'm not connected to an AI provider. Please configure one in the launcher Settings > AI."
            return

        with self._lock:
            ctx = self._sessions.get(session_id)
            if not ctx:
                yield "Session not found."
                return

        # Build engine context system message
        engine = ctx.metadata.get("last_engine")
        system_parts: list[str] = []
        if engine:
            system_parts.append(
                f"The user is currently working in the {engine} physics engine. "
                "Tailor your responses to that context when relevant."
            )

        # Prepend RAG-retrieved docs relevant to the last user message
        rag_sources: list[dict[str, str]] = []
        if self._rag_store is not None:
            last_user_msg = next(
                (m.content for m in reversed(ctx.messages) if m.role == "user"),
                None,
            )
            if last_user_msg:
                try:
                    rag_results = self._rag_store.query(last_user_msg, top_k=5)
                    if rag_results:
                        snippets = "\n\n".join(
                            f"[{doc.metadata.get('source', doc.id)}]\n{doc.content[:400]}"
                            for doc, _score in rag_results
                        )
                        system_parts.append(
                            "Relevant documentation retrieved for this query:\n\n"
                            + snippets
                        )
                        for doc, _score in rag_results:
                            rag_sources.append(
                                {
                                    "path": doc.metadata.get("source", doc.id),
                                    "excerpt": doc.content[:400],
                                }
                            )
                        logger.debug(
                            "RAG: prepended %d docs for session %s",
                            len(rag_results),
                            session_id,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("RAG query failed: %s", exc)

        if system_parts:
            from src.shared.python.ai.types import Message

            temp_messages = list(ctx.messages)
            temp_messages.insert(
                0, Message(role="system", content="\n\n".join(system_parts))
            )
        else:
            temp_messages = list(ctx.messages)

        # Create a temporary context copy for the adapter
        from src.shared.python.ai.types import ConversationContext

        temp_ctx = ConversationContext(
            session_id=ctx.session_id,
            messages=temp_messages,
            user_expertise=ctx.user_expertise,
            expertise_level=ctx.expertise_level,
            metadata=ctx.metadata,
        )

        full_response: list[str] = []
        tool_declarations = self._get_tool_declarations()

        # Run in thread pool and yield chunks
        # We use a queue-based approach for true streaming
        import json
        import queue

        chunk_queue: queue.Queue[str | None] = queue.Queue()

        # Emit the RAG context event up front (before the model response).
        if rag_sources:
            chunk_queue.put(
                EVENT_CHUNK_PREFIX
                + json.dumps({"type": "context", "sources": rag_sources})
            )

        def _emit_event(event: dict[str, Any]) -> None:
            chunk_queue.put(EVENT_CHUNK_PREFIX + json.dumps(event))

        def _run_tool(tc_id: str, name: str, args: dict[str, Any]) -> tuple[bool, str]:
            """Run a tool under a timeout; return (ok, text)."""
            import concurrent.futures

            def _call() -> Any:
                return self._tool_registry.execute(name, args, tool_call_id=tc_id)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_call)
                try:
                    tool_result = future.result(timeout=TOOL_CALL_TIMEOUT_S)
                except concurrent.futures.TimeoutError as te:
                    _emit_event(
                        {
                            "type": "tool_error",
                            "tool": name,
                            "detail": (f"tool timed out after {TOOL_CALL_TIMEOUT_S}s"),
                        }
                    )
                    future.cancel()
                    return False, f"Tool timeout: {te}"
                except ToolExecutionError as te:
                    logger.warning("Tool not found: %s", name)
                    _emit_event({"type": "tool_error", "tool": name, "detail": str(te)})
                    return False, f"Tool error: {te}"
                except (RuntimeError, ValueError) as exc:
                    _emit_event(
                        {"type": "tool_error", "tool": name, "detail": str(exc)}
                    )
                    return False, f"Tool error: {exc}"

            if tool_result.success:
                _emit_event(
                    {
                        "type": "tool_call_result",
                        "tool": name,
                        "result": tool_result.result,
                    }
                )
                return True, str(tool_result.result)
            _emit_event(
                {
                    "type": "tool_error",
                    "tool": name,
                    "detail": str(tool_result.error),
                }
            )
            return False, f"Tool error: {tool_result.error}"

        def _stream_once(current_ctx: Any) -> list[dict[str, Any]]:
            """Stream one adapter pass; return any tool calls that need running."""
            pending_tool_calls: dict[str, Any] = {}
            tool_call_order: list[str] = []
            for chunk in self._adapter.stream_response(  # type: ignore[union-attr]
                "",
                current_ctx,
                tool_declarations,
            ):
                if chunk.content:
                    chunk_queue.put(chunk.content)
                    full_response.append(chunk.content)
                if chunk.tool_call_delta:
                    delta = chunk.tool_call_delta
                    entries: list[dict[str, Any]] = delta.get("tool_calls") or [delta]
                    for entry in entries:
                        tc_id = entry.get("id", "")
                        if tc_id and tc_id not in pending_tool_calls:
                            pending_tool_calls[tc_id] = {
                                "name": entry.get("name", ""),
                                "arguments_raw": entry.get("arguments", ""),
                            }
                            tool_call_order.append(tc_id)
                        elif tc_id:
                            pending_tool_calls[tc_id]["arguments_raw"] += entry.get(
                                "arguments", ""
                            )
            return [
                {"id": tc_id, **pending_tool_calls[tc_id]} for tc_id in tool_call_order
            ]

        def _stream_to_queue() -> None:
            """Stream adapter chunks; execute tool calls with timeout + loop-guard."""
            try:
                total_calls = 0
                passes = 0
                max_passes = MAX_TOOL_CALLS_PER_TURN + 1
                while passes < max_passes:
                    passes += 1
                    pending = _stream_once(temp_ctx)
                    if not pending:
                        break
                    did_run = False
                    for tc in pending:
                        if total_calls >= MAX_TOOL_CALLS_PER_TURN:
                            _emit_event(
                                {
                                    "type": "tool_error",
                                    "tool": tc["name"],
                                    "detail": "max tool calls exceeded",
                                }
                            )
                            return
                        tc_id = tc["id"]
                        name = tc["name"]
                        try:
                            args = json.loads(tc["arguments_raw"] or "{}")
                        except (ValueError, KeyError):
                            args = {}
                        _emit_event(
                            {
                                "type": "tool_call_started",
                                "tool": name,
                                "args": args,
                            }
                        )
                        ok, text = _run_tool(tc_id, name, args)
                        temp_ctx.add_tool_result(tc_id, text)
                        ctx.add_tool_result(tc_id, text)
                        full_response.append(text)
                        chunk_queue.put(text)
                        total_calls += 1
                        did_run = True
                        logger.info("Tool call executed: %s (ok=%s)", name, ok)
                    if not did_run:
                        break
            except (RuntimeError, ValueError, OSError) as e:
                chunk_queue.put(f"\n[Error: {e}]")
            finally:
                chunk_queue.put(None)  # Sentinel

        thread = threading.Thread(target=_stream_to_queue, daemon=True)
        thread.start()

        while True:
            try:
                item = await asyncio.to_thread(chunk_queue.get, timeout=60.0)
            except (FileNotFoundError, OSError):
                break
            if item is None:
                break
            yield item

        thread.join(timeout=5.0)

        # Save assistant response to context
        complete_response = "".join(full_response)
        if complete_response:
            with self._lock:
                ctx.add_assistant_message(complete_response)
                self._persist_session(session_id)

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
