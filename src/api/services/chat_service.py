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
from typing import TYPE_CHECKING

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

    def __init__(self) -> None:
        self._sessions: OrderedDict[str, ConversationContext] = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self._adapter: BaseAgentAdapter | None = None
        self._lock = threading.Lock()
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
        lambda self, session_id, message, engine_context=None: session_id is not None
        and len(session_id) > 0,
        "Session ID must be a non-empty string",
    )
    @precondition(
        lambda self, session_id, message, engine_context=None: message is not None
        and len(message) > 0,
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
        if engine:
            system_hint = (
                f"The user is currently working in the {engine} physics engine. "
                "Tailor your responses to that context when relevant."
            )
            # Temporarily inject system context (don't persist it)
            from src.shared.python.ai.types import Message

            temp_messages = list(ctx.messages)
            temp_messages.insert(0, Message(role="system", content=system_hint))
        else:
            temp_messages = list(ctx.messages)

        # Create a temporary context copy for the adapter
        from src.shared.python.ai.types import ConversationContext

        temp_ctx = ConversationContext(
            session_id=ctx.session_id,
            messages=temp_messages,
            user_expertise=ctx.user_expertise,
            metadata=ctx.metadata,
        )

        full_response: list[str] = []

        def _run_sync() -> list[str]:
            """Run synchronous adapter streaming in thread."""
            chunks: list[str] = []
            for chunk in self._adapter.stream_response(  # type: ignore[union-attr]
                temp_ctx.messages[-1].content if temp_ctx.messages else "",
                temp_ctx,
                [],  # No tools for now
            ):
                if chunk.content:
                    chunks.append(chunk.content)
            return chunks

        # Run in thread pool and yield chunks
        # We use a queue-based approach for true streaming
        import queue

        chunk_queue: queue.Queue[str | None] = queue.Queue()

        def _stream_to_queue() -> None:
            try:
                for chunk in self._adapter.stream_response(  # type: ignore[union-attr]
                    "",  # message already in context
                    temp_ctx,
                    [],
                ):
                    if chunk.content:
                        chunk_queue.put(chunk.content)
                        full_response.append(chunk.content)
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

    def get_session_history(self, session_id: str) -> list[dict]:
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

    def list_sessions(self) -> list[dict]:
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
