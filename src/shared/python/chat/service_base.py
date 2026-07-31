"""Shared base class for AI chat session management.

Provides the common session lifecycle, TTL eviction, and streaming
interface that both UpstreamDrift and Gasification_Model extend.

Design Contracts:
    - ``get_or_create_session`` postcondition: returned session is never None
    - ``add_user_message`` precondition: session_id is non-empty,
      message is 1-10000 chars
    - ``stream_response`` postcondition: yields at least one item or raises

This module has ZERO application-specific imports.
"""

from __future__ import annotations

import abc
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from src.shared.python.chat_contracts.conversation import ArchivedConversationContext

logger = logging.getLogger(__name__)


# ── Session Data ─────────────────────────────────────────────────────


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSession:
    """In-memory chat session with message history.

    Attributes:
        session_id: Unique session identifier.
        messages: Ordered list of messages.
        metadata: Application-specific metadata (e.g., active engine).
        created_at: Monotonic creation time for TTL calculations.
    """

    session_id: str = field(default_factory=lambda: f"session_{uuid.uuid4().hex[:12]}")
    messages: list[ChatMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)

    def add_message(self, role: str, content: str, **kwargs: Any) -> ChatMessage:
        """Add a message to the session.

        Args:
            role: Message role (user, assistant, system, tool).
            content: Message content.
            **kwargs: Additional fields for ChatMessage.

        Returns:
            The newly created ChatMessage.
        """
        msg = ChatMessage(role=role, content=content, **kwargs)
        self.messages.append(msg)
        return msg

    @property
    def message_count(self) -> int:
        """Return the number of messages in this session."""
        return len(self.messages)


def _conversation_context_from_session(session: ChatSession) -> Any:
    """Convert a shared chat session to the AI memory-manager context shape."""
    context = ArchivedConversationContext(
        session_id=session.session_id,
        metadata=dict(session.metadata),
    )
    for message in session.messages:
        context.add_message(
            message.role,
            message.content,
            tool_call_id=message.tool_call_id,
            metadata=dict(message.metadata),
        )
    return context


def _default_memory_manager_factory() -> Any:
    """Lazily construct the optional AI memory manager.

    ``chat`` accepts this runtime dependency as an injectable factory so hosts
    and tests can supply their own memory backend without importing ``ai`` at
    module import time.
    """
    import importlib

    try:
        module = importlib.import_module("src.shared.python.ai.memory_manager")
    except ImportError:
        module = importlib.import_module("ai.memory_manager")
    return module.MemoryManager()


# ── Service Base ─────────────────────────────────────────────────────


class ChatServiceBase(abc.ABC):
    """Abstract base class for chat session management.

    Subclasses must implement ``_create_adapter`` and may override
    ``stream_response`` for provider-specific streaming.

    Configuration is intentionally exposed as class attributes so that
    subclasses can override limits per-application.

    Args:
        max_sessions: Maximum concurrent sessions before LRU eviction.
        session_ttl_seconds: Idle timeout before session eviction.
        max_messages_per_session: Maximum messages before FIFO eviction.
    """

    MAX_SESSIONS: int = 50
    SESSION_TTL_SECONDS: int = 7200  # 2 hours
    MAX_MESSAGES_PER_SESSION: int = 100

    def __init__(
        self,
        *,
        memory_manager_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()
        self._memory_manager_factory = (
            memory_manager_factory or _default_memory_manager_factory
        )

    # ── Session Lifecycle ────────────────────────────────────────────

    def get_or_create_session(self, session_id: str | None) -> ChatSession:
        """Return an existing session or create a new one.

        Postcondition: returned session is never None.

        Args:
            session_id: Existing session ID, or None to create new.

        Returns:
            The resolved or newly created ChatSession.
        """
        with self._lock:
            self._cleanup_expired()

            if session_id and session_id in self._sessions:
                # Move to end of OrderedDict to preserve true LRU order
                self._sessions.move_to_end(session_id)
                self._timestamps[session_id] = time.monotonic()
                return self._sessions[session_id]

            # Try app-specific session loading (e.g., from disk)
            if session_id:
                loaded = self._load_session(session_id)
                if loaded is not None:
                    self._sessions[session_id] = loaded
                    self._timestamps[session_id] = time.monotonic()
                    return loaded

            # Create new session
            session = ChatSession()
            self._sessions[session.session_id] = session
            self._timestamps[session.session_id] = time.monotonic()

            # Evict oldest if over limit
            while len(self._sessions) > self.MAX_SESSIONS:
                oldest_sid, _ = self._sessions.popitem(last=False)
                self._timestamps.pop(oldest_sid, None)
                logger.info("Evicted session %s (LRU limit)", oldest_sid)

            logger.info("Created chat session %s", session.session_id)
            return session

    def add_user_message(
        self,
        session_id: str,
        message: str,
        app_context: str | None = None,
    ) -> str:
        """Add a user message to a session.

        Preconditions:
            - ``session_id`` is a non-empty string
            - ``message`` is 1-10000 characters

        Args:
            session_id: Target session.
            message: User message content.
            app_context: Optional application context hint.

        Returns:
            A unique message ID.

        Raises:
            ValueError: If session not found or preconditions violated.
        """
        if not session_id or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not message or len(message) > 10000:
            raise ValueError("message must be 1-10000 characters")

        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

            if app_context:
                session.metadata["last_context"] = app_context

            session.add_message("user", message)

            # Enforce message limit (FIFO eviction)
            while len(session.messages) > self.MAX_MESSAGES_PER_SESSION:
                session.messages.pop(0)

            self._timestamps[session_id] = time.monotonic()
            self._persist_session(session_id)

        return uuid.uuid4().hex[:12]

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return message history for a session.

        Args:
            session_id: Target session.

        Returns:
            List of message dicts with role, content, timestamp.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                }
                for msg in session.messages
            ]

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all active sessions.

        Returns:
            List of session summary dicts.
        """
        with self._lock:
            result = []
            for sid, session in self._sessions.items():
                contexts: list[str] = []
                if session.metadata.get("last_context"):
                    contexts.append(session.metadata["last_context"])
                result.append(
                    {
                        "session_id": sid,
                        "message_count": session.message_count,
                        "created_at": (
                            str(session.messages[0].timestamp)
                            if session.messages
                            else ""
                        ),
                        "last_active": (
                            str(session.messages[-1].timestamp)
                            if session.messages
                            else ""
                        ),
                        "app_contexts": contexts,
                    }
                )
            return result

    # ── Streaming (must be implemented by subclass) ──────────────────

    @abc.abstractmethod
    async def stream_response(self, session_id: str) -> AsyncIterator[Any]:
        """Stream AI response chunks for the latest user message.

        Postcondition: yields at least one item (content or error).

        Args:
            session_id: Session to generate response for.

        Yields:
            Response chunks (str for text, dict for tool events).
        """
        ...  # pragma: no cover

    async def condense_session(self, session_id: str) -> None:
        """Condense the session history to reduce token usage.

        Default implementation logs a warning and does nothing.
        Subclasses should override with an LLM summarization of the current
        thread and replace the history with the condensed version.

        Args:
            session_id: Target session to condense.
        """
        logger.warning(
            "condense_session not implemented for %s; override in subclass",
            type(self).__name__,
        )

    def condense_to_memory(
        self,
        conversation_ids: list[str],
        *,
        memory_manager: Any | None = None,
    ) -> dict[str, Any]:
        """Extract explicit user memories from selected chat sessions.

        This is the synchronous shared API used by launcher history panels.
        It bridges :class:`ChatServiceBase` sessions into the shared
        ``MemoryManager.digest_archived_contexts`` contract and returns a
        structured result instead of forcing downstream launchers to ship
        deterministic stub responses when the API is absent.

        Args:
            conversation_ids: Session IDs to inspect.
            memory_manager: Optional injected manager for tests or hosts that
                store ``user_memory.json`` outside the default Tools profile.

        Returns:
            Dict containing ``status``, ``requested``, ``processed``,
            ``inserted``, ``missing``, and ``memory_path``.

        Raises:
            TypeError: If ``conversation_ids`` is not a list of strings.
            ValueError: If any session id is blank.
        """
        if not isinstance(conversation_ids, list):
            raise TypeError("conversation_ids must be a list of strings")
        if not all(isinstance(item, str) for item in conversation_ids):
            raise TypeError("conversation_ids must be a list of strings")

        normalized_ids = [item.strip() for item in conversation_ids]
        if any(not item for item in normalized_ids):
            raise ValueError("conversation_ids must contain non-empty strings")

        contexts = []
        missing: list[str] = []
        with self._lock:
            for session_id in normalized_ids:
                session = self._sessions.get(session_id)
                if session is None:
                    session = self._load_session(session_id)
                if session is None:
                    missing.append(session_id)
                    continue
                contexts.append(_conversation_context_from_session(session))

        if memory_manager is None:
            memory_manager = self._memory_manager_factory()

        inserted = memory_manager.digest_archived_contexts(contexts)
        memory_file = getattr(memory_manager, "memory_file", None)
        return {
            "status": "ok" if contexts else "empty",
            "requested": len(normalized_ids),
            "processed": len(contexts),
            "inserted": inserted,
            "missing": missing,
            "memory_path": str(memory_file) if memory_file is not None else "",
        }

    async def execute_skill(self, session_id: str, skill_id: str) -> None:
        """Execute a predefined skill or workflow.

        Default implementation logs a warning and does nothing.
        Subclasses should override to implement skill execution.

        Args:
            session_id: Target session.
            skill_id: ID of the skill to execute.
        """
        logger.warning(
            "execute_skill not implemented for %s; override in subclass",
            type(self).__name__,
        )

    async def request_review(self, session_id: str, provider: str) -> str:
        """Request a multi-agent review of the current thread.

        Default implementation logs a warning and returns the original
        session ID unchanged.  Subclasses should override to spawn a
        dedicated review session.

        Args:
            session_id: Target session.
            provider: The LLM provider to use for the review.

        Returns:
            The session ID of the newly created review session.
        """
        logger.warning(
            "request_review not implemented for %s; override in subclass",
            type(self).__name__,
        )
        return session_id

    async def refresh_models(self) -> list[dict[str, Any]]:
        """Refresh the list of available models from the provider.

        Default implementation raises ``NotImplementedError``.  Subclasses
        that wire a provider with a model-listing API (e.g. Ollama)
        should override.  The router converts this exception into a clear
        ``"refresh_models not supported by this service"`` error reply
        instead of leaving the action silently broken (Tools issue #2751).

        Returns:
            List of model info dicts (matches ``ChatModelInfo``).
        """
        raise NotImplementedError("refresh_models must be implemented by subclass")

    async def index_codebase(self, root_path: str) -> dict[str, Any]:
        """Trigger a re-index of the codebase for RAG.

        Default implementation raises ``NotImplementedError``.  Subclasses
        that bundle an embedding/codemap pipeline should override.  The
        router converts this exception into a clear
        ``"index_codebase not supported by this service"`` error reply
        instead of leaving the action silently broken (Tools issue #2751).

        Args:
            root_path: Filesystem path to the project root.

        Returns:
            Index status dict (matches ``ChatIndexStatusResponse``).
        """
        raise NotImplementedError("index_codebase must be implemented by subclass")

    # ── Hooks for subclass customization ─────────────────────────────

    def _load_session(self, session_id: str) -> ChatSession | None:
        """Load a session from persistent storage.

        Override in subclass to enable disk persistence.

        Args:
            session_id: Session to load.

        Returns:
            Loaded session or None if not found.
        """
        return None

    def _persist_session(self, session_id: str) -> None:
        """Save a session to persistent storage.

        Override in subclass to enable disk persistence.

        Args:
            session_id: Session to persist.
        """
        return  # Default no-op; override in subclass

    # ── Internal ─────────────────────────────────────────────────────

    def _cleanup_expired(self) -> None:
        """Evict sessions exceeding TTL."""
        now = time.monotonic()
        expired = [
            sid
            for sid, ts in self._timestamps.items()
            if now - ts > self.SESSION_TTL_SECONDS
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._timestamps.pop(sid, None)
            logger.info("Evicted session %s (TTL expired)", sid)
