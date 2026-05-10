"""Shared base class for AI chat session management.

Provides the common session lifecycle, TTL eviction, and streaming
interface that both UpstreamDrift and Gasification_Model extend.

Design Contracts:
    - ``get_or_create_session`` postcondition: returned session is never None
    - ``add_user_message`` precondition: session_id is non-empty, message is 1-10000 chars
    - ``stream_response`` postcondition: yields at least one item or raises

This module has ZERO application-specific imports.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

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

    def __init__(self) -> None:
        self._sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()

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
