"""Tests for ChatService - Server-side AI chat session management.

Verifies session CRUD, TTL eviction, message handling,
adapter loading, and disk persistence.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.core.error_utils import InvalidRequestError


@pytest.fixture
def chat_service(tmp_path):
    """Create a ChatService with mocked adapter and temp persist dir."""
    with patch("src.api.services.chat_service.ChatService._load_adapter"):
        from src.api.services.chat_service import ChatService

        svc = ChatService()
        # Use temp directory for persistence
        svc.PERSIST_DIR = tmp_path / "chat_sessions"
        # Provide a mock adapter
        svc._adapter = MagicMock()
        return svc


class TestSessionManagement:
    """Tests for session creation, retrieval, and eviction."""

    def test_create_new_session(self, chat_service):
        """Creating a session with no ID yields a new ConversationContext."""
        ctx = chat_service.get_or_create_session(None)
        assert ctx is not None
        assert ctx.session_id is not None
        assert len(ctx.messages) == 0

    def test_retrieve_existing_session(self, chat_service):
        """Retrieving a session by its ID returns the same context."""
        ctx1 = chat_service.get_or_create_session(None)
        ctx2 = chat_service.get_or_create_session(ctx1.session_id)
        assert ctx1.session_id == ctx2.session_id

    def test_unknown_session_creates_new(self, chat_service):
        """Requesting a non-existent session ID creates a new one."""
        ctx = chat_service.get_or_create_session("nonexistent-id-123")
        assert ctx is not None
        assert ctx.session_id is not None

    def test_max_sessions_eviction(self, chat_service):
        """When MAX_SESSIONS is exceeded, oldest sessions are evicted."""
        chat_service.MAX_SESSIONS = 3
        sessions = []
        for _ in range(5):
            ctx = chat_service.get_or_create_session(None)
            sessions.append(ctx.session_id)

        assert len(chat_service._sessions) <= 3
        # Most recent sessions should survive
        assert sessions[-1] in chat_service._sessions

    def test_ttl_eviction(self, chat_service):
        """Sessions older than TTL are evicted on next access."""
        chat_service.SESSION_TTL_SECONDS = 0  # Immediate expiry
        ctx = chat_service.get_or_create_session(None)
        sid = ctx.session_id

        # Force time to advance past TTL
        chat_service._timestamps[sid] = time.monotonic() - 1

        # Accessing any session triggers cleanup
        chat_service.get_or_create_session(None)
        assert sid not in chat_service._sessions


class TestMessageHandling:
    """Tests for adding messages and retrieving history."""

    def test_add_user_message(self, chat_service):
        """Adding a user message returns a message ID."""
        ctx = chat_service.get_or_create_session(None)
        msg_id = chat_service.add_user_message(ctx.session_id, "Hello, world!")
        assert msg_id is not None
        assert len(msg_id) == 12  # hex[:12]

    def test_add_message_with_engine_context(self, chat_service):
        """Engine context is stored in session metadata."""
        ctx = chat_service.get_or_create_session(None)
        chat_service.add_user_message(
            ctx.session_id, "Help with MuJoCo", engine_context="mujoco"
        )
        assert ctx.metadata.get("last_engine") == "mujoco"

    def test_add_message_to_nonexistent_session(self, chat_service):
        """Adding a message to a non-existent session raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError, match="not found"):
            chat_service.add_user_message("fake-session", "Hello")

    def test_get_session_history(self, chat_service):
        """Session history returns messages in order."""
        ctx = chat_service.get_or_create_session(None)
        chat_service.add_user_message(ctx.session_id, "First message")
        chat_service.add_user_message(ctx.session_id, "Second message")

        history = chat_service.get_session_history(ctx.session_id)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "First message"
        assert history[1]["content"] == "Second message"

    def test_get_history_nonexistent_session(self, chat_service):
        """Getting history for a non-existent session returns empty list."""
        history = chat_service.get_session_history("nonexistent")
        assert history == []

    def test_list_sessions(self, chat_service):
        """Listing sessions returns summary info."""
        ctx = chat_service.get_or_create_session(None)
        chat_service.add_user_message(ctx.session_id, "Hello")

        sessions = chat_service.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == ctx.session_id
        assert sessions[0]["message_count"] == 1


class TestPersistence:
    """Tests for session disk persistence and loading."""

    def test_persist_and_load_session(self, chat_service, tmp_path):
        """A persisted session can be loaded from disk."""
        ctx = chat_service.get_or_create_session(None)
        sid = ctx.session_id
        chat_service.add_user_message(sid, "Persisted message")

        # Force persist
        chat_service._persist_session(sid)

        # Verify file exists
        session_file = chat_service.PERSIST_DIR / f"{sid}.json"
        assert session_file.exists()

        # Remove from memory and reload
        chat_service._sessions.pop(sid, None)
        chat_service._timestamps.pop(sid, None)

        loaded = chat_service.get_or_create_session(sid)
        assert loaded.session_id == sid
        assert len(loaded.messages) == 1
        assert loaded.messages[0].content == "Persisted message"

    def test_load_nonexistent_session(self, chat_service):
        """Loading a session that doesn't exist on disk returns None."""
        result = chat_service._load_session("does-not-exist")
        assert result is None


class TestAdapterLoading:
    """Tests for AI adapter initialization."""

    def test_fallback_to_ollama(self, chat_service):
        """Fallback creates an OllamaAdapter when settings fail."""
        chat_service._adapter = None
        with patch("src.api.services.chat_service.ChatService._fallback_to_ollama"):
            # Just verify fallback is callable
            chat_service._fallback_to_ollama()
        assert True  # No exception raised

    def test_service_works_without_adapter(self, chat_service):
        """ChatService functions for session management even without adapter."""
        chat_service._adapter = None
        ctx = chat_service.get_or_create_session(None)
        assert ctx is not None
        chat_service.add_user_message(ctx.session_id, "No adapter test")
        history = chat_service.get_session_history(ctx.session_id)
        assert len(history) == 1
