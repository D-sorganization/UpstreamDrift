"""Tests for ChatServiceBase shared session management.

Covers:
- Session creation and retrieval
- User message addition with precondition validation
- Message limit (FIFO eviction)
- Session TTL eviction
- Session count limit (LRU eviction)
- Session history retrieval
- Session listing
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from chat.service_base import ChatMessage, ChatServiceBase, ChatSession

# ── Concrete test subclass ───────────────────────────────────────────


class _TestChatService(ChatServiceBase):
    """Minimal concrete subclass for testing."""

    async def stream_response(self, session_id: str) -> AsyncIterator[Any]:
        yield "test response"


# ── ChatSession tests ────────────────────────────────────────────────


class TestChatSession:
    def test_session_has_unique_id(self) -> None:
        s1 = ChatSession()
        s2 = ChatSession()
        assert s1.session_id != s2.session_id

    def test_add_message(self) -> None:
        session = ChatSession()
        msg = session.add_message("user", "hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert session.message_count == 1

    def test_message_count(self) -> None:
        session = ChatSession()
        session.add_message("user", "a")
        session.add_message("assistant", "b")
        assert session.message_count == 2


class TestChatMessage:
    def test_message_fields(self) -> None:
        msg = ChatMessage(role="user", content="test")
        assert msg.role == "user"
        assert msg.content == "test"
        assert msg.tool_call_id is None
        assert msg.metadata == {}

    def test_message_with_tool_call_id(self) -> None:
        msg = ChatMessage(role="tool", content="result", tool_call_id="tc_123")
        assert msg.tool_call_id == "tc_123"


# ── ChatServiceBase tests ───────────────────────────────────────────


class TestChatServiceBase:
    def test_create_session(self) -> None:
        svc = _TestChatService()
        session = svc.get_or_create_session(None)
        assert session is not None
        assert session.session_id.startswith("session_")

    def test_get_existing_session(self) -> None:
        svc = _TestChatService()
        s1 = svc.get_or_create_session(None)
        s2 = svc.get_or_create_session(s1.session_id)
        assert s1.session_id == s2.session_id

    def test_add_user_message(self) -> None:
        svc = _TestChatService()
        session = svc.get_or_create_session(None)
        msg_id = svc.add_user_message(session.session_id, "hello")
        assert msg_id is not None
        assert len(msg_id) == 12
        assert session.message_count == 1

    def test_add_user_message_with_context(self) -> None:
        svc = _TestChatService()
        session = svc.get_or_create_session(None)
        svc.add_user_message(session.session_id, "hello", app_context="gas")
        assert session.metadata["last_context"] == "gas"

    def test_add_user_message_empty_session_id_raises(self) -> None:
        svc = _TestChatService()
        with pytest.raises(ValueError, match="non-empty"):
            svc.add_user_message("", "hello")

    def test_add_user_message_empty_message_raises(self) -> None:
        svc = _TestChatService()
        session = svc.get_or_create_session(None)
        with pytest.raises(ValueError, match="1-10000"):
            svc.add_user_message(session.session_id, "")

    def test_add_user_message_too_long_raises(self) -> None:
        svc = _TestChatService()
        session = svc.get_or_create_session(None)
        with pytest.raises(ValueError, match="1-10000"):
            svc.add_user_message(session.session_id, "x" * 10001)

    def test_add_user_message_unknown_session_raises(self) -> None:
        svc = _TestChatService()
        with pytest.raises(ValueError, match="not found"):
            svc.add_user_message("nonexistent_id", "hello")

    def test_message_limit_eviction(self) -> None:
        svc = _TestChatService()
        svc.MAX_MESSAGES_PER_SESSION = 5
        session = svc.get_or_create_session(None)
        for i in range(7):
            svc.add_user_message(session.session_id, f"msg {i}")
        assert session.message_count == 5
        # Oldest messages evicted
        assert session.messages[0].content == "msg 2"

    def test_session_limit_eviction(self) -> None:
        svc = _TestChatService()
        svc.MAX_SESSIONS = 3
        sessions = []
        for _ in range(5):
            sessions.append(svc.get_or_create_session(None))
        # Only 3 should remain
        assert len(svc._sessions) == 3

    def test_get_session_history(self) -> None:
        svc = _TestChatService()
        session = svc.get_or_create_session(None)
        svc.add_user_message(session.session_id, "hello")
        history = svc.get_session_history(session.session_id)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "hello"

    def test_get_session_history_unknown_session(self) -> None:
        svc = _TestChatService()
        history = svc.get_session_history("nonexistent")
        assert history == []

    def test_list_sessions(self) -> None:
        svc = _TestChatService()
        svc.get_or_create_session(None)
        svc.get_or_create_session(None)
        sessions = svc.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_includes_context(self) -> None:
        svc = _TestChatService()
        session = svc.get_or_create_session(None)
        svc.add_user_message(session.session_id, "test", app_context="gasification")
        sessions = svc.list_sessions()
        assert "gasification" in sessions[0]["app_contexts"]

    def test_ttl_eviction(self) -> None:
        svc = _TestChatService()
        svc.SESSION_TTL_SECONDS = 0  # Immediate expiry
        session = svc.get_or_create_session(None)
        sid = session.session_id
        # Creating a new session triggers cleanup
        svc.get_or_create_session(None)
        assert sid not in svc._sessions

    def test_load_session_hook_returns_none(self) -> None:
        svc = _TestChatService()
        result = svc._load_session("whatever")
        assert result is None

    def test_persist_session_hook_is_noop(self) -> None:
        svc = _TestChatService()
        session = svc.get_or_create_session(None)
        # Should not raise
        svc._persist_session(session.session_id)


@pytest.mark.asyncio
class TestChatServiceStreaming:
    async def test_stream_response(self) -> None:
        svc = _TestChatService()
        session = svc.get_or_create_session(None)
        svc.add_user_message(session.session_id, "hello")
        chunks = []
        async for chunk in svc.stream_response(session.session_id):
            chunks.append(chunk)
        assert chunks == ["test response"]
