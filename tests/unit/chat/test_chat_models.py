"""Tests for src.shared.python.chat.models (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.shared.python.chat.models import (
    ChatChunkResponse,
    ChatHistoryResponse,
    ChatMessageRequest,
    ChatSessionInfo,
)

# ---------------------------------------------------------------------------
# ChatMessageRequest
# ---------------------------------------------------------------------------


class TestChatMessageRequest:
    def test_chat_models_basic_construction(self) -> None:
        req = ChatMessageRequest(message="hello")
        assert req.message == "hello"

    def test_default_expertise_level(self) -> None:
        req = ChatMessageRequest(message="hello")
        assert req.expertise_level == "beginner"

    def test_app_context_optional(self) -> None:
        req = ChatMessageRequest(message="hello")
        assert req.app_context is None

    def test_app_context_set(self) -> None:
        req = ChatMessageRequest(message="hello", app_context="mujoco")
        assert req.app_context == "mujoco"

    def test_expertise_level_set(self) -> None:
        req = ChatMessageRequest(message="hello", expertise_level="advanced")
        assert req.expertise_level == "advanced"

    def test_empty_message_raises(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessageRequest(message="")

    def test_message_too_long_raises(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessageRequest(message="x" * 10001)


# ---------------------------------------------------------------------------
# ChatChunkResponse
# ---------------------------------------------------------------------------


class TestChatChunkResponse:
    def test_chat_models_basic_construction(self) -> None:
        chunk = ChatChunkResponse(content="hello")
        assert chunk.content == "hello"

    def test_default_is_final_false(self) -> None:
        chunk = ChatChunkResponse(content="hi")
        assert chunk.is_final is False

    def test_default_index_zero(self) -> None:
        chunk = ChatChunkResponse(content="hi")
        assert chunk.index == 0

    def test_is_final_set(self) -> None:
        chunk = ChatChunkResponse(content="done", is_final=True)
        assert chunk.is_final is True

    def test_index_set(self) -> None:
        chunk = ChatChunkResponse(content="hi", index=3)
        assert chunk.index == 3


# ---------------------------------------------------------------------------
# ChatSessionInfo
# ---------------------------------------------------------------------------


class TestChatSessionInfo:
    def test_chat_models_basic_construction(self) -> None:
        info = ChatSessionInfo(
            session_id="abc",
            message_count=5,
            created_at="2024-01-01",
            last_active="2024-01-02",
        )
        assert info.session_id == "abc"
        assert info.message_count == 5

    def test_default_app_contexts_empty(self) -> None:
        info = ChatSessionInfo(
            session_id="x",
            message_count=0,
            created_at="2024-01-01",
            last_active="2024-01-01",
        )
        assert info.app_contexts == []


# ---------------------------------------------------------------------------
# ChatHistoryResponse
# ---------------------------------------------------------------------------


class TestChatHistoryResponse:
    def test_chat_models_basic_construction(self) -> None:
        resp = ChatHistoryResponse(
            session_id="s1", messages=[{"role": "user", "content": "hi"}]
        )
        assert resp.session_id == "s1"
        assert len(resp.messages) == 1

    def test_empty_messages(self) -> None:
        resp = ChatHistoryResponse(session_id="s2", messages=[])
        assert resp.messages == []
