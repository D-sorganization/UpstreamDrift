"""Tests for chat API block models."""

import pytest
from pydantic import ValidationError

from src.api.models.chat import (
    ChatChunkResponse,
    ChatHistoryResponse,
    ChatMessageRequest,
    ChatSessionInfo,
)


def test_chat_message_request_valid() -> None:
    """Test valid chat message requests."""
    req = ChatMessageRequest(message="hello")
    assert req.message == "hello"
    assert req.engine_context is None
    assert req.expertise_level == "beginner"

    req2 = ChatMessageRequest(
        message="help me with drake",
        engine_context="drake",
        expertise_level="advanced",
    )
    assert req2.engine_context == "drake"


def test_chat_message_request_invalid() -> None:
    """Test invalid requests."""
    # Empty message
    with pytest.raises(ValidationError):
        ChatMessageRequest(message="")


def test_chat_chunk_response() -> None:
    """Test creating a chunk response."""
    resp = ChatChunkResponse(content="chunk")
    assert resp.content == "chunk"
    assert not resp.is_final
    assert resp.index == 0


def test_chat_session_info() -> None:
    """Test creating session info."""
    info = ChatSessionInfo(
        session_id="123",
        message_count=5,
        created_at="2026",
        last_active="2026",
    )
    assert info.session_id == "123"
    assert len(info.engine_contexts) == 0


def test_chat_history_response() -> None:
    """Test creating history response."""
    history = ChatHistoryResponse(
        session_id="123",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert history.session_id == "123"
    assert len(history.messages) == 1
