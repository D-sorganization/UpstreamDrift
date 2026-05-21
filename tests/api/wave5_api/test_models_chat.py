"""Tests for src/api/models/chat.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.models import chat

pytestmark = pytest.mark.unit


def test_style_prompt_known_styles() -> None:
    for s in ("concise", "standard", "detailed"):
        out = chat.style_prompt(s)
        assert isinstance(out, str)
        assert out


def test_style_prompt_unknown_falls_back_to_default() -> None:
    out = chat.style_prompt("nonsense")
    assert out == chat.RESPONSE_STYLE_PROMPTS[chat.DEFAULT_RESPONSE_STYLE]


def test_style_prompt_none_falls_back() -> None:
    assert chat.style_prompt(None) == chat.RESPONSE_STYLE_PROMPTS["standard"]


def test_chat_message_request_default_style() -> None:
    req = chat.ChatMessageRequest(message="hello")
    assert req.response_style == "standard"


def test_chat_message_request_explicit_response_style_wins() -> None:
    req = chat.ChatMessageRequest(
        message="hi", response_style="concise", expertise_level="beginner"
    )
    assert req.response_style == "concise"


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("beginner", "detailed"),
        ("INTERMEDIATE", "standard"),
        ("advanced", "concise"),
        ("Expert", "concise"),
    ],
)
def test_chat_message_legacy_expertise_back_fill(level: str, expected: str) -> None:
    req = chat.ChatMessageRequest(message="hi", expertise_level=level)
    assert req.response_style == expected


def test_chat_message_unknown_legacy_keeps_default() -> None:
    req = chat.ChatMessageRequest(message="hi", expertise_level="alien")
    # Unknown mapping should leave default in place
    assert req.response_style == "standard"


def test_chat_message_requires_min_length() -> None:
    with pytest.raises(ValidationError):
        chat.ChatMessageRequest(message="")


def test_chat_message_max_length_enforced() -> None:
    with pytest.raises(ValidationError):
        chat.ChatMessageRequest(message="x" * 10001)


def test_chat_model_info_round_trip() -> None:
    info = chat.ChatModelInfo(name="llama3", provider="ollama", display_name="L3")
    assert info.model_dump()["name"] == "llama3"


def test_chat_model_list_response_default_models() -> None:
    resp = chat.ChatModelListResponse(refreshed_at="2024-01-01T00:00:00Z")
    assert resp.models == []


def test_chat_chunk_defaults() -> None:
    c = chat.ChatChunkResponse(content="x")
    assert c.is_final is False
    assert c.index == 0


def test_chat_session_info_default_contexts() -> None:
    s = chat.ChatSessionInfo(
        session_id="s", message_count=0, created_at="a", last_active="b"
    )
    assert s.engine_contexts == []


def test_chat_history_response_messages_required() -> None:
    h = chat.ChatHistoryResponse(session_id="s", messages=[{"role": "user"}])
    assert h.messages[0]["role"] == "user"


def test_chat_index_status_states() -> None:
    s = chat.ChatIndexStatusResponse(state="running")
    assert s.files_parsed == 0
    s2 = chat.ChatIndexStatusResponse(
        state="complete", files_parsed=5, symbols_inserted=10, duration_seconds=1.5
    )
    assert s2.duration_seconds == 1.5
