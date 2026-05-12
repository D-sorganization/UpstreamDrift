"""Tests for chat API block models."""

import pytest
from pydantic import ValidationError
from src.api.models.chat import (
    DEFAULT_RESPONSE_STYLE,
    RESPONSE_STYLE_PROMPTS,
    ChatChunkResponse,
    ChatHistoryResponse,
    ChatIndexStatusResponse,
    ChatMessageRequest,
    ChatModelInfo,
    ChatModelListResponse,
    ChatSessionInfo,
    style_prompt,
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


# Tools issue #2552 / PR #2568: response_style field + helpers.


def test_chat_message_request_default_response_style() -> None:
    """A bare request defaults to ``standard`` (Tools #2552)."""
    req = ChatMessageRequest(message="hi")
    assert req.response_style == "standard"
    assert DEFAULT_RESPONSE_STYLE == "standard"


def test_chat_message_request_explicit_response_style() -> None:
    """Caller-supplied ``response_style`` is preserved verbatim."""
    req = ChatMessageRequest(message="hi", response_style="concise")
    assert req.response_style == "concise"


def test_chat_message_request_rejects_unknown_style() -> None:
    """Pydantic rejects values outside the ResponseStyle Literal."""
    with pytest.raises(ValidationError):
        ChatMessageRequest(message="hi", response_style="enthusiastic")


def test_legacy_expertise_level_maps_to_style() -> None:
    """Legacy ``expertise_level`` back-fills ``response_style`` (#2552)."""
    assert (
        ChatMessageRequest(message="hi", expertise_level="beginner").response_style
        == "detailed"
    )
    assert (
        ChatMessageRequest(message="hi", expertise_level="intermediate").response_style
        == "standard"
    )
    assert (
        ChatMessageRequest(message="hi", expertise_level="advanced").response_style
        == "concise"
    )
    assert (
        ChatMessageRequest(message="hi", expertise_level="expert").response_style
        == "concise"
    )


def test_explicit_response_style_overrides_legacy_expertise() -> None:
    """When both fields are set, ``response_style`` wins."""
    req = ChatMessageRequest(
        message="hi",
        response_style="detailed",
        expertise_level="advanced",
    )
    assert req.response_style == "detailed"


def test_response_style_prompts_keys() -> None:
    """The prompt table covers exactly the three styles."""
    assert set(RESPONSE_STYLE_PROMPTS.keys()) == {"concise", "standard", "detailed"}
    for value in RESPONSE_STYLE_PROMPTS.values():
        assert isinstance(value, str) and value


def test_style_prompt_unknown_falls_back() -> None:
    """``style_prompt`` returns the default prompt for unknown / None values."""
    default = RESPONSE_STYLE_PROMPTS[DEFAULT_RESPONSE_STYLE]
    assert style_prompt(None) == default
    assert style_prompt("not-a-real-style") == default
    assert style_prompt("concise") == RESPONSE_STYLE_PROMPTS["concise"]


# Tools issue #2547 / PR #2566: ChatModelInfo + ChatModelListResponse.


def test_chat_model_info_minimal() -> None:
    """ChatModelInfo only needs name + provider (Tools #2547)."""
    info = ChatModelInfo(name="llama3.1:8b", provider="ollama")
    assert info.name == "llama3.1:8b"
    assert info.provider == "ollama"
    assert info.display_name is None


def test_chat_model_info_with_display_name() -> None:
    """ChatModelInfo carries an optional display_name."""
    info = ChatModelInfo(
        name="gpt-4o",
        provider="openai",
        display_name="GPT-4o",
    )
    assert info.display_name == "GPT-4o"


def test_chat_model_list_response_default_empty() -> None:
    """ChatModelListResponse defaults models to an empty list."""
    resp = ChatModelListResponse(refreshed_at="2026-05-11T00:00:00+00:00")
    assert resp.models == []
    assert resp.refreshed_at.startswith("2026-05-11")


def test_chat_model_list_response_with_models() -> None:
    """ChatModelListResponse carries a list of ChatModelInfo entries."""
    resp = ChatModelListResponse(
        refreshed_at="2026-05-11T00:00:00+00:00",
        models=[
            ChatModelInfo(name="llama3.1:8b", provider="ollama"),
            ChatModelInfo(name="mistral", provider="ollama"),
        ],
    )
    assert len(resp.models) == 2
    assert resp.models[0].name == "llama3.1:8b"


def test_chat_index_status_response_running() -> None:
    """ChatIndexStatusResponse defaults numeric fields to 0 (Tools #2549)."""
    resp = ChatIndexStatusResponse(state="running")
    assert resp.state == "running"
    assert resp.files_parsed == 0
    assert resp.symbols_inserted == 0
    assert resp.duration_seconds is None
    assert resp.error is None


def test_chat_index_status_response_complete() -> None:
    """ChatIndexStatusResponse carries totals + duration when complete."""
    resp = ChatIndexStatusResponse(
        state="complete",
        files_parsed=42,
        symbols_inserted=320,
        duration_seconds=1.25,
    )
    assert resp.state == "complete"
    assert resp.files_parsed == 42
    assert resp.symbols_inserted == 320
    assert resp.duration_seconds == 1.25
    assert resp.error is None


def test_chat_index_status_response_error() -> None:
    """ChatIndexStatusResponse carries an error string on state='error'."""
    resp = ChatIndexStatusResponse(state="error", error="permission denied")
    assert resp.state == "error"
    assert resp.error == "permission denied"
