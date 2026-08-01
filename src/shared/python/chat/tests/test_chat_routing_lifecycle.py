# ruff: noqa: E501
"""Value-asserting chat tests for session lifecycle, history, and routing.

Covers issue #3184: session lifecycle, conversation-history accumulation, and
WebSocket message routing/dispatch were thinly covered relative to the module
count. These tests drive the real ``create_chat_router`` against a concrete
``ChatServiceBase`` subclass (no mocks of the unit under test) so the assertions
observe real routed payloads and accumulated history, plus exercise the
``chat.models`` contract objects end to end.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from chat.models import (
    DEFAULT_RESPONSE_STYLE,
    ChatChunkResponse,
    ChatHistoryResponse,
    ChatMessageRequest,
    ChatModelInfo,
    ChatModelListResponse,
    ChatSessionInfo,
    make_full_thinking_capabilities,
    make_none_only_capabilities,
    style_prompt,
)
from chat.router_factory import create_chat_router
from chat.service_base import ChatServiceBase
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Concrete service used as the real unit under test for routing.
# ---------------------------------------------------------------------------


class _EchoChatService(ChatServiceBase):
    """Minimal concrete service that streams a deterministic echo response.

    Records every streamed-for session so tests can assert routing reached
    the service rather than mocking the dispatch layer away.
    """

    def __init__(self) -> None:
        super().__init__()
        self.streamed_for: list[str] = []

    async def stream_response(self, session_id: str) -> AsyncIterator[Any]:
        self.streamed_for.append(session_id)
        history = self.get_session_history(session_id)
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"),
            "",
        )
        yield f"echo: {last_user}"
        # Record the assistant turn so history accumulation is observable.
        session = self._sessions[session_id]
        session.add_message("assistant", f"echo: {last_user}")


@pytest.fixture
def service() -> _EchoChatService:
    return _EchoChatService()


@pytest.fixture
def client(service: _EchoChatService) -> TestClient:
    app = FastAPI()
    app.state.chat_service = service
    app.include_router(create_chat_router())
    return TestClient(app)


# ---------------------------------------------------------------------------
# Session lifecycle + message routing through the real WebSocket router.
# ---------------------------------------------------------------------------


class TestWebSocketRouting:
    def test_new_connection_assigns_session(self, client: TestClient) -> None:
        """Connecting with 'new' yields a generated session id."""
        with client.websocket_connect("/ws/chat/new") as ws:
            info = ws.receive_json()
            assert info["type"] == "session_info"
            assert info["session_id"].startswith("session_")

    def test_send_routes_to_stream_and_completes(
        self, client: TestClient, service: _EchoChatService
    ) -> None:
        """A 'send' action routes to stream_response and emits chunk+complete."""
        with client.websocket_connect("/ws/chat/new") as ws:
            session_id = ws.receive_json()["session_id"]
            ws.send_json({"action": "send", "message": "hello world"})

            chunk = ws.receive_json()
            assert chunk == {"type": "chunk", "content": "echo: hello world"}

            complete = ws.receive_json()
            assert complete == {"type": "complete", "session_id": session_id}

        # Routing actually reached the concrete service for this session.
        assert service.streamed_for == [session_id]

    def test_history_accumulates_user_and_assistant(self, client: TestClient) -> None:
        """History action returns accumulated user + assistant turns in order."""
        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()  # session_info
            ws.send_json({"action": "send", "message": "first"})
            ws.receive_json()  # chunk
            ws.receive_json()  # complete

            ws.send_json({"action": "history"})
            history = ws.receive_json()

        assert history["type"] == "history"
        roles = [m["role"] for m in history["messages"]]
        contents = [m["content"] for m in history["messages"]]
        assert roles == ["user", "assistant"]
        assert contents == ["first", "echo: first"]

    def test_empty_message_is_rejected_without_streaming(
        self, client: TestClient, service: _EchoChatService
    ) -> None:
        """Blank 'send' message routes to an error, never to the service."""
        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()
            ws.send_json({"action": "send", "message": "   "})
            err = ws.receive_json()
        assert err == {"type": "error", "detail": "Empty message"}
        assert service.streamed_for == []

    def test_new_session_action_rotates_session_id(self, client: TestClient) -> None:
        """'new_session' issues a fresh session id distinct from the first."""
        with client.websocket_connect("/ws/chat/new") as ws:
            first = ws.receive_json()["session_id"]
            ws.send_json({"action": "new_session"})
            created = ws.receive_json()
        assert created["type"] == "session_created"
        assert created["session_id"] != first

    def test_unknown_action_returns_error(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()
            ws.send_json({"action": "frobnicate"})
            err = ws.receive_json()
        assert err["type"] == "error"
        assert "Unknown action" in err["detail"]

    def test_reconnect_to_existing_session_reuses_history(
        self, client: TestClient
    ) -> None:
        """Reconnecting with a known id resumes the same session history."""
        with client.websocket_connect("/ws/chat/new") as ws:
            session_id = ws.receive_json()["session_id"]
            ws.send_json({"action": "send", "message": "persist me"})
            ws.receive_json()
            ws.receive_json()

        with client.websocket_connect(f"/ws/chat/{session_id}") as ws2:
            resumed = ws2.receive_json()
            assert resumed["session_id"] == session_id
            ws2.send_json({"action": "history"})
            history = ws2.receive_json()
        assert any(m["content"] == "persist me" for m in history["messages"])


class TestRestFallback:
    def test_list_sessions_endpoint(
        self, client: TestClient, service: _EchoChatService
    ) -> None:
        session = service.get_or_create_session(None)
        service.add_user_message(session.session_id, "hi")
        resp = client.get("/chat/sessions")
        assert resp.status_code == 200
        ids = [s["session_id"] for s in resp.json()]
        assert session.session_id in ids

    def test_history_endpoint_blank_id_returns_empty(self, client: TestClient) -> None:
        resp = client.get("/chat/sessions/%20/history")
        assert resp.status_code == 200
        assert resp.json()["messages"] == []


class TestRoutedActionDispatch:
    """Exercise the remaining router action branches against the default service."""

    def test_condense_action_returns_history(self, client: TestClient) -> None:
        """'condense' uses the default no-op condense then echoes history."""
        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()
            ws.send_json({"action": "condense"})
            resp = ws.receive_json()
        assert resp["type"] == "history"
        assert resp["messages"] == []

    def test_skill_invoke_missing_id_errors(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()
            ws.send_json({"action": "skill_invoke"})
            err = ws.receive_json()
        assert err == {"type": "error", "detail": "Missing skill_id"}

    def test_skill_invoke_default_noop_returns_history(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()
            ws.send_json({"action": "skill_invoke", "skill_id": "noop"})
            resp = ws.receive_json()
        assert resp["type"] == "history"

    def test_request_review_missing_provider_errors(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()
            ws.send_json({"action": "request_review"})
            err = ws.receive_json()
        assert err == {"type": "error", "detail": "Missing provider"}

    def test_request_review_default_returns_same_session(
        self, client: TestClient
    ) -> None:
        with client.websocket_connect("/ws/chat/new") as ws:
            session_id = ws.receive_json()["session_id"]
            ws.send_json({"action": "request_review", "provider": "openai"})
            resp = ws.receive_json()
        assert resp["type"] == "review_started"
        assert resp["new_session_id"] == session_id

    def test_refresh_models_not_supported(self, client: TestClient) -> None:
        """Default service raises NotImplementedError -> structured error reply."""
        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()
            ws.send_json({"action": "refresh_models"})
            err = ws.receive_json()
        assert err["type"] == "error"
        assert "refresh_models not supported" in err["detail"]

    def test_index_codebase_not_supported(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()
            ws.send_json({"action": "index_codebase", "root_path": "."})
            err = ws.receive_json()
        assert err["type"] == "error"
        assert "index_codebase not supported" in err["detail"]


# ---------------------------------------------------------------------------
# chat.models — contract value objects.
# ---------------------------------------------------------------------------


class TestChatModels:
    def test_request_defaults_response_style(self) -> None:
        req = ChatMessageRequest(message="hi")
        assert req.response_style == DEFAULT_RESPONSE_STYLE
        assert req.app_context is None

    def test_expertise_level_backfills_response_style(self) -> None:
        """Legacy expertise_level maps onto response_style when unset."""
        req = ChatMessageRequest(message="hi", expertise_level="advanced")
        assert req.response_style == "concise"

    def test_explicit_response_style_wins_over_expertise(self) -> None:
        req = ChatMessageRequest(
            message="hi", expertise_level="beginner", response_style="concise"
        )
        assert req.response_style == "concise"

    def test_message_min_length_enforced(self) -> None:
        with pytest.raises(ValueError):
            ChatMessageRequest(message="")

    def test_style_prompt_known_and_fallback(self) -> None:
        assert "concisely" in style_prompt("concise").lower()
        # Unknown -> default standard fragment.
        assert style_prompt("bogus") == style_prompt(DEFAULT_RESPONSE_STYLE)

    def test_chunk_response_defaults(self) -> None:
        chunk = ChatChunkResponse(content="abc")
        assert chunk.index == 0
        assert chunk.is_final is False

    def test_session_info_and_history_round_trip(self) -> None:
        info = ChatSessionInfo(
            session_id="s1",
            message_count=2,
            created_at="t0",
            last_active="t1",
        )
        assert info.app_contexts == []
        history = ChatHistoryResponse(
            session_id="s1", messages=[{"role": "user", "content": "x"}]
        )
        assert history.messages[0]["content"] == "x"

    def test_model_list_response(self) -> None:
        info = ChatModelInfo(name="m", provider="ollama")
        assert info.available is True
        resp = ChatModelListResponse(models=[info], refreshed_at="2026-01-01T00:00:00Z")
        assert resp.models[0].provider == "ollama"


class TestThinkingCapabilities:
    def test_none_only_capabilities(self) -> None:
        caps = make_none_only_capabilities("openai")
        assert caps.level_names() == ("none",)
        assert caps.find_level("none") is not None
        assert caps.find_level("high") is None

    def test_full_capabilities_default_and_lookup(self) -> None:
        caps = make_full_thinking_capabilities("anthropic", default_level_name="medium")
        assert caps.default_level_name == "medium"
        assert caps.level_names() == ("none", "low", "medium", "high")
        high = caps.find_level("high")
        assert high is not None
        assert high.budget_tokens == 16384

    def test_invalid_default_level_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_full_thinking_capabilities("x", default_level_name="extreme")
