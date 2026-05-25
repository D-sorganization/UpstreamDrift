"""Tests for chat WebSocket and REST endpoints.

Uses FastAPI TestClient to verify the WebSocket protocol
and REST fallback endpoints.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from src.api.routes import chat_ws

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    """Use asyncio backend only (trio not installed)."""
    return "asyncio"


@pytest.fixture(autouse=True)
def local_mode_env():
    """Run all chat WebSocket tests in local mode (auth bypassed by design)."""
    with patch.dict(os.environ, {"GOLF_SUITE_MODE": "local"}):
        yield


@pytest.fixture
def mock_chat_service() -> MagicMock:
    """Create a mock ChatService."""
    svc = MagicMock()

    # Mock ConversationContext
    mock_ctx = MagicMock()
    mock_ctx.session_id = "test-session-123"
    mock_ctx.messages = []
    mock_ctx.metadata = {}

    svc.get_or_create_session.return_value = mock_ctx
    svc.add_user_message.return_value = "msg-abc123"
    svc.get_session_history.return_value = [
        {
            "role": "user",
            "content": "Hello",
            "timestamp": "2026-01-01T00:00:00",
        }
    ]
    svc.refresh_models.return_value = {
        "models": [
            {"name": "llama3.1:8b", "provider": "ollama", "display_name": None},
            {"name": "mistral", "provider": "ollama", "display_name": None},
        ],
        "refreshed_at": "2026-05-11T00:00:00+00:00",
    }
    svc.list_sessions.return_value = [
        {
            "session_id": "test-session-123",
            "message_count": 1,
            "created_at": "2026-01-01T00:00:00",
            "last_active": "2026-01-01T00:00:00",
            "engine_contexts": ["mujoco"],
        }
    ]

    # Make stream_response an async generator
    async def mock_stream(session_id: str) -> AsyncGenerator[str, None]:
        yield "Hello "
        yield "world!"

    svc.stream_response = mock_stream

    # Tools #2549: async run_codemap_rebuild needs an awaitable mock.
    async def mock_rebuild() -> dict:
        return {
            "state": "complete",
            "files_parsed": 7,
            "symbols_inserted": 42,
            "duration_seconds": 0.5,
            "error": None,
        }

    svc.run_codemap_rebuild = mock_rebuild

    return svc


@pytest.fixture
def app(mock_chat_service: MagicMock) -> FastAPI:
    """Create a FastAPI app with chat routes."""
    test_app = FastAPI()
    test_app.state.chat_service = mock_chat_service
    test_app.include_router(chat_ws.router, prefix="/api")
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


class TestWebSocket:
    """Tests for the WebSocket chat endpoint."""

    def test_connect_new_session(self, client, mock_chat_service) -> None:
        """Connecting with 'new' creates a new session."""
        with client.websocket_connect("/api/ws/chat/new") as ws:
            data = ws.receive_json()
            assert data["type"] == "session_info"
            assert data["session_id"] == "test-session-123"

        mock_chat_service.get_or_create_session.assert_called_with(None)

    def test_connect_existing_session(self, client, mock_chat_service) -> None:
        """Connecting with an existing session ID retrieves it."""
        with client.websocket_connect("/api/ws/chat/test-session-123") as ws:
            data = ws.receive_json()
            assert data["type"] == "session_info"
            assert data["session_id"] == "test-session-123"

        mock_chat_service.get_or_create_session.assert_called_with("test-session-123")

    def test_send_message_and_stream(self, client, mock_chat_service) -> None:
        """Sending a message streams response chunks then complete."""
        with client.websocket_connect("/api/ws/chat/new") as ws:
            # Consume session_info
            ws.receive_json()

            ws.send_json(
                {
                    "action": "send",
                    "message": "Hello AI",
                    "engine_context": "mujoco",
                }
            )

            # Should receive chunks
            chunk1 = ws.receive_json()
            assert chunk1["type"] == "chunk"
            assert chunk1["content"] == "Hello "

            chunk2 = ws.receive_json()
            assert chunk2["type"] == "chunk"
            assert chunk2["content"] == "world!"

            # Should receive complete
            complete = ws.receive_json()
            assert complete["type"] == "complete"
            assert complete["session_id"] == "test-session-123"

    def test_send_empty_message(self, client) -> None:
        """Sending an empty message returns an error."""
        with client.websocket_connect("/api/ws/chat/new") as ws:
            ws.receive_json()  # session_info

            ws.send_json({"action": "send", "message": ""})
            error = ws.receive_json()
            assert error["type"] == "error"
            assert "Empty" in error["detail"]

    def test_history_action(self, client, mock_chat_service) -> None:
        """Requesting history returns messages."""
        with client.websocket_connect("/api/ws/chat/new") as ws:
            ws.receive_json()  # session_info

            ws.send_json({"action": "history"})
            data = ws.receive_json()
            assert data["type"] == "history"
            assert len(data["messages"]) == 1
            assert data["messages"][0]["content"] == "Hello"

    def test_new_session_action(self, client, mock_chat_service) -> None:
        """Requesting new_session creates a fresh session."""
        with client.websocket_connect("/api/ws/chat/new") as ws:
            ws.receive_json()  # session_info

            ws.send_json({"action": "new_session"})
            data = ws.receive_json()
            assert data["type"] == "session_created"
            assert "session_id" in data

    def test_refresh_models_action(self, client, mock_chat_service) -> None:
        """``refresh_models`` returns a ``model_list`` payload (Tools #2547)."""
        with client.websocket_connect("/api/ws/chat/new") as ws:
            ws.receive_json()  # session_info

            ws.send_json({"action": "refresh_models"})
            data = ws.receive_json()
            assert data["type"] == "model_list"
            assert isinstance(data["models"], list)
            assert len(data["models"]) == 2
            assert data["models"][0]["name"] == "llama3.1:8b"
            assert data["models"][0]["provider"] == "ollama"
            assert "refreshed_at" in data

        mock_chat_service.refresh_models.assert_called()

    def test_index_codebase_action(self, client, mock_chat_service) -> None:
        """``index_codebase`` ships a 'running' then a 'complete' status (Tools #2549)."""
        with client.websocket_connect("/api/ws/chat/new") as ws:
            ws.receive_json()  # session_info

            ws.send_json({"action": "index_codebase"})
            running = ws.receive_json()
            assert running["type"] == "index_status"
            assert running["state"] == "running"
            assert running["files_parsed"] == 0

            done = ws.receive_json()
            assert done["type"] == "index_status"
            assert done["state"] == "complete"
            assert done["files_parsed"] == 7
            assert done["symbols_inserted"] == 42

    def test_unknown_action(self, client) -> None:
        """Sending an unknown action returns an error."""
        with client.websocket_connect("/api/ws/chat/new") as ws:
            ws.receive_json()  # session_info

            ws.send_json({"action": "invalid_action"})
            error = ws.receive_json()
            assert error["type"] == "error"
            assert "Unknown action" in error["detail"]

    def test_streaming_error_is_sanitized_and_socket_stays_open(
        self, client, mock_chat_service, caplog
    ) -> None:
        """Unexpected streaming failures should keep tracebacks server-side."""

        async def broken_stream(session_id: str) -> AsyncGenerator[str, None]:
            yield "partial"
            raise RuntimeError("token=super-secret")

        mock_chat_service.stream_response = broken_stream

        with (
            caplog.at_level("ERROR"),
            client.websocket_connect("/api/ws/chat/new") as ws,
        ):
            ws.receive_json()  # session_info
            ws.send_json({"action": "send", "message": "Hello AI"})

            chunk = ws.receive_json()
            assert chunk == {"type": "chunk", "content": "partial"}

            error = ws.receive_json()
            assert error == {"type": "error", "detail": "Internal server error"}

            ws.send_json({"action": "history"})
            history = ws.receive_json()
            assert history["type"] == "history"

        assert any(
            record.message == "Error during streaming response"
            and record.exc_info is not None
            for record in caplog.records
        )

    async def test_connection_error_is_sanitized(
        self, mock_chat_service, caplog
    ) -> None:
        """Transport failures should not leak raw exception details to clients."""

        class FakeWebSocket:
            def __init__(self, chat_service: MagicMock) -> None:
                self.app = SimpleNamespace(
                    state=SimpleNamespace(chat_service=chat_service)
                )
                self.url = SimpleNamespace(path="/ws/chat/new")
                self.sent: list[dict[str, object]] = []
                self._receive_calls = 0

            async def accept(self) -> None:
                return None

            async def send_json(self, payload: dict[str, object]) -> None:
                self.sent.append(payload)

            async def receive_json(self) -> dict[str, object]:
                self._receive_calls += 1
                raise OSError("session=secret-session")

        websocket = FakeWebSocket(mock_chat_service)

        with caplog.at_level("ERROR"):
            await chat_ws.chat_stream(websocket, "new")

        assert websocket.sent[0] == {
            "type": "session_info",
            "session_id": "test-session-123",
        }
        assert websocket.sent[-1] == {"type": "error", "detail": "Connection error"}
        assert any(
            record.message == "Chat WebSocket connection error"
            and record.exc_info is not None
            for record in caplog.records
        )

    async def test_disconnect_log_uses_session_token(
        self, mock_chat_service, caplog
    ) -> None:
        """Disconnect logs must not expose the raw session identifier."""

        sensitive_session_id = "golfer@example.com"
        mock_chat_service.get_or_create_session.return_value.session_id = (
            sensitive_session_id
        )

        class FakeWebSocket:
            def __init__(self, chat_service: MagicMock) -> None:
                self.app = SimpleNamespace(
                    state=SimpleNamespace(chat_service=chat_service)
                )
                self.url = SimpleNamespace(path="/ws/chat/new")
                self.sent: list[dict[str, object]] = []

            async def accept(self) -> None:
                return None

            async def send_json(self, payload: dict[str, object]) -> None:
                self.sent.append(payload)

            async def receive_json(self) -> dict[str, object]:
                raise WebSocketDisconnect

        websocket = FakeWebSocket(mock_chat_service)
        expected_token = chat_ws._session_log_token(sensitive_session_id)

        with caplog.at_level("DEBUG"):
            await chat_ws.chat_stream(websocket, sensitive_session_id)  # type: ignore[arg-type]

        assert websocket.sent[0] == {
            "type": "session_info",
            "session_id": sensitive_session_id,
        }
        assert any(
            record.message
            == f"Chat WebSocket disconnected: session_token={expected_token}"
            for record in caplog.records
        )
        assert all(
            sensitive_session_id not in record.message for record in caplog.records
        )


class TestRESTEndpoints:
    """Tests for the REST fallback endpoints."""

    def test_chat_ws_list_sessions(self, client, mock_chat_service) -> None:
        """GET /chat/sessions returns session list."""
        response = client.get("/api/chat/sessions")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["session_id"] == "test-session-123"
        assert data[0]["engine_contexts"] == ["mujoco"]

    def test_chat_ws_get_history(self, client, mock_chat_service) -> None:
        """GET /chat/sessions/{id}/history returns messages."""
        response = client.get("/api/chat/sessions/test-session-123/history")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"
