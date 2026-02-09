"""Tests for chat WebSocket and REST endpoints.

Uses FastAPI TestClient to verify the WebSocket protocol
and REST fallback endpoints.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import chat_ws

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend():
    """Use asyncio backend only (trio not installed)."""
    return "asyncio"


@pytest.fixture
def mock_chat_service():
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
    async def mock_stream(session_id):
        yield "Hello "
        yield "world!"

    svc.stream_response = mock_stream

    return svc


@pytest.fixture
def app(mock_chat_service):
    """Create a FastAPI app with chat routes."""
    test_app = FastAPI()
    test_app.state.chat_service = mock_chat_service
    test_app.include_router(chat_ws.router, prefix="/api")
    return test_app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestWebSocket:
    """Tests for the WebSocket chat endpoint."""

    def test_connect_new_session(self, client, mock_chat_service):
        """Connecting with 'new' creates a new session."""
        with client.websocket_connect("/api/ws/chat/new") as ws:
            data = ws.receive_json()
            assert data["type"] == "session_info"
            assert data["session_id"] == "test-session-123"

        mock_chat_service.get_or_create_session.assert_called_with(None)

    def test_connect_existing_session(self, client, mock_chat_service):
        """Connecting with an existing session ID retrieves it."""
        with client.websocket_connect("/api/ws/chat/test-session-123") as ws:
            data = ws.receive_json()
            assert data["type"] == "session_info"
            assert data["session_id"] == "test-session-123"

        mock_chat_service.get_or_create_session.assert_called_with("test-session-123")

    def test_send_message_and_stream(self, client, mock_chat_service):
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

    def test_send_empty_message(self, client):
        """Sending an empty message returns an error."""
        with client.websocket_connect("/api/ws/chat/new") as ws:
            ws.receive_json()  # session_info

            ws.send_json({"action": "send", "message": ""})
            error = ws.receive_json()
            assert error["type"] == "error"
            assert "Empty" in error["detail"]

    def test_history_action(self, client, mock_chat_service):
        """Requesting history returns messages."""
        with client.websocket_connect("/api/ws/chat/new") as ws:
            ws.receive_json()  # session_info

            ws.send_json({"action": "history"})
            data = ws.receive_json()
            assert data["type"] == "history"
            assert len(data["messages"]) == 1
            assert data["messages"][0]["content"] == "Hello"

    def test_new_session_action(self, client, mock_chat_service):
        """Requesting new_session creates a fresh session."""
        with client.websocket_connect("/api/ws/chat/new") as ws:
            ws.receive_json()  # session_info

            ws.send_json({"action": "new_session"})
            data = ws.receive_json()
            assert data["type"] == "session_created"
            assert "session_id" in data

    def test_unknown_action(self, client):
        """Sending an unknown action returns an error."""
        with client.websocket_connect("/api/ws/chat/new") as ws:
            ws.receive_json()  # session_info

            ws.send_json({"action": "invalid_action"})
            error = ws.receive_json()
            assert error["type"] == "error"
            assert "Unknown action" in error["detail"]


class TestRESTEndpoints:
    """Tests for the REST fallback endpoints."""

    def test_list_sessions(self, client, mock_chat_service):
        """GET /chat/sessions returns session list."""
        response = client.get("/api/chat/sessions")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["session_id"] == "test-session-123"
        assert data[0]["engine_contexts"] == ["mujoco"]

    def test_get_history(self, client, mock_chat_service):
        """GET /chat/sessions/{id}/history returns messages."""
        response = client.get("/api/chat/sessions/test-session-123/history")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"
