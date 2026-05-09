"""Unit tests for the chat WebSocket API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.chat_ws import router


class MockSession:
    def __init__(self, session_id):
        self.session_id = session_id


class MockChatService:
    def __init__(self):
        self.add_message_error = False
        self.stream_chunks = [{"type": "chunk", "content": "mock "}, "response"]

    def get_or_create_session(self, session_id):
        return MockSession(session_id or "new_id")

    def list_sessions(self):
        return [{"session_id": "session_1"}]

    def get_session_history(self, session_id):
        return [{"role": "user", "content": "hello"}]

    def add_user_message(self, session_id, message, engine_context):
        if self.add_message_error:
            raise ValueError("Test error")

    async def stream_response(self, session_id):
        for chunk in self.stream_chunks:
            yield chunk


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with the chat router."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.state.chat_service = MockChatService()
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_routes_chat_ws_list_sessions(client: TestClient) -> None:
    """Test listing chat sessions."""
    response = client.get("/chat/sessions")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_routes_chat_ws_get_history(client: TestClient) -> None:
    """Test getting chat history."""
    response = client.get("/chat/sessions/session_1/history")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session_1"
    assert len(data["messages"]) == 1


def test_chat_websocket_new_session_flow(client: TestClient) -> None:
    """Test WebSocket connection and the new_session action."""
    with client.websocket_connect("/ws/chat/new") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "session_info"
        assert data["session_id"] == "new_id"

        websocket.send_json({"action": "new_session"})
        data = websocket.receive_json()
        assert data["type"] == "session_created"
        assert data["session_id"] == "new_id"


def test_chat_websocket_history_flow(client: TestClient) -> None:
    """Test getting history through WebSocket."""
    with client.websocket_connect("/ws/chat/session_1") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "session_info"
        assert data["session_id"] == "session_1"

        websocket.send_json({"action": "history"})
        data = websocket.receive_json()
        assert data["type"] == "history"
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "hello"


def test_chat_websocket_send_flow(client: TestClient) -> None:
    """Test sending a message and streaming the response."""
    with client.websocket_connect("/ws/chat/session_1") as websocket:
        # Ignore session info
        websocket.receive_json()

        # Test empty message
        websocket.send_json({"action": "send", "message": "   "})
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["detail"] == "Empty message"

        # Test valid message
        websocket.send_json(
            {"action": "send", "message": "hello", "engine_context": "mujoco"}
        )

        # Receive chunk 1 (dict)
        data = websocket.receive_json()
        assert data["type"] == "chunk"
        assert data["content"] == "mock "

        # Receive chunk 2 (string)
        data = websocket.receive_json()
        assert data["type"] == "chunk"
        assert data["content"] == "response"

        # Receive complete
        data = websocket.receive_json()
        assert data["type"] == "complete"
        assert data["session_id"] == "session_1"


def test_chat_websocket_send_error(client: TestClient, app: FastAPI) -> None:
    """Test sending a message when add_user_message raises an error."""
    app.state.chat_service.add_message_error = True
    with client.websocket_connect("/ws/chat/session_1") as websocket:
        websocket.receive_json()
        websocket.send_json({"action": "send", "message": "hello"})
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["detail"] == "Test error"


def test_chat_websocket_unknown_action(client: TestClient) -> None:
    """Test sending an unknown action."""
    with client.websocket_connect("/ws/chat/session_1") as websocket:
        websocket.receive_json()
        websocket.send_json({"action": "unknown_action"})
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert "Unknown action: unknown_action" in data["detail"]


def test_chat_websocket_disconnect(client: TestClient) -> None:
    """Test that disconnect is handled silently."""
    with client.websocket_connect("/ws/chat/session_1") as websocket:
        websocket.receive_json()
        websocket.close()
