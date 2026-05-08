"""Unit tests for the chat WebSocket API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.chat_ws import router


class MockSession:
    def __init__(self, session_id):
        self.session_id = session_id


class MockChatService:
    def get_or_create_session(self, session_id):
        return MockSession(session_id or "new_id")

    def list_sessions(self):
        return [{"session_id": "session_1"}]

    def get_session_history(self, session_id):
        return [{"role": "user", "content": "hello"}]


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


def test_list_sessions(client: TestClient) -> None:
    """Test listing chat sessions."""
    response = client.get("/chat/sessions")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_get_history(client: TestClient) -> None:
    """Test getting chat history."""
    response = client.get("/chat/sessions/session_1/history")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session_1"
    assert len(data["messages"]) == 1


def test_chat_websocket(client: TestClient) -> None:
    """Test WebSocket connection."""
    with client.websocket_connect("/ws/chat/new") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "session_info"
        assert data["session_id"] == "new_id"

        websocket.send_json({"action": "history"})
        data = websocket.receive_json()
        assert data["type"] == "history"
        assert len(data["messages"]) == 1
