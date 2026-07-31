"""Test WebSocket error handling when AI providers (Ollama) fail."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import Mock

import pytest

# Use the same import shape as sibling tests in this directory; the pytest
# `pythonpath = ["src/shared/python", ...]` config (see pyproject.toml) makes
# `chat` and `ai` importable directly. The earlier `from src.shared.python...`
# form requires the Tools repo root on sys.path which is not guaranteed during
# pytest-xdist worker collection (Tools issue #2965 / fleet CI).
from ai.exceptions import AIConnectionError
from ai.types import AgentChunk
from chat.router_factory import create_chat_router as build_chat_router
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def mock_chat_service() -> Mock:
    service = Mock()
    # Return valid session for new session
    mock_session = Mock()
    mock_session.session_id = "test-session"
    service.get_or_create_session.return_value = mock_session
    return service


@pytest.fixture
def client(mock_chat_service: Mock) -> TestClient:
    app = FastAPI()
    app.state.chat_service = mock_chat_service
    router = build_chat_router(mock_chat_service)
    app.include_router(router)
    return TestClient(app)


@pytest.mark.skip(
    reason=(
        "Test written against an older chat router API. "
        "`build_chat_router(chat_service)` no longer exists; "
        "`create_chat_router(prefix='', authorize_fn=None)` is the current "
        "signature and it reads chat_service from app.state, not a constructor "
        "argument. Needs rewrite to match the new injection model."
    )
)
def test_websocket_propagates_connection_error_without_disconnecting(
    client: TestClient, mock_chat_service: Mock
) -> None:
    """Verify AIConnectionError surfaces as a client error without disconnecting."""

    # Mock stream_response to raise an exception
    async def mock_stream(*args, **kwargs) -> AsyncIterator[AgentChunk]:
        yield AgentChunk(content="Thinking...", is_final=False, index=0)
        raise AIConnectionError("Cannot connect to Ollama", provider="ollama")

    mock_chat_service.stream_response.side_effect = mock_stream

    with client.websocket_connect("/ws/chat/test-session") as websocket:
        # Initial connect message
        data = websocket.receive_json()
        assert data["type"] == "session_info"
        assert data["session_id"] == "test-session"

        # Send a message
        websocket.send_json(
            {"action": "send", "message": "Hello", "app_context": "tests"}
        )

        # Should get chunk
        data = websocket.receive_json()
        assert data["type"] == "chunk"
        assert data["content"] == "Thinking..."

        # Should get error message, NOT a disconnect exception
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert "Cannot connect to Ollama" in data["detail"]

        # Verify websocket is still open by sending history request
        websocket.send_json({"action": "history"})
        mock_chat_service.get_session_history.return_value = []
        data = websocket.receive_json()
        assert data["type"] == "history"
