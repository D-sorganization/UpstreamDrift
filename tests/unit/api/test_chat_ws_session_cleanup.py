"""WebSocket session-cleanup tests for the chat router (issue #7150).

A connection that creates an ephemeral ("new") session must release it on
disconnect so per-connection state does not accumulate.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.shared.python.chat.router_factory import create_chat_router

pytestmark = pytest.mark.unit


class _Session:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


class _CountingChatService:
    """Minimal chat service that tracks live sessions like the real base."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._counter = 0

    def get_or_create_session(self, session_id: str | None) -> _Session:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        if session_id is None:
            self._counter += 1
            session_id = f"sess-{self._counter}"
        session = _Session(session_id)
        self._sessions[session_id] = session
        return session

    def end_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def session_count(self) -> int:
        return len(self._sessions)

    def list_sessions(self) -> list[dict[str, Any]]:
        return [{"session_id": sid} for sid in self._sessions]

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        return []


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(create_chat_router())
    app.state.chat_service = _CountingChatService()
    return TestClient(app)


def test_new_session_is_released_on_disconnect(client: TestClient) -> None:
    service = client.app.state.chat_service
    baseline = service.session_count()

    for _ in range(5):
        with client.websocket_connect("/ws/chat/new") as ws:
            info = ws.receive_json()
            assert info["type"] == "session_info"
        # context exit closes the socket -> disconnect -> cleanup

    assert service.session_count() == baseline


def test_existing_session_is_not_destroyed_on_disconnect(client: TestClient) -> None:
    service = client.app.state.chat_service
    # Pre-create a durable session the client reconnects to.
    service.get_or_create_session("durable-1")
    before = service.session_count()

    with client.websocket_connect("/ws/chat/durable-1") as ws:
        ws.receive_json()

    # The reconnected (non-ephemeral) session must survive disconnect.
    assert service.session_count() == before
    assert service.end_session("durable-1") is True
