"""Auth coverage for the excluded HTTP routes on WS-mixed routers.

Regression tests for issues #6888 and #6889.

The ``realtime`` and ``chat_ws`` routers are excluded from auto-discovery
(``route_registry._EXCLUDED_MODULES``) because they expose self-authenticating
WebSocket endpoints. Before these fixes their *HTTP* endpoints carried no auth
at all, so in cloud/remote mode an unauthenticated client could:

* ``POST /realtime/publish`` — inject arbitrary JSON to every WS subscriber
  (#6888, HIGH), and
* ``GET /chat/sessions`` / ``GET /chat/sessions/{id}/history`` — enumerate
  active sessions and read other users' history (#6889, MED).

``server.py`` now attaches ``ws_compatible_auth_dependency`` to every mount of
these routers. That dependency enforces the bearer header on HTTP requests only
(respecting ``is_auth_disabled()``); WebSocket connections fall through to the
route handler's own ``resolve_ws_user`` gate.

These tests mount the *real* routers with the *real* dependency, mirroring the
wiring in ``server.py``, and assert:

(a) unauthenticated ``POST /realtime/publish`` → 401/403 in remote mode,
(b) unauthenticated ``GET /chat/sessions`` + ``/history`` → 401/403 in remote
    mode,
(c) auth-disabled / local mode still allows all of the above, and
(d) the WS ``subscribe`` / ``chat_stream`` routes still connect with a valid
    user (the HTTP dependency does not break the socket handshake).
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.routes import chat_ws as chat_ws_module
from src.api.routes import realtime as realtime_module
from src.api.route_registry import ws_compatible_auth_dependency

_REMOTE_ENV = {"GOLF_SUITE_MODE": "remote", "GOLF_AUTH_DISABLED": "false"}
_LOCAL_ENV = {"GOLF_SUITE_MODE": "local"}


class _MockSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


class _MockChatService:
    def list_sessions(self) -> list[dict[str, str]]:
        return [{"session_id": "session_1"}]

    def get_session_history(self, session_id: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": "hello"}]

    def get_or_create_session(self, session_id: str | None) -> _MockSession:
        return _MockSession(session_id or "new_id")


@pytest.fixture
def app() -> FastAPI:
    """App that mounts both excluded routers exactly as ``server.py`` does."""
    deps = [Depends(ws_compatible_auth_dependency)]
    test_app = FastAPI()
    test_app.include_router(chat_ws_module.router, dependencies=deps)
    test_app.include_router(realtime_module.router, dependencies=deps)
    test_app.state.chat_service = _MockChatService()
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    # raise_server_exceptions=False so a rejected dependency surfaces as an
    # HTTP status (401/403) rather than re-raising HTTPException in the test.
    return TestClient(app, raise_server_exceptions=False)


# ── (a) POST /realtime/publish unauthenticated in remote mode (#6888) ──


def test_realtime_publish_unauthenticated_rejected_in_remote_mode(
    client: TestClient,
) -> None:
    with patch.dict(os.environ, _REMOTE_ENV, clear=False):
        response = client.post(
            "/realtime/publish",
            json={"channel": "pose/canonical", "payload": {"spoofed": True}},
        )
    assert response.status_code in (401, 403), response.text


# ── (b) chat REST endpoints unauthenticated in remote mode (#6889) ──


def test_chat_sessions_unauthenticated_rejected_in_remote_mode(
    client: TestClient,
) -> None:
    with patch.dict(os.environ, _REMOTE_ENV, clear=False):
        response = client.get("/chat/sessions")
    assert response.status_code in (401, 403), response.text


def test_chat_history_unauthenticated_rejected_in_remote_mode(
    client: TestClient,
) -> None:
    with patch.dict(os.environ, _REMOTE_ENV, clear=False):
        response = client.get("/chat/sessions/session_1/history")
    assert response.status_code in (401, 403), response.text


# ── (c) auth-disabled / local mode still allows the HTTP endpoints ──


def test_local_mode_allows_chat_sessions(client: TestClient) -> None:
    with patch.dict(os.environ, _LOCAL_ENV, clear=False):
        response = client.get("/chat/sessions")
    assert response.status_code == 200, response.text
    assert response.json() == [{"session_id": "session_1"}]


def test_local_mode_allows_chat_history(client: TestClient) -> None:
    with patch.dict(os.environ, _LOCAL_ENV, clear=False):
        response = client.get("/chat/sessions/session_1/history")
    assert response.status_code == 200, response.text
    assert response.json()["session_id"] == "session_1"


def test_local_mode_allows_realtime_publish(client: TestClient) -> None:
    with patch.dict(os.environ, _LOCAL_ENV, clear=False):
        # No subscribers, but auth must not block the publish itself.
        response = client.post(
            "/realtime/publish",
            json={"channel": "pose/canonical", "payload": {"ok": True}},
        )
    assert response.status_code == 200, response.text
    assert response.json()["delivered"] == 0


# ── (d) WS routes still connect with a valid user despite the HTTP dep ──


def test_realtime_subscribe_ws_connects_with_valid_user(
    client: TestClient,
) -> None:
    """A valid WS subscription must survive the router-level HTTP dependency.

    The dependency declares ``request: Request``; if it were attached without
    WS-scope handling FastAPI would raise ``TypeError`` for the WS scope and
    drop the connection. We patch ``resolve_ws_user`` to stand in for a valid
    token so the test does not need a real JWT or DB.
    """
    fake_user = object()
    # Entering websocket_connect performs the handshake. If the router-level
    # HTTP dependency mis-injected into the WS scope it would raise and the
    # connect would fail with a 1008/close here.
    with (
        patch.dict(os.environ, _REMOTE_ENV, clear=False),
        patch.object(
            realtime_module,
            "resolve_ws_user",
            AsyncMock(return_value=fake_user),
        ),
        client.websocket_connect("/realtime/subscribe?channel=pose/canonical") as ws,
    ):
        ws.close()


def test_chat_stream_ws_connects_with_valid_user(client: TestClient) -> None:
    """The chat WebSocket must still connect through the HTTP auth dependency."""
    fake_user = object()
    with (
        patch.dict(os.environ, _REMOTE_ENV, clear=False),
        patch.object(
            chat_ws_module,
            "resolve_ws_user",
            AsyncMock(return_value=fake_user),
        ),
        client.websocket_connect("/ws/chat/session_1") as ws,
    ):
        data = ws.receive_json()
        assert data["type"] == "session_info"
        assert data["session_id"] == "session_1"
