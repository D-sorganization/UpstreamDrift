"""Origin and launcher-token guard tests for local WebSockets (#7275)."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.unit]


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


class _GuardWebSocket:
    def __init__(
        self,
        *,
        origin: str | None = "http://localhost:5173",
        query_token: str | None = "expected-token",
        protocol: str | None = None,
        expected_token: str = "expected-token",
    ) -> None:
        self.headers: dict[str, str] = {}
        if origin is not None:
            self.headers["origin"] = origin
        if protocol is not None:
            self.headers["sec-websocket-protocol"] = protocol
        self.query_params: dict[str, str] = {}
        if query_token is not None:
            self.query_params["launcher_token"] = query_token
        self.url = MagicMock()
        self.url.path = "/ws/test"
        self.app = SimpleNamespace(
            state=SimpleNamespace(launcher_csrf_token=expected_token)
        )
        self.close_called = False
        self.close_code: int | None = None

    async def close(self, code: int = 1000) -> None:
        self.close_called = True
        self.close_code = code


async def test_accepts_loopback_origin_with_query_token() -> None:
    from src.api.auth.middleware import LocalUser
    from src.api.auth.ws_auth import resolve_ws_user

    ws: Any = _GuardWebSocket(origin="http://127.0.0.1:3000")
    with patch.dict(os.environ, {"GOLF_SUITE_MODE": "local"}, clear=True):
        user = await resolve_ws_user(ws)

    assert isinstance(user, LocalUser)
    assert not ws.close_called


async def test_accepts_loopback_origin_with_protocol_token() -> None:
    from src.api.auth.middleware import LocalUser
    from src.api.auth.ws_auth import resolve_ws_user

    ws: Any = _GuardWebSocket(
        origin="http://localhost:3000",
        query_token=None,
        protocol="upstreamdrift.v1, launcher-token.expected-token",
    )
    with patch.dict(os.environ, {"GOLF_SUITE_MODE": "local"}, clear=True):
        user = await resolve_ws_user(ws)

    assert isinstance(user, LocalUser)
    assert not ws.close_called


async def test_rejects_external_origin_before_accept() -> None:
    from src.api.auth.ws_auth import resolve_ws_user

    ws: Any = _GuardWebSocket(origin="https://attacker.example")
    with patch.dict(os.environ, {"GOLF_SUITE_MODE": "local"}, clear=True):
        user = await resolve_ws_user(ws)

    assert user is None
    assert ws.close_called
    assert ws.close_code == 1008


async def test_rejects_missing_launcher_token_before_accept() -> None:
    from src.api.auth.ws_auth import resolve_ws_user

    ws: Any = _GuardWebSocket(query_token=None)
    with patch.dict(os.environ, {"GOLF_SUITE_MODE": "local"}, clear=True):
        user = await resolve_ws_user(ws)

    assert user is None
    assert ws.close_called
    assert ws.close_code == 1008


async def test_rejects_invalid_launcher_token_before_accept() -> None:
    from src.api.auth.ws_auth import resolve_ws_user

    ws: Any = _GuardWebSocket(query_token="wrong-token")
    with patch.dict(os.environ, {"GOLF_SUITE_MODE": "local"}, clear=True):
        user = await resolve_ws_user(ws)

    assert user is None
    assert ws.close_called
    assert ws.close_code == 1008
