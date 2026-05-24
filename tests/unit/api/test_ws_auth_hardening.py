"""TDD tests for WebSocket authentication hardening (issue #5913).

Covers:
1. resolve_ws_user() — auth gate called before accept()
2. _clamp_speed_factor() — rejects non-positive speed_factor from clients
3. set_speed command propagates validated value (zero/negative → default)
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Mock WebSocket that records close() calls
# ---------------------------------------------------------------------------


class _MockWebSocket:
    def __init__(self, auth_header: str = "") -> None:
        self.headers: dict[str, str] = {}
        if auth_header:
            self.headers["authorization"] = auth_header
        from unittest.mock import MagicMock

        self.url = MagicMock()
        self.url.path = "/ws/test"
        self.close_called = False
        self.close_code: int | None = None

    async def close(self, code: int = 1000) -> None:
        self.close_called = True
        self.close_code = code


# ---------------------------------------------------------------------------
# 1. resolve_ws_user — local mode (every connection admitted, WARNING logged)
# ---------------------------------------------------------------------------


class TestResolveWsUserLocalMode:
    """In local/auth-disabled mode, connections are admitted with a WARNING."""

    async def test_returns_local_user(self) -> None:
        from src.api.auth.middleware import LocalUser
        from src.api.auth.ws_auth import resolve_ws_user

        ws = _MockWebSocket()
        with patch.dict(os.environ, {"GOLF_SUITE_MODE": "local"}):
            user = await resolve_ws_user(ws)

        assert isinstance(user, LocalUser)

    async def test_does_not_close_connection(self) -> None:
        from src.api.auth.ws_auth import resolve_ws_user

        ws = _MockWebSocket()
        with patch.dict(os.environ, {"GOLF_SUITE_MODE": "local"}):
            await resolve_ws_user(ws)

        assert not ws.close_called

    async def test_logs_warning(self) -> None:
        from src.api.auth.ws_auth import resolve_ws_user

        ws = _MockWebSocket()
        with (
            patch.dict(os.environ, {"GOLF_SUITE_MODE": "local"}),
            patch("src.api.auth.ws_auth.logger") as mock_logger,
        ):
            await resolve_ws_user(ws)

        mock_logger.warning.assert_called_once()
        message = mock_logger.warning.call_args[0][0]
        assert "local" in message.lower() or "auth-disabled" in message.lower()


# ---------------------------------------------------------------------------
# 2. resolve_ws_user — cloud mode (missing / invalid token → 1008)
# ---------------------------------------------------------------------------


class TestResolveWsUserCloudMode:
    """Cloud mode requires a valid Bearer token; failures close with code 1008."""

    def _cloud_env(self) -> dict[str, str]:
        return {"GOLF_SUITE_MODE": "cloud", "GOLF_AUTH_DISABLED": "false"}

    async def test_no_auth_header_returns_none(self) -> None:
        from src.api.auth.ws_auth import resolve_ws_user

        ws = _MockWebSocket(auth_header="")
        with patch.dict(os.environ, self._cloud_env(), clear=True):
            result = await resolve_ws_user(ws)

        assert result is None

    async def test_no_auth_header_closes_with_1008(self) -> None:
        from src.api.auth.ws_auth import resolve_ws_user

        ws = _MockWebSocket(auth_header="")
        with patch.dict(os.environ, self._cloud_env(), clear=True):
            await resolve_ws_user(ws)

        assert ws.close_called
        assert ws.close_code == 1008

    async def test_non_bearer_scheme_closes_with_1008(self) -> None:
        from src.api.auth.ws_auth import resolve_ws_user

        ws = _MockWebSocket(auth_header="Basic dXNlcjpwYXNz")
        with patch.dict(os.environ, self._cloud_env(), clear=True):
            result = await resolve_ws_user(ws)

        assert result is None
        assert ws.close_code == 1008

    async def test_invalid_token_closes_with_1008(self) -> None:
        from fastapi import HTTPException

        from src.api.auth.ws_auth import resolve_ws_user

        ws = _MockWebSocket(auth_header="Bearer badtoken")
        with (
            patch.dict(os.environ, self._cloud_env(), clear=True),
            patch("src.api.auth.ws_auth.security_manager") as mock_sm,
        ):
            mock_sm.verify_token.side_effect = HTTPException(
                status_code=401, detail="bad"
            )
            result = await resolve_ws_user(ws)

        assert result is None
        assert ws.close_called
        assert ws.close_code == 1008

    async def test_valid_token_returns_local_user(self) -> None:
        from src.api.auth.middleware import LocalUser

        from src.api.auth.ws_auth import resolve_ws_user

        ws = _MockWebSocket(auth_header="Bearer validtoken")
        with (
            patch.dict(os.environ, self._cloud_env(), clear=True),
            patch("src.api.auth.ws_auth.security_manager") as mock_sm,
        ):
            mock_sm.verify_token.return_value = {"sub": "42", "type": "access"}
            result = await resolve_ws_user(ws)

        assert isinstance(result, LocalUser)
        assert not ws.close_called


# ---------------------------------------------------------------------------
# 3. _clamp_speed_factor — unit tests
# ---------------------------------------------------------------------------


class TestClampSpeedFactor:
    """_clamp_speed_factor must reject zero, negative, NaN, Inf values."""

    def test_positive_float_accepted(self) -> None:
        from src.api.routes.simulation_ws import _clamp_speed_factor

        assert _clamp_speed_factor(2.5) == pytest.approx(2.5)

    def test_integer_accepted(self) -> None:
        from src.api.routes.simulation_ws import _clamp_speed_factor

        assert _clamp_speed_factor(3) == pytest.approx(3.0)

    def test_zero_uses_default(self) -> None:
        from src.api.routes.simulation_ws import (
            _DEFAULT_SPEED_FACTOR,
            _clamp_speed_factor,
        )

        assert _clamp_speed_factor(0.0) == pytest.approx(_DEFAULT_SPEED_FACTOR)

    def test_negative_uses_default(self) -> None:
        from src.api.routes.simulation_ws import (
            _DEFAULT_SPEED_FACTOR,
            _clamp_speed_factor,
        )

        assert _clamp_speed_factor(-1.0) == pytest.approx(_DEFAULT_SPEED_FACTOR)

    def test_nan_uses_default(self) -> None:
        from src.api.routes.simulation_ws import (
            _DEFAULT_SPEED_FACTOR,
            _clamp_speed_factor,
        )

        assert _clamp_speed_factor(float("nan")) == pytest.approx(_DEFAULT_SPEED_FACTOR)

    def test_inf_uses_default(self) -> None:
        from src.api.routes.simulation_ws import (
            _DEFAULT_SPEED_FACTOR,
            _clamp_speed_factor,
        )

        assert _clamp_speed_factor(float("inf")) == pytest.approx(_DEFAULT_SPEED_FACTOR)

    def test_non_numeric_string_uses_default(self) -> None:
        from src.api.routes.simulation_ws import (
            _DEFAULT_SPEED_FACTOR,
            _clamp_speed_factor,
        )

        assert _clamp_speed_factor("fast") == pytest.approx(_DEFAULT_SPEED_FACTOR)

    def test_numeric_string_accepted(self) -> None:
        from src.api.routes.simulation_ws import _clamp_speed_factor

        assert _clamp_speed_factor("3.0") == pytest.approx(3.0)

    def test_none_uses_default(self) -> None:
        from src.api.routes.simulation_ws import (
            _DEFAULT_SPEED_FACTOR,
            _clamp_speed_factor,
        )

        assert _clamp_speed_factor(None) == pytest.approx(_DEFAULT_SPEED_FACTOR)


# ---------------------------------------------------------------------------
# 4. set_speed command — validates via _handle_client_commands
# ---------------------------------------------------------------------------


class _Stats:
    def __init__(self, speed_factor: float) -> None:
        self.speed_factor = speed_factor


class _SimulationService:
    def __init__(self, speed_factor: float) -> None:
        self.stats = _Stats(speed_factor)


class _AppState:
    def __init__(self, speed_factor: float) -> None:
        self.simulation_service = _SimulationService(speed_factor)


class _App:
    def __init__(self, speed_factor: float) -> None:
        self.state = _AppState(speed_factor)


class _WS:
    def __init__(self, speed_factor: float = 1.0) -> None:
        self.app = _App(speed_factor)


class TestSetSpeedValidation:
    """set_speed action must not store non-positive speed_factor values."""

    async def test_zero_speed_uses_default(self) -> None:
        from src.api.routes.simulation_ws import (
            _DEFAULT_SPEED_FACTOR,
            _handle_client_commands,
        )

        ws: Any = _WS(speed_factor=2.0)

        async def recv() -> dict[str, Any]:
            return {"action": "set_speed", "speed_factor": 0.0}

        ws.receive_json = recv  # type: ignore[attr-defined]
        config: dict[str, Any] = {"speed_factor": 2.0}
        await _handle_client_commands(ws, config)

        assert config["speed_factor"] == pytest.approx(_DEFAULT_SPEED_FACTOR)
        assert ws.app.state.simulation_service.stats.speed_factor == pytest.approx(
            _DEFAULT_SPEED_FACTOR
        )

    async def test_negative_speed_uses_default(self) -> None:
        from src.api.routes.simulation_ws import (
            _DEFAULT_SPEED_FACTOR,
            _handle_client_commands,
        )

        ws: Any = _WS(speed_factor=2.0)

        async def recv() -> dict[str, Any]:
            return {"action": "set_speed", "speed_factor": -99.0}

        ws.receive_json = recv  # type: ignore[attr-defined]
        config: dict[str, Any] = {"speed_factor": 2.0}
        await _handle_client_commands(ws, config)

        assert config["speed_factor"] == pytest.approx(_DEFAULT_SPEED_FACTOR)

    async def test_positive_speed_accepted(self) -> None:
        from src.api.routes.simulation_ws import _handle_client_commands

        ws: Any = _WS(speed_factor=1.0)

        async def recv() -> dict[str, Any]:
            return {"action": "set_speed", "speed_factor": 4.5}

        ws.receive_json = recv  # type: ignore[attr-defined]
        config: dict[str, Any] = {"speed_factor": 1.0}
        await _handle_client_commands(ws, config)

        assert config["speed_factor"] == pytest.approx(4.5)
        assert ws.app.state.simulation_service.stats.speed_factor == pytest.approx(4.5)
