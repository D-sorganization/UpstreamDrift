"""Regression: local-mode WebSockets must be *possible* to authenticate.

Issue #8075 -- the Sidekick chat panel sat on
``Disconnected - retrying in 3s...`` forever. ``enforce_local_websocket_guard``
compares the client's proof against ``app.state.launcher_csrf_token``, but only
``src/api/local_server.py`` ever published one. The app the desktop launcher
actually runs (``python -m src.api.server``) left it unset, and an unset
expected token can never match -- so *every* local WebSocket was refused with a
1008 close, surfacing to QWebSocket as ``403 Forbidden``.

The existing WS tests all set the token by hand on a throwaway app, which is
why production's omission went unnoticed. These tests assert the property on
the real app objects.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.api.auth.ws_auth import (
    install_launcher_capability_token,
    new_launcher_csrf_token,
)

pytestmark = pytest.mark.unit


def test_new_token_is_unguessable_and_unique() -> None:
    first = new_launcher_csrf_token()
    second = new_launcher_csrf_token()

    assert first != second
    assert len(first) >= 32


def test_install_is_idempotent() -> None:
    """Re-provisioning must not invalidate proofs already handed out."""
    app = SimpleNamespace(state=SimpleNamespace())

    first = install_launcher_capability_token(app)
    second = install_launcher_capability_token(app)

    assert first == second
    assert app.state.launcher_csrf_token == first


def test_install_replaces_an_empty_placeholder() -> None:
    app = SimpleNamespace(state=SimpleNamespace(launcher_csrf_token=""))

    token = install_launcher_capability_token(app)

    assert token
    assert app.state.launcher_csrf_token == token


def test_install_rejects_missing_app() -> None:
    with pytest.raises(ValueError, match="app must be provided"):
        install_launcher_capability_token(None)


def test_install_rejects_app_without_state() -> None:
    with pytest.raises(ValueError, match=r"Starlette-style \.state"):
        install_launcher_capability_token(object())


@pytest.mark.slow
def test_api_server_app_publishes_a_launcher_capability_token() -> None:
    """The launcher-started app must satisfy the local WebSocket guard.

    Without this, the Sidekick chat dock (and every other local WebSocket)
    can never connect no matter what the client sends.
    """
    with patch.dict(os.environ, {"GOLF_SUITE_MODE": "local"}):
        from src.api.server import app

    token = getattr(app.state, "launcher_csrf_token", "")

    assert isinstance(token, str)
    assert token, (
        "src.api.server.app must publish app.state.launcher_csrf_token; "
        "enforce_local_websocket_guard rejects every WebSocket without it"
    )
