"""WebSocket authentication helpers (issue #5913).

Called before ``websocket.accept()`` on every WS endpoint so network clients
cannot access the physics simulator or chat service without credentials.

Local-mode bypass
-----------------
When ``GOLF_SUITE_MODE=local`` or ``GOLF_AUTH_DISABLED=true`` the connection is
admitted only after an allowed loopback ``Origin`` and the launcher capability
token are verified, with a WARNING emitted so local-mode use remains visible in
production logs.

Cloud mode
----------
Requires a valid ``Authorization: Bearer <JWT>`` header.  If the header is
absent or the token is invalid the socket is closed with code 1008
(Policy Violation) before ``accept()`` is called.
"""

from __future__ import annotations

import logging
import secrets
from secrets import compare_digest
from urllib.parse import urlparse

from fastapi import WebSocket

from src.api.auth.middleware import LocalUser, is_local_mode
from src.api.auth.models import User
from src.api.auth.security import security_manager
from src.api.database import SessionLocal

logger = logging.getLogger(__name__)

_WS_CLOSE_POLICY_VIOLATION = 1008
_WS_TOKEN_QUERY_KEYS = ("launcher_token", "launcher_csrf_token")
_WS_TOKEN_PROTOCOL_PREFIX = "launcher-token."


def new_launcher_csrf_token() -> str:
    """Mint the local launcher capability token.

    Every app that serves local-mode WebSockets must publish one of these on
    ``app.state.launcher_csrf_token``; :func:`enforce_local_websocket_guard`
    compares the client's proof against it. An app that omits it rejects
    *every* local WebSocket, because the empty expected token can never match
    (issue #8075).
    """
    return secrets.token_urlsafe(32)


def install_launcher_capability_token(app: object) -> str:
    """Ensure ``app.state.launcher_csrf_token`` holds a usable token.

    Idempotent: an already-provisioned token is returned unchanged so a
    restart of one subsystem cannot invalidate proofs already handed out.

    Returns:
        The token now published on the app state.
    """
    if app is None:
        raise ValueError("app must be provided")
    state = getattr(app, "state", None)
    if state is None:
        raise ValueError("app must expose a Starlette-style .state")

    existing = getattr(state, "launcher_csrf_token", "")
    if isinstance(existing, str) and existing:
        return existing

    token = new_launcher_csrf_token()
    state.launcher_csrf_token = token
    return token


def _is_loopback_origin(value: str) -> bool:
    """Return True when an Origin value points at the local launcher UI."""
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        return False
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def _iter_ws_protocol_values(websocket: WebSocket) -> tuple[str, ...]:
    """Return requested Sec-WebSocket-Protocol values as normalized tokens."""
    header_value = websocket.headers.get("sec-websocket-protocol", "")
    if not header_value:
        return ()
    return tuple(part.strip() for part in header_value.split(",") if part.strip())


def _launcher_token_from_websocket(websocket: WebSocket, expected_token: str) -> str:
    """Extract a launcher capability token from query params or subprotocols."""
    for key in _WS_TOKEN_QUERY_KEYS:
        token = websocket.query_params.get(key)
        if token:
            return str(token)

    exact_protocol_token = ""
    for protocol in _iter_ws_protocol_values(websocket):
        if protocol.startswith(_WS_TOKEN_PROTOCOL_PREFIX):
            return protocol.removeprefix(_WS_TOKEN_PROTOCOL_PREFIX)
        if compare_digest(protocol, expected_token):
            exact_protocol_token = protocol
    return exact_protocol_token


async def enforce_local_websocket_guard(websocket: WebSocket) -> bool:
    """Require loopback Origin and launcher token before accepting local WS.

    Preconditions:
        - ``websocket`` has not been accepted yet.
        - ``websocket.app.state.launcher_csrf_token`` is the local launcher
          capability token issued via ``/api/launcher/manifest``.

    Postcondition:
        Returns ``True`` only when the Origin is loopback and the provided
        capability token matches. On failure, closes the socket with 1008.
    """
    assert websocket is not None, "websocket must be provided"

    origin = websocket.headers.get("origin", "")
    if not origin or not _is_loopback_origin(origin):
        logger.warning(
            "WebSocket local guard rejected non-loopback origin. path=%s origin=%r",
            websocket.url.path,
            origin,
        )
        await websocket.close(code=_WS_CLOSE_POLICY_VIOLATION)
        return False

    app_state = getattr(getattr(websocket, "app", None), "state", None)
    expected_token = getattr(app_state, "launcher_csrf_token", "")
    provided_token = (
        _launcher_token_from_websocket(websocket, expected_token)
        if isinstance(expected_token, str)
        else ""
    )
    if (
        not isinstance(expected_token, str)
        or not expected_token
        or not compare_digest(provided_token, expected_token)
    ):
        logger.warning(
            "WebSocket local guard rejected missing or invalid launcher proof. path=%s",
            websocket.url.path,
        )
        await websocket.close(code=_WS_CLOSE_POLICY_VIOLATION)
        return False

    return True


async def resolve_ws_user(websocket: WebSocket) -> User | LocalUser | None:
    """Authenticate a WebSocket connection before ``websocket.accept()``.

    In local mode, returns :class:`LocalUser` after logging a WARNING.
    In cloud mode, validates the ``Authorization: Bearer`` header, resolves
    and verifies the active user from the database. Returns the :class:`User` on success.

    Args:
        websocket: The incoming WebSocket connection (not yet accepted).

    Returns:
        :class:`User` or :class:`LocalUser` on success, ``None`` if authentication failed.

    Postcondition:
        If ``None`` is returned, the WebSocket has been closed (code 1008)
        and the caller must return immediately without calling ``accept()``.
    """
    assert websocket is not None, "websocket must be provided"

    if is_local_mode():
        if not await enforce_local_websocket_guard(websocket):
            return None
        logger.warning(
            "WebSocket accepted with local launcher Origin/capability guard "
            "(local/auth-disabled mode). "
            "path=%s",
            websocket.url.path,
        )
        return LocalUser()

    auth_header: str = websocket.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        await websocket.close(code=_WS_CLOSE_POLICY_VIOLATION)
        return None

    token = auth_header.split(" ", 1)[1]
    try:
        payload = security_manager.verify_token(token, "access")
        user_id = payload.get("sub")
        if user_id is None:
            logger.warning(
                "WebSocket auth rejected: sub claim missing. path=%s",
                websocket.url.path,
            )
            await websocket.close(code=_WS_CLOSE_POLICY_VIOLATION)
            return None

        # Resolve database session and retrieve real user
        db = SessionLocal()
        try:
            user: User | None = db.query(User).filter(User.id == int(user_id)).first()
            if user is None:
                logger.warning(
                    "WebSocket auth rejected: user %s not found. path=%s",
                    user_id,
                    websocket.url.path,
                )
                await websocket.close(code=_WS_CLOSE_POLICY_VIOLATION)
                return None
            if not user.is_active:
                logger.warning(
                    "WebSocket auth rejected: user %s is inactive. path=%s",
                    user_id,
                    websocket.url.path,
                )
                await websocket.close(code=_WS_CLOSE_POLICY_VIOLATION)
                return None

            # Expunge the user from session to allow closing the session
            db.expunge(user)
            return user
        finally:
            db.close()

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "WebSocket auth rejected: invalid or expired session (%s). path=%s",
            exc,
            websocket.url.path,
        )
        await websocket.close(code=_WS_CLOSE_POLICY_VIOLATION)
        return None
