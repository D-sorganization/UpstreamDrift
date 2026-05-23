"""WebSocket authentication helpers (issue #5913).

Called before ``websocket.accept()`` on every WS endpoint so network clients
cannot access the physics simulator or chat service without credentials.

Local-mode bypass
-----------------
When ``GOLF_SUITE_MODE=local`` or ``GOLF_AUTH_DISABLED=true`` the connection is
admitted, but a WARNING is emitted so the bypass is visible in production logs.

Cloud mode
----------
Requires a valid ``Authorization: Bearer <JWT>`` header.  If the header is
absent or the token is invalid the socket is closed with code 1008
(Policy Violation) before ``accept()`` is called.
"""

from __future__ import annotations

import logging

from fastapi import WebSocket

from src.api.auth.middleware import LocalUser, is_local_mode
from src.api.auth.security import security_manager

logger = logging.getLogger(__name__)

_WS_CLOSE_POLICY_VIOLATION = 1008


async def resolve_ws_user(websocket: WebSocket) -> LocalUser | None:
    """Authenticate a WebSocket connection before ``websocket.accept()``.

    In local mode, returns :class:`LocalUser` after logging a WARNING.
    In cloud mode, validates the ``Authorization: Bearer`` header.  Returns
    :class:`LocalUser` on success.

    Args:
        websocket: The incoming WebSocket connection (not yet accepted).

    Returns:
        :class:`LocalUser` on success, ``None`` if authentication failed.

    Postcondition:
        If ``None`` is returned, the WebSocket has been closed (code 1008)
        and the caller must return immediately without calling ``accept()``.
    """
    assert websocket is not None, "websocket must be provided"

    if is_local_mode():
        logger.warning(
            "WebSocket accepted without authentication (local/auth-disabled mode). "
            "Do not set GOLF_AUTH_DISABLED or GOLF_SUITE_MODE=local in production. "
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
        security_manager.verify_token(token, "access")
        return LocalUser()
    except Exception:  # noqa: BLE001
        logger.warning(
            "WebSocket auth rejected: invalid or expired token. path=%s",
            websocket.url.path,
        )
        await websocket.close(code=_WS_CLOSE_POLICY_VIOLATION)
        return None
