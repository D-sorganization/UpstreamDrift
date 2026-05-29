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
from src.api.auth.models import User
from src.api.auth.security import security_manager
from src.api.database import SessionLocal

logger = logging.getLogger(__name__)

_WS_CLOSE_POLICY_VIOLATION = 1008


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
        payload = security_manager.verify_token(token, "access")
        user_id = payload.get("sub")
        if user_id is None:
            logger.warning(
                "WebSocket auth rejected: token sub claim missing. path=%s",
                websocket.url.path,
            )
            await websocket.close(code=_WS_CLOSE_POLICY_VIOLATION)
            return None

        # Resolve database session and retrieve real user
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == int(user_id)).first()
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
            "WebSocket auth rejected: invalid or expired token (%s). path=%s",
            exc,
            websocket.url.path,
        )
        await websocket.close(code=_WS_CLOSE_POLICY_VIOLATION)
        return None
