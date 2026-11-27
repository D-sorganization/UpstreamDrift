"""Launcher capability check for the local desktop API app.

The launcher UI reads the capability from ``GET /api/launcher/manifest`` and
returns it in the ``X-Launcher-CSRF-Token`` header on every API request. This
middleware refuses ``/api/`` requests that do not carry it.

Routes a browser requests without custom headers stay open: health probes, the
manifest that hands out the capability, tile logos (``<img>``) and the API docs
pages. WebSocket handshakes are checked separately in ``src.api.auth.ws_auth``.

Image and download routes are guarded too. The UI fetches them with the header
and shows them from object URLs (``ui/src/api/launcherMedia.ts``), because a
plain ``<img src>`` or ``<a href>`` request cannot carry it.

Opt-out: ``UPSTREAMDRIFT_LOCAL_API_ALLOW_UNAUTHENTICATED=1`` (also ``true`` or
``yes``) turns the check off for loopback callers only and logs a warning at
startup. It is meant for local debugging; leave it unset for normal use.
"""

from __future__ import annotations

import logging
import os
from secrets import compare_digest
from urllib.parse import urlparse

from fastapi import FastAPI
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

LAUNCHER_CAPABILITY_HEADER = "X-Launcher-CSRF-Token"
ALLOW_UNAUTHENTICATED_ENV = "UPSTREAMDRIFT_LOCAL_API_ALLOW_UNAUTHENTICATED"

_GUARDED_PREFIX = "/api/"
_OPEN_PATHS = frozenset(
    {
        "/api/health",
        "/api/v1/health",
        "/api/launcher/manifest",
        "/api/docs",
        "/api/redoc",
    }
)
_OPEN_PREFIXES = ("/api/launcher/logos/", "/api/docs/")
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
# "testclient" is the in-process Starlette TestClient peer.
_LOOPBACK_PEERS = _LOOPBACK_HOSTS | {"testclient"}
_TRUTHY = frozenset({"1", "true", "yes"})


def allow_unauthenticated_requested() -> bool:
    """Return True when the documented opt-out variable is set to a truthy value."""
    return os.environ.get(ALLOW_UNAUTHENTICATED_ENV, "").strip().lower() in _TRUTHY


def requires_capability(path: str) -> bool:
    """Return True when ``path`` must carry the launcher capability."""
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not path.startswith(_GUARDED_PREFIX):
        return False
    return path not in _OPEN_PATHS and not path.startswith(_OPEN_PREFIXES)


def _is_loopback_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.hostname in _LOOPBACK_HOSTS


def _is_loopback_peer(scope: Scope) -> bool:
    client = scope.get("client")
    # No network peer means an in-process ASGI call, local by definition.
    return client is None or client[0] in _LOOPBACK_PEERS


def _refusal(detail: str) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": detail})


class LocalCapabilityMiddleware:
    """Refuse local API requests that lack the launcher capability.

    Preconditions:
        The wrapped application sets ``scope["app"]`` and stores the expected
        capability on ``app.state.launcher_csrf_token``.

    Postcondition:
        A guarded request reaches the application only when it carries the
        expected capability and no non-loopback ``Origin``/``Referer``, or when
        the opt-out is enabled and the caller is on the loopback interface. A
        missing or blank expected capability refuses every guarded request.
    """

    def __init__(self, app: ASGIApp, *, allow_unauthenticated: bool = False) -> None:
        if app is None:
            raise ValueError("app must be provided")
        self.app = app
        self.allow_unauthenticated = allow_unauthenticated

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not requires_capability(scope["path"]):
            await self.app(scope, receive, send)
            return
        if self.allow_unauthenticated and _is_loopback_peer(scope):
            await self.app(scope, receive, send)
            return

        refusal = self._check(scope)
        if refusal is not None:
            logger.warning("Local API request refused: path=%s", scope["path"])
            await refusal(scope, receive, send)
            return
        await self.app(scope, receive, send)

    @staticmethod
    def _check(scope: Scope) -> JSONResponse | None:
        headers = Headers(scope=scope)
        state = getattr(scope.get("app"), "state", None)
        expected = getattr(state, "launcher_csrf_token", "")
        provided = headers.get(LAUNCHER_CAPABILITY_HEADER, "")
        if (
            not isinstance(expected, str)
            or not expected
            or not compare_digest(provided.encode(), expected.encode())
        ):
            return _refusal("Launcher capability is required")
        for name in ("origin", "referer"):
            value = headers.get(name)
            if value and not _is_loopback_url(value):
                return _refusal("Launcher capability origin is not allowed")
        return None


def configure_local_capability_guard(app: FastAPI) -> None:
    """Install the capability check on the local app.

    Call this before the CORS middleware is added so the check sits inside it:
    preflight requests are answered by CORS and refusals keep their CORS headers.
    """
    if app is None:
        raise ValueError("app must be provided")
    allow_unauthenticated = allow_unauthenticated_requested()
    if allow_unauthenticated:
        logger.warning(
            "%s is set: the local API serves loopback callers WITHOUT the "
            "launcher capability. Unset it for normal use.",
            ALLOW_UNAUTHENTICATED_ENV,
        )
    app.add_middleware(
        LocalCapabilityMiddleware, allow_unauthenticated=allow_unauthenticated
    )
