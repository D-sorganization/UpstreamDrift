"""Host-header validation driven by the shared ``ALLOWED_HOSTS`` setting."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from src.api.config import get_allowed_hosts


def configure_trusted_hosts(app: FastAPI) -> None:
    """Serve only requests addressed to an allowed host.

    The allowed list comes from the ``ALLOWED_HOSTS`` setting, the same policy
    ``src/api/server.py`` applies, so the two apps cannot drift. Call this
    after the other middleware is added so the check is outermost and runs
    first.

    Raises:
        ValueError: If ``app`` is not provided.
    """
    if app is None:
        raise ValueError("app must be provided")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=get_allowed_hosts())
