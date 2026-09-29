"""Shared rate limiting primitives for the API.

Centralises the slowapi ``Limiter`` instance and a small env-driven helper so
route modules can apply ``@limiter.limit(...)`` decorators without importing
``server.py`` (which would create a circular import).

Limits are sourced from environment variables with sensible defaults so
operators can tune behaviour without code changes. The default policy targets
expensive endpoints (simulation, optimisation, training, inverse dynamics)
that should not be hit at high frequency by a single client.
"""

from __future__ import annotations

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

# Single shared limiter. ``key_func`` falls back to the remote address which is
# correct for the lightweight protections we want here. Operators running the
# API behind a proxy should configure ``X-Forwarded-For`` handling at the
# proxy/middleware layer rather than overriding the key function.
limiter = Limiter(key_func=get_remote_address)


def get_limit(env_var: str, default: str) -> str:
    """Return the rate limit string for ``env_var`` falling back to ``default``.

    The value is read at import time when used as a decorator argument so
    operators must restart the process to pick up changes; tests can call this
    helper directly when they need to read the live value.

    Args:
        env_var: Name of the environment variable holding a slowapi-compatible
            limit string (e.g. ``"5/minute"``).
        default: Fallback limit string used when the variable is unset or
            blank.

    Returns:
        The configured limit string.
    """
    value = os.environ.get(env_var, "").strip()
    return value or default
