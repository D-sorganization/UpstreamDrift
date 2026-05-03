"""Environment gate for runtime diagnostic and debug endpoints."""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}


def debug_endpoints_enabled() -> bool:
    """Return whether runtime debug endpoints should be registered.

    Postcondition: production disables debug endpoints unless
    ``UPSTREAM_DRIFT_DEBUG_ENDPOINTS`` is explicitly truthy.
    """
    environment = os.environ.get("UPSTREAM_DRIFT_ENV", "development").casefold()
    explicit_flag = os.environ.get("UPSTREAM_DRIFT_DEBUG_ENDPOINTS", "")
    debug_enabled = explicit_flag.casefold() in _TRUTHY
    return environment != "production" or debug_enabled
