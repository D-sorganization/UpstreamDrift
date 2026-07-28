"""Environment gate for runtime diagnostic and debug endpoints."""

from __future__ import annotations

import os

from src.api.environment import is_production_environment

_TRUTHY = {"1", "true", "yes", "on"}


def debug_endpoints_enabled() -> bool:
    """Return whether runtime debug endpoints should be registered.

    Postcondition: production disables debug endpoints unless
    ``UPSTREAM_DRIFT_DEBUG_ENDPOINTS`` is explicitly truthy. "Production" is
    resolved from ``ENVIRONMENT`` or the legacy ``UPSTREAM_DRIFT_ENV``
    (issue #7994).
    """
    explicit_flag = os.environ.get("UPSTREAM_DRIFT_DEBUG_ENDPOINTS", "")
    debug_enabled = explicit_flag.casefold() in _TRUTHY
    return not is_production_environment() or debug_enabled
