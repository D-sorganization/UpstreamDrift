"""Engine-agnostic provider registry for motion-matching.

Issue #4516 ships the per-engine wrapper for Drake; the registry it
plugs into is the foundation defined in #4514. This module provides a
minimal, forward-compatible registry surface that the Drake provider
(and subsequent per-engine providers) can register against today,
without blocking on the full canonical schema landing on ``main``.

Public API:
    register_provider(provider)  -- idempotent registration.
    get_provider(engine_name)    -- typed lookup.
    available_engines()          -- sorted list of registered engines.
    clear_registry()             -- test-helper to wipe all entries.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

__all__ = [
    "available_engines",
    "clear_registry",
    "get_provider",
    "register_provider",
]


from .provider import (
    available_engines,
    get_provider,
    register_provider,
)

# clear_registry is not in provider.py, but it's used in tests. We can implement it here by reaching into provider.py's internals, or we can just import the internal ones.
from .provider import _REGISTRY, _REGISTRY_LOCK

def clear_registry() -> None:
    """Remove every entry from the registry. Test helper only."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()

