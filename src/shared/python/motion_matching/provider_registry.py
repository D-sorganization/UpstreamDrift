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
    register_provider as _register_canonical,
)

def register_provider(provider: Any) -> None:
    """Register an engine-side ``fit_swing`` provider.

    Maintains legacy validation (raises TypeError for missing/invalid
    engine_name or missing fit_swing) before delegating to the
    canonical registry.
    """
    name = getattr(provider, "engine_name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(
            "provider must expose a non-empty 'engine_name' string attribute"
        )
    if not callable(getattr(provider, "fit_swing", None)):
        raise TypeError(
            f"provider for engine '{name}' must expose a callable 'fit_swing'"
        )
    
    # Validation passed, delegate to the canonical registry
    _register_canonical(provider)

# clear_registry is not in provider.py, but it's used in tests. We can implement it here by reaching into provider.py's internals, or we can just import the internal ones.
from .provider import _REGISTRY, _REGISTRY_LOCK

def clear_registry() -> None:
    """Remove every entry from the registry. Test helper only."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()

