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

import threading
from typing import Any

__all__ = [
    "available_engines",
    "clear_registry",
    "get_provider",
    "register_provider",
]


_REGISTRY: dict[str, Any] = {}
_LOCK = threading.Lock()


def register_provider(provider: Any) -> None:
    """Register an engine-side ``fit_swing`` provider.

    Args:
        provider: An object exposing ``engine_name`` (str) and a
            ``fit_swing(target, opts) -> FitResult`` callable. The
            registry stays Protocol-agnostic so it remains compatible
            with the canonical Protocol introduced in issue #4514 once
            that lands.

    Behaviour:
        Registration is idempotent: re-registering the *same* provider
        instance under the same ``engine_name`` is a no-op. Registering
        a *different* provider under a name already in the registry
        replaces the existing entry (last-writer-wins) so test fixtures
        and reload cycles do the obvious thing.

    Raises:
        TypeError: If ``provider`` lacks an ``engine_name`` string or a
            callable ``fit_swing`` attribute.
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
    with _LOCK:
        _REGISTRY[name] = provider


def get_provider(engine_name: str) -> Any:
    """Look up a registered provider by engine name.

    Raises:
        KeyError: If no provider is registered under ``engine_name``.
    """
    with _LOCK:
        try:
            return _REGISTRY[engine_name]
        except KeyError as exc:
            available = sorted(_REGISTRY)
            raise KeyError(
                f"no fit_swing provider registered for engine '{engine_name}'; "
                f"available={available}"
            ) from exc


def available_engines() -> list[str]:
    """Return the sorted list of registered engine names."""
    with _LOCK:
        return sorted(_REGISTRY)


def clear_registry() -> None:
    """Remove every entry from the registry. Test helper only."""
    with _LOCK:
        _REGISTRY.clear()
