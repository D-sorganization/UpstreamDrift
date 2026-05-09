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


_REGISTRY: dict[str, Any] = {}
_LOCK = threading.Lock()
_logger = logging.getLogger(__name__)


def _provider_qualname(provider: object) -> str:
    """Return the fully-qualified ``module.qualname`` for ``provider``'s class.

    Used to detect re-registrations that originate from the same logical
    class even after :func:`importlib.reload` has rebuilt it (and thus
    broken ``type(a) is type(b)`` identity).
    """
    cls = type(provider)
    module = getattr(cls, "__module__", "") or ""
    qualname = getattr(cls, "__qualname__", cls.__name__)
    return f"{module}.{qualname}" if module else qualname


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
        instance, or any instance of the *same* provider class (matched
        by fully-qualified ``module.qualname`` so :func:`importlib.reload`
        shadows count), is a no-op and emits a DEBUG log. Registering a
        *different* provider class for an already-occupied ``engine_name``
        raises :class:`ValueError` naming both classes.

    Raises:
        TypeError: If ``provider`` lacks an ``engine_name`` string or a
            callable ``fit_swing`` attribute.
        ValueError: If a *different* provider class tries to register
            under an ``engine_name`` already taken.
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
        existing = _REGISTRY.get(name)
        if existing is provider:
            _logger.debug(
                "register_provider: %r already registered (same instance); no-op",
                name,
            )
            return
        if existing is not None and (
            type(existing) is type(provider)
            or _provider_qualname(existing) == _provider_qualname(provider)
        ):
            _logger.debug(
                "register_provider: %r already registered to %s; no-op",
                name,
                _provider_qualname(existing),
            )
            return
        if existing is not None:
            raise ValueError(
                f"engine_name {name!r} is already registered to "
                f"{_provider_qualname(existing)}; got "
                f"{_provider_qualname(provider)}"
            )
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
