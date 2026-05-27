"""Canonical ``FitSwingProvider`` Protocol and registry.

Per cross-engine parity (#4513) and the canonical fit_swing API (#4514),
every physics engine's motion-matching driver MUST implement the
:class:`FitSwingProvider` Protocol so the upstream matcher can dispatch
through a single registry instead of a switch on engine name.

This module is intentionally minimal: it defines the Protocol, a plain
:class:`FitOptions` carrier, a :class:`MultiSourceTarget` adapter
(``target.club`` / ``target.body`` slots), and a thread-safe registry.
Engine-specific options (cost weights, integrator settings, minimizer
flags) live alongside each engine's ``fit_swing.py`` and are passed
through ``FitOptions.engine_options``.

Public API:
    FitSwingProvider  -- Protocol every engine implements.
    FitOptions        -- canonical carrier for fit knobs + engine extras.
    MultiSourceTarget -- bundle of (optional) club + body targets.
    register_provider -- attach a provider instance to the registry.
    get_provider      -- look up a provider by ``engine_name``.
    available_engines -- list registered engine names.
"""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .club_target import ClubTarget
from .fit_result import CanonicalFitResult

__all__ = [
    "FitOptions",
    "FitSwingProvider",
    "MultiSourceTarget",
    "available_engines",
    "get_provider",
    "register_provider",
]


@dataclass(frozen=True)
class FitOptions:
    """Canonical, engine-agnostic carrier for ``fit_swing`` options.

    The canonical knobs (max iterations, RNG seed) live as direct fields;
    everything else is funnelled through ``engine_options`` so engine
    adapters can pass their native options dataclass without losing
    type information.

    Attributes:
        maxiter:        upper bound on solver iterations (engines free to
                        clamp to their own ceiling).
        rng_seed:       seed for any stochastic warm-start draws.
        engine_options: opaque per-engine options object (e.g. the engine's
                        own ``FitOptions`` dataclass). The provider adapter
                        is responsible for reading the right type.
    """

    maxiter: int = 200
    rng_seed: int = 0
    engine_options: Any = None


@dataclass(frozen=True)
class MultiSourceTarget:
    """Bundle of (optional) club and body targets.

    Per #4519 each provider declares whether it consumes ``.club`` and / or
    ``.body``; MuJoCo currently consumes only ``.club``.

    At least one of ``club`` or ``body`` MUST be set; constructing the
    bundle with neither raises :class:`ValueError`.
    """

    club: ClubTarget | None = None
    body: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.club is None and self.body is None:
            raise ValueError(
                "MultiSourceTarget must have at least one of "
                "(club, body) set; both are None"
            )


@runtime_checkable
class FitSwingProvider(Protocol):
    """Engine-side adapter from canonical motion-matching API to a fitter.

    Each physics engine ships exactly one provider instance and registers
    it via :func:`register_provider` at import time. The matcher then
    drives every engine through this Protocol.

    Required attributes:
        engine_name: lowercase engine identifier (``"mujoco"``, etc.).

    Required methods:
        fit_swing(target, opts) -> CanonicalFitResult
        supports_body_target() -> bool
        supports_ball_target() -> bool

    Optional methods:
        engine_version() -> str
            Version string of the underlying physics engine wheel
            (e.g. ``pydrake.__version__``). Used to stamp leaderboard rows
            so two runs against different wheels are distinguishable.
            Defaults to ``"unknown"`` for back-compat with providers that
            predate this hook (issue #4705).
    """

    engine_name: str

    def fit_swing(
        self,
        target: MultiSourceTarget | ClubTarget,
        opts: FitOptions,
    ) -> CanonicalFitResult: ...

    def supports_body_target(self) -> bool: ...

    def supports_ball_target(self) -> bool: ...

    def engine_version(self) -> str:
        """Return the underlying engine's version string.

        Default implementation returns ``"unknown"`` so providers
        predating issue #4705 stay Protocol-compliant. Real providers
        should override to query their engine's ``__version__``
        attribute (with a ``try/except ImportError`` fallback so the
        provider stays constructible without the engine wheel).
        """
        return "unknown"


# --- Registry ---------------------------------------------------------------

_REGISTRY: dict[str, FitSwingProvider] = {}
_REGISTRY_LOCK = threading.Lock()
_logger = logging.getLogger(__name__)


def _provider_qualname(provider: object) -> str:
    """Return the fully-qualified ``module.qualname`` for a provider class.

    Used to detect re-registrations that originate from the *same* logical
    provider class even after :func:`importlib.reload` has rebuilt the
    class object (and thus broken ``type(a) is type(b)`` identity).
    """
    cls = type(provider)
    module = getattr(cls, "__module__", "") or ""
    qualname = getattr(cls, "__qualname__", cls.__name__)
    return f"{module}.{qualname}" if module else qualname


def _same_provider_class(a: object, b: object) -> bool:
    """Return ``True`` if *a* and *b* are instances of the same logical class.

    Beyond the obvious ``type(a) is type(b)`` and matching
    ``module.qualname``, this also catches the case where the same
    ``.py`` source file is importable under two different module paths
    (e.g. because a test adds an engine sub-tree to ``sys.path``).  In
    that situation the ``__qualname__`` portions match and both classes
    resolve to the same ``__file__``.
    """
    if type(a) is type(b):
        return True
    if _provider_qualname(a) == _provider_qualname(b):
        return True
    # Same bare class name + same source file → same logical class despite
    # different module paths (dual-import via sys.path manipulation).
    cls_a, cls_b = type(a), type(b)
    if getattr(cls_a, "__qualname__", None) == getattr(cls_b, "__qualname__", None):
        try:
            file_a = inspect.getfile(cls_a)
            file_b = inspect.getfile(cls_b)
            if file_a == file_b:
                return True
        except (TypeError, OSError):
            pass
    return False


def register_provider(provider: FitSwingProvider) -> None:
    """Register ``provider`` under its ``engine_name``.

    Registration is idempotent: re-registering the same provider instance,
    or any instance of the same provider class (matched by fully-qualified
    ``module.qualname`` so :func:`importlib.reload` shadows still count,
    or by matching ``__qualname__`` + source file so dual-import via
    ``sys.path`` manipulation is also detected), is a no-op and emits a
    DEBUG log. Registering a *different* provider class for an
    already-occupied ``engine_name`` raises :class:`ValueError` naming both
    the existing and the incoming class.
    """
    name = getattr(provider, "engine_name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"provider must expose a non-empty engine_name str, got {name!r}"
        )
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(name)
        if existing is provider:
            _logger.debug(
                "register_provider: %r already registered (same instance); no-op",
                name,
            )
            return
        if existing is not None and _same_provider_class(existing, provider):
            # Same logical class — covers ordinary re-imports,
            # ``importlib.reload`` shadows, and dual sys.path imports.
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


def get_provider(engine_name: str) -> FitSwingProvider:
    """Return the provider registered under ``engine_name``.

    Raises :class:`KeyError` if no provider has registered yet (callers
    typically need to import the engine's motion-matching package first).
    """
    with _REGISTRY_LOCK:
        if engine_name not in _REGISTRY:
            raise KeyError(
                f"no FitSwingProvider registered for {engine_name!r}; "
                f"registered: {sorted(_REGISTRY)}"
            )
        return _REGISTRY[engine_name]


def available_engines() -> list[str]:
    """Return the sorted list of currently-registered engine names."""
    with _REGISTRY_LOCK:
        return sorted(_REGISTRY)
