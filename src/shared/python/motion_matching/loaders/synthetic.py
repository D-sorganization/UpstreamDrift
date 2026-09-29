"""Synthetic-target dispatcher.

Engine-specific implementations register themselves by calling
:func:`register_backend`. The cross-engine entry point
:func:`synthesize_target_from_coefficients` looks up the backend by name
and forwards the call.

The dispatcher itself does NOT import any engine package — that would
defeat the lazy-import policy that keeps the GUI from pulling MuJoCo /
Drake / Pinocchio transitively. Callers must import the desired backend
module first (e.g. ``from src.engines.physics_engines.mujoco.python.motion_matching
import synthesize``) which will register itself on import. After that,
calling ``synthesize_target_from_coefficients(theta, opts, engine="mujoco")``
dispatches to the registered implementation.

Issue history: stub originally raised ``NotImplementedError`` (#014/#018);
MuJoCo backend lands in #4122; Simscape and Drake backends to follow.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..club_target import AlignOptions, ClubTarget

logger = logging.getLogger(__name__)

__all__ = [
    "SyntheticBackend",
    "available_backends",
    "register_backend",
    "synthesize_target_from_coefficients",
]


class SyntheticBackend(Protocol):
    """Engine-specific implementation contract.

    Every backend takes the same ``(theta, opts)`` pair the cross-engine
    entry point accepts and returns a fully-validated :class:`ClubTarget`.
    """

    def __call__(
        self, theta: NDArray[np.float64], opts: AlignOptions
    ) -> ClubTarget: ...


# Module-level registry. Keys are engine names (``"mujoco"``, ``"simscape"``,
# ``"drake"``, ...). Values are backend callables.
_BACKENDS: dict[str, Callable[[NDArray[np.float64], AlignOptions], ClubTarget]] = {}


def register_backend(
    engine: str,
    backend: Callable[[NDArray[np.float64], AlignOptions], ClubTarget],
) -> None:
    """Register ``backend`` as the implementation for ``engine``.

    Idempotent: re-registering the same callable for the same engine is a
    no-op; replacing with a different callable logs a warning so accidental
    overrides are visible.

    Args:
        engine:  Engine identifier, e.g. ``"mujoco"``. Case-insensitive.
        backend: Callable matching :class:`SyntheticBackend`.

    Raises:
        TypeError: if ``backend`` is not callable.
        ValueError: if ``engine`` is not a non-empty string.
    """
    if not isinstance(engine, str) or not engine:
        raise ValueError("engine must be a non-empty string")
    if not callable(backend):
        raise TypeError(f"backend must be callable; got {type(backend).__name__}")
    key = engine.lower()
    existing = _BACKENDS.get(key)
    if existing is not None and existing is not backend:
        logger.warning("replacing existing %r backend with %r", engine, backend)
    _BACKENDS[key] = backend


def available_backends() -> tuple[str, ...]:
    """Return a sorted tuple of registered engine names."""
    return tuple(sorted(_BACKENDS))


def synthesize_target_from_coefficients(
    theta: NDArray[np.float64],
    opts: AlignOptions | None = None,
    *,
    engine: str | None = None,
) -> ClubTarget:
    """Dispatch to the registered backend for ``engine``.

    Args:
        theta: ``(n,)`` flat coefficient vector. Each backend imposes its
            own layout / bounds; this dispatcher does not validate.
        opts:  :class:`AlignOptions`; defaults to ``AlignOptions()``.
        engine: Engine name. ``None`` defers to ``opts.engine`` if that
            attribute exists, otherwise raises ``ValueError``.

    Returns:
        Validated :class:`ClubTarget` from the backend.

    Raises:
        ValueError: if no engine is specified.
        LookupError: if no backend is registered for the requested engine.
            The error message lists the registered backends so the caller
            can tell whether they need to import the engine module first.
    """
    if opts is None:
        opts = AlignOptions()

    # Allow callers to pass the engine via opts (forward compatibility for
    # if/when AlignOptions grows an explicit ``engine`` field).
    engine_name = engine or getattr(opts, "engine", None)
    if not engine_name:
        raise ValueError(
            "no engine specified. Pass engine= explicitly or set opts.engine. "
            f"Available backends: {available_backends()}"
        )
    key = str(engine_name).lower()
    backend = _BACKENDS.get(key)
    if backend is None:
        raise LookupError(
            f"no synthetic backend registered for engine {engine_name!r}. "
            f"Available: {available_backends()}. "
            "Make sure you've imported the engine's synthesize module "
            "(e.g. ``from src.engines.physics_engines.mujoco.python."
            "motion_matching import synthesize``)."
        )

    theta_arr = np.ascontiguousarray(np.asarray(theta, dtype=np.float64).reshape(-1))
    logger.debug(
        "dispatching synthesize_target_from_coefficients to %r backend "
        "(theta shape %s)",
        key,
        theta_arr.shape,
    )
    return backend(theta_arr, opts)
