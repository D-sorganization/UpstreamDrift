"""Optional-dependency capability checks for simulation backends.

The suite must import and run (ODE + MuJoCo-CPU backends) on a machine with no
NVIDIA GPU and no Warp stack installed. These helpers perform *guarded* imports
so that capability can be queried without raising, and provide ``require_*``
gates that fail loudly with an actionable message when a GPU backend is asked
for without its extra installed.

Results are cached: import availability does not change within a process.
"""

from __future__ import annotations

from functools import lru_cache

from src.shared.python.logging_pkg.logging_config import get_logger

from .exceptions import BackendNotAvailableError

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def has_mujoco() -> bool:
    """Return ``True`` if the CPU MuJoCo bindings import successfully."""
    return _can_import("mujoco")


@lru_cache(maxsize=1)
def has_warp() -> bool:
    """Return ``True`` if the full MuJoCo Warp GPU stack is importable.

    Requires both ``warp`` (NVIDIA Warp) and ``mujoco_warp`` (MJWarp). Either
    one missing means the GPU backend cannot run.
    """
    return _can_import("warp") and _can_import("mujoco_warp")


@lru_cache(maxsize=1)
def has_mjx() -> bool:
    """Return ``True`` if the ``mujoco-mjx`` + JAX stack imports successfully."""
    return _can_import("jax") and _can_import("jaxlib") and _can_import("mujoco.mjx")


@lru_cache(maxsize=1)
def warp_device_available() -> bool:
    """Return ``True`` if Warp reports at least one usable CUDA device.

    Importing the stack is necessary but not sufficient — a machine may have the
    wheels installed yet no CUDA device visible. Never raises.
    """
    if not has_warp():
        return False
    try:
        import warp as wp

        wp.init()
        return bool(wp.get_cuda_device_count() > 0)
    except (ImportError, RuntimeError, AttributeError) as exc:
        logger.debug("Warp CUDA probe failed: %s", exc)
        return False


def require_mujoco() -> None:
    """Raise :class:`BackendNotAvailableError` if CPU MuJoCo is unavailable."""
    if not has_mujoco():
        raise BackendNotAvailableError(
            "The MuJoCo CPU backend requires the 'mujoco' package. "
            "Install it with: pip install 'mujoco>=3.6,<4'"
        )


def require_warp() -> None:
    """Raise :class:`BackendNotAvailableError` if the Warp GPU stack is missing."""
    if not has_warp():
        raise BackendNotAvailableError(
            "The MuJoCo Warp backend requires the optional GPU stack "
            "(CUDA + NVIDIA GPU). Install it with: pip install 'upstream-drift[warp]' "
            "(provides 'mujoco-warp' and 'warp-lang'). "
            "The suite runs fully on CPU via the 'ode' and 'mujoco' backends "
            "without this extra."
        )


def require_mjx() -> None:
    """Raise :class:`BackendNotAvailableError` if MJX/JAX is unavailable."""
    if not has_mjx():
        raise BackendNotAvailableError(
            "The MJX backend requires MuJoCo's JAX stack. Install it with: "
            "pip install 'upstream-drift[mjx]' (adds mujoco-mjx, JAX, and "
            "JAXLIB). The suite runs fully on CPU via the 'ode' and 'mujoco' "
            "backends without this extra."
        )


def _can_import(module_name: str) -> bool:
    """Return whether ``module_name`` imports without error.

    Uses :func:`importlib.util.find_spec` first (cheap, no side effects) then a
    real import to catch packages that are discoverable but broken (e.g. a Warp
    wheel present without a CUDA toolkit).
    """
    import importlib
    import importlib.util

    try:
        if importlib.util.find_spec(module_name) is None:
            return False
        importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        logger.debug("Optional module %r unavailable: %s", module_name, exc)
        return False
    return True
