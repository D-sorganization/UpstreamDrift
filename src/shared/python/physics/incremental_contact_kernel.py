"""One impulse-integration algorithm with optional no-fastmath Numba compilation."""

from __future__ import annotations

import math
import importlib
import importlib.util
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

numba: Any = (
    importlib.import_module("numba")
    if importlib.util.find_spec("numba") is not None
    else None
)


class ContactKernelInput(NamedTuple):
    """Explicit physical and integration inputs for the contact kernel."""

    relative_initial: np.ndarray
    normal: np.ndarray
    inverse_mass: np.ndarray
    tangent: np.ndarray
    tangent_mass: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    friction: float
    energetic_restitution: float
    steps_per_compression: int


def _disk_source(
    target: np.ndarray,
    tangent_mass: np.ndarray,
    radius: float,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, bool]:
    if radius <= 0:
        return np.zeros(2), bool(np.linalg.norm(target) > 0)
    free = np.linalg.solve(tangent_mass, target)
    if np.linalg.norm(free) <= radius:
        return free, False
    coefficients = eigenvectors.T @ target
    squared_radius = radius * radius
    low, high = 0.0, float(max(1.0, np.linalg.norm(target) / radius))
    while (coefficients[0] / (eigenvalues[0] + high)) ** 2 + (
        coefficients[1] / (eigenvalues[1] + high)
    ) ** 2 > squared_radius:
        high *= 2
    for _ in range(48):
        middle = (low + high) * 0.5
        excess = (
            (coefficients[0] / (eigenvalues[0] + middle)) ** 2
            + (coefficients[1] / (eigenvalues[1] + middle)) ** 2
            - squared_radius
        )
        if excess > 0:
            low = middle
        else:
            high = middle
    clipped = eigenvectors @ (coefficients / (eigenvalues + high))
    if np.linalg.norm(clipped) > radius * (1 + 1e-10):
        raise RuntimeError("tangential Coulomb disk postcondition failed")
    return clipped, True


_disk_kernel = (
    numba.njit(cache=True, fastmath=False)(_disk_source)
    if numba is not None
    else _disk_source
)


def _trial_source(
    current: np.ndarray,
    step: float,
    physical: tuple,
    disk_solver: Callable[..., tuple[np.ndarray, bool]],
) -> tuple[np.ndarray, float, float, bool]:
    (
        relative_initial,
        normal,
        inverse_mass,
        tangent,
        tangent_mass,
        eigenvalues,
        eigenvectors,
        friction,
    ) = physical
    relative = relative_initial - inverse_mass @ current
    normal_before = float(relative @ normal)
    tangent_target = tangent.T @ (relative - step * inverse_mass @ normal)
    tangential, sliding = disk_solver(
        tangent_target, tangent_mass, friction * step, eigenvalues, eigenvectors
    )
    delta = step * normal + tangent @ tangential
    normal_after = float((relative - inverse_mass @ delta) @ normal)
    return delta, normal_before, normal_after, sliding


_trial_kernel = (
    numba.njit(cache=True, fastmath=False)(_trial_source)
    if numba is not None
    else _trial_source
)


def _trial_at_step_source(
    impulse: np.ndarray,
    step: float,
    physical: tuple,
    trial_solver: Callable[..., tuple[np.ndarray, float, float, bool]],
    disk_solver: Callable[..., tuple[np.ndarray, bool]],
) -> tuple[np.ndarray, float, float, bool]:
    """Evaluate one unchanged contact trial with its shared physical inputs."""
    return trial_solver(impulse, step, physical, disk_solver)


_trial_at_step_kernel = (
    numba.njit(cache=True, fastmath=False)(_trial_at_step_source)
    if numba is not None
    else _trial_at_step_source
)


def _compression_source(
    impulse: np.ndarray,
    base_step: float,
    steps_per_compression: int,
    physical: tuple,
    trial_solver: Callable[..., tuple[np.ndarray, float, float, bool]],
    disk_solver: Callable[..., tuple[np.ndarray, bool]],
    trial_at_step: Callable[..., tuple[np.ndarray, float, float, bool]],
) -> tuple[np.ndarray, float, bool]:
    compression_work = 0.0
    slipped = False
    compression_done = False
    for _ in range(10 * steps_per_compression):
        step = base_step
        delta, before, after, is_sliding = trial_at_step(
            impulse, step, physical, trial_solver, disk_solver
        )
        if after <= 0:
            lo, hi = 0.0, step
            for _ in range(48):
                mid = (lo + hi) * 0.5
                if (
                    trial_at_step(impulse, mid, physical, trial_solver, disk_solver)[2]
                    > 0
                ):
                    lo = mid
                else:
                    hi = mid
            step = hi
            delta, before, after, is_sliding = trial_at_step(
                impulse, step, physical, trial_solver, disk_solver
            )
            compression_done = True
        impulse += delta
        compression_work += 0.5 * (before + after) * step
        slipped |= is_sliding
        if compression_done:
            break
    if not compression_done:
        raise RuntimeError("compression event not reached within impulse horizon")
    if compression_work <= 0:
        raise RuntimeError("normal compression work must be positive")
    return impulse, compression_work, slipped


_compression_kernel = (
    numba.njit(cache=True, fastmath=False)(_compression_source)
    if numba is not None
    else _compression_source
)


def _rebound_source(
    impulse: np.ndarray,
    base_step: float,
    steps_per_compression: int,
    physical: tuple,
    target_work: float,
    trial_solver: Callable[..., tuple[np.ndarray, float, float, bool]],
    disk_solver: Callable[..., tuple[np.ndarray, bool]],
    trial_at_step: Callable[..., tuple[np.ndarray, float, float, bool]],
) -> tuple[np.ndarray, float, bool]:
    rebound_work = 0.0
    slipped = False
    rebound_done = False
    for _ in range(10 * steps_per_compression):
        if target_work - rebound_work <= 1e-12:
            rebound_done = True
            break
        step = base_step
        delta, before, after, is_sliding = trial_at_step(
            impulse, step, physical, trial_solver, disk_solver
        )
        gained = -0.5 * (before + after) * step
        if gained >= target_work - rebound_work:
            lo, hi = 0.0, step
            for _ in range(48):
                mid = (lo + hi) * 0.5
                _, start_mid, end_mid, _ = trial_at_step(
                    impulse, mid, physical, trial_solver, disk_solver
                )
                work_mid = -0.5 * (start_mid + end_mid) * mid
                if work_mid < target_work - rebound_work:
                    lo = mid
                else:
                    hi = mid
            step = hi
            delta, before, after, is_sliding = trial_at_step(
                impulse, step, physical, trial_solver, disk_solver
            )
            gained = -0.5 * (before + after) * step
        impulse += delta
        rebound_work += gained
        slipped |= is_sliding
    if not rebound_done and target_work - rebound_work > 1e-12:
        raise RuntimeError("rebound work target not reached within impulse horizon")
    if abs(rebound_work - target_work) > 1e-8:
        raise RuntimeError("energetic restitution postcondition failed")
    return impulse, rebound_work, slipped


_rebound_kernel = (
    numba.njit(cache=True, fastmath=False)(_rebound_source)
    if numba is not None
    else _rebound_source
)


def _integrate_source(
    physical: tuple,
    energetic_restitution: float,
    steps_per_compression: int,
    trial_solver: Callable[..., tuple[np.ndarray, float, float, bool]],
    disk_solver: Callable[..., tuple[np.ndarray, bool]],
    trial_at_step: Callable[..., tuple[np.ndarray, float, float, bool]],
    compression_solver: Callable[..., tuple[np.ndarray, float, bool]],
    rebound_solver: Callable[..., tuple[np.ndarray, float, bool]],
) -> tuple[np.ndarray, float, float, bool]:
    relative_initial, normal, inverse_mass = physical[:3]
    approach = float(relative_initial @ normal)
    base_step = approach / (
        float(normal @ inverse_mass @ normal) * steps_per_compression
    )
    if not math.isfinite(base_step) or base_step <= 0:
        raise RuntimeError("incremental normal impulse step is invalid")
    impulse = np.zeros(3)
    impulse, compression_work, compression_slipped = compression_solver(
        impulse,
        base_step,
        steps_per_compression,
        physical,
        trial_solver,
        disk_solver,
        trial_at_step,
    )
    target_work = energetic_restitution**2 * compression_work
    impulse, rebound_work, rebound_slipped = rebound_solver(
        impulse,
        base_step,
        steps_per_compression,
        physical,
        target_work,
        trial_solver,
        disk_solver,
        trial_at_step,
    )
    return (
        impulse,
        compression_work,
        rebound_work,
        compression_slipped | rebound_slipped,
    )


_compiled_integrator = (
    numba.njit(cache=True, fastmath=False)(_integrate_source)
    if numba is not None
    else None
)


def integrate_impulse(
    inputs: ContactKernelInput,
    *,
    backend: str,
) -> tuple[np.ndarray, float, float, bool]:
    """Dispatch the identical kernel source, never swallowing JIT failures."""
    physical = (
        inputs.relative_initial,
        inputs.normal,
        inputs.inverse_mass,
        inputs.tangent,
        inputs.tangent_mass,
        inputs.eigenvalues,
        inputs.eigenvectors,
        inputs.friction,
    )
    if backend == "auto":
        backend = "numba" if _compiled_integrator is not None else "python_reference"
    if backend == "python_reference":
        return _integrate_source(
            physical,
            inputs.energetic_restitution,
            inputs.steps_per_compression,
            _trial_source,
            _disk_source,
            _trial_at_step_source,
            _compression_source,
            _rebound_source,
        )
    if backend == "numba" and _compiled_integrator is not None:
        return _compiled_integrator(
            physical,
            inputs.energetic_restitution,
            inputs.steps_per_compression,
            _trial_kernel,
            _disk_kernel,
            _trial_at_step_kernel,
            _compression_kernel,
            _rebound_kernel,
        )
    raise ValueError("backend must be available numba or python_reference")


def contact_backend_info() -> dict[str, str | bool | None]:
    """Expose the active numerical implementation for experiment receipts."""
    return {
        "active_backend": "numba" if numba is not None else "python_reference",
        "numba_version": None if numba is None else numba.__version__,
        "llvmlite_version": None if numba is None else numba.llvmlite.__version__,
        "fastmath": False,
    }
