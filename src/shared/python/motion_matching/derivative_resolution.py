"""Numerical derivative resolution and measured derivative floors (#10069).

Provides utilities for determining scale-aware finite-difference perturbation
steps and measuring truncation vs. roundoff noise floors on stiff or contact-coupled
forward simulation objectives.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]

DEFAULT_FD_STEPS: tuple[float, ...] = (
    1e-1,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
    1e-9,
    1e-10,
    1e-11,
    1e-12,
)


@dataclass(frozen=True)
class DerivativeResolutionResult:
    """Diagnostic outcome of derivative floor and step resolution measurement."""

    optimal_step: float
    noise_floor: float
    measured_slopes: tuple[float, ...]
    steps: tuple[float, ...]
    relative_errors: tuple[float, ...]
    is_resolved: bool
    status: str


def compute_finite_difference_step_vector(
    x0: Array,
    *,
    default_step: float = 1e-4,
    floor: float = 1e-8,
) -> Array:
    """Compute scale-aware finite-difference perturbation vector.

    h_i = max(floor, default_step * (1.0 + |x0[i]|)).

    Preconditions:
    - x0 must be 1D finite float array.
    - default_step > 0, floor > 0.
    """
    arr = np.asarray(x0, dtype=np.float64)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError("x0 must be a finite 1D array")
    if default_step <= 0.0 or floor <= 0.0:
        raise ValueError("default_step and floor must be strictly positive")

    scaled = default_step * (1.0 + np.abs(arr))
    return np.maximum(scaled, floor)


def _validate_steps(steps: Sequence[float] | None) -> tuple[float, ...]:
    eval_steps = DEFAULT_FD_STEPS if steps is None else tuple(float(s) for s in steps)
    if len(eval_steps) < 2:
        raise ValueError("steps must contain at least 2 entries")
    for i, s in enumerate(eval_steps):
        if s <= 0.0 or not math.isfinite(s):
            raise ValueError(f"step {s} must be strictly positive and finite")
        if i > 0 and s >= eval_steps[i - 1]:
            raise ValueError("steps must be strictly decreasing")
    return eval_steps


def _compute_perturbed_slopes(
    f: Callable[[Array], Array],
    arr: Array,
    y0: Array,
    component_idx: int,
    eval_steps: tuple[float, ...],
) -> list[float]:
    n = len(arr)
    slopes: list[float] = []
    unit_vec = np.zeros(n, dtype=np.float64)
    unit_vec[component_idx] = 1.0

    for h in eval_steps:
        x_perturbed = arr + h * unit_vec
        yh = np.asarray(f(x_perturbed), dtype=np.float64)
        if not np.isfinite(yh).all():
            slopes.append(float("nan"))
            continue
        diff = (yh[0] - y0[0]) / h
        slopes.append(float(diff))
    return slopes


def measure_derivative_floor(
    f: Callable[[Array], Array],
    x0: Array,
    *,
    steps: Sequence[float] | None = None,
    component_idx: int = 0,
    rel_tol: float = 1e-2,
) -> DerivativeResolutionResult:
    """Measure derivative floor and optimal finite difference step for a scalar function f(x)[0].

    Preconditions:
    - x0 must be 1D finite float array.
    - 0 <= component_idx < len(x0).
    - steps must be strictly decreasing positive floats.
    """
    arr = np.asarray(x0, dtype=np.float64)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError("x0 must be a finite 1D array")
    n = len(arr)
    if not (0 <= component_idx < n):
        raise ValueError(f"component_idx {component_idx} out of range [0, {n})")

    eval_steps = _validate_steps(steps)

    y0 = np.asarray(f(arr), dtype=np.float64)
    if y0.size == 0 or not np.isfinite(y0).all():
        raise ValueError("f(x0) produced non-finite output")

    slopes = _compute_perturbed_slopes(f, arr, y0, component_idx, eval_steps)

    valid_pairs = [
        (s, slope)
        for s, slope in zip(eval_steps, slopes, strict=True)
        if math.isfinite(slope)
    ]
    if not valid_pairs:
        return DerivativeResolutionResult(
            optimal_step=eval_steps[0],
            noise_floor=float("nan"),
            measured_slopes=tuple(slopes),
            steps=eval_steps,
            relative_errors=tuple(float("nan") for _ in eval_steps),
            is_resolved=False,
            status="All perturbed evaluations failed or non-finite",
        )

    # Reference slope estimation from median of middle thirds
    valid_slopes = [p[1] for p in valid_pairs]
    mid_start = max(0, len(valid_slopes) // 4)
    mid_end = max(mid_start + 1, (3 * len(valid_slopes)) // 4)
    ref_slope = float(np.median(valid_slopes[mid_start:mid_end]))

    denom = abs(ref_slope) if abs(ref_slope) > 1e-6 else 1.0
    rel_errors: list[float] = []
    for s in slopes:
        if math.isfinite(s):
            rel_errors.append(abs(s - ref_slope) / denom)
        else:
            rel_errors.append(float("inf"))

    # Noise floor: detect where cancellation noise begins to dominate for small h
    noise_floor = 0.0
    for i in range(len(eval_steps) - 1, 0, -1):
        if rel_errors[i] > rel_errors[i - 1] and rel_errors[i] > rel_tol:
            noise_floor = max(noise_floor, eval_steps[i])

    # Optimal step: minimum relative error
    best_idx = int(np.argmin(rel_errors))
    optimal_step = eval_steps[best_idx]
    is_resolved = bool(rel_errors[best_idx] <= rel_tol)

    status = (
        f"Resolved at h={optimal_step:.2e} with rel_err={rel_errors[best_idx]:.2e}"
        if is_resolved
        else f"Unresolved: best rel_err={rel_errors[best_idx]:.2e} exceeds tolerance"
    )

    return DerivativeResolutionResult(
        optimal_step=optimal_step,
        noise_floor=noise_floor,
        measured_slopes=tuple(slopes),
        steps=eval_steps,
        relative_errors=tuple(rel_errors),
        is_resolved=is_resolved,
        status=status,
    )
