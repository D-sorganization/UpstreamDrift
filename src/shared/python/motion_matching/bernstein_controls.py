"""Pure bounded Bernstein controls shared by constrained motion fitters.

Each actuator is represented by a row of Bernstein control points.  The
non-negative basis sums to one on the normalized horizon, so every evaluated
control remains within the row's control-point bounds.
"""

from __future__ import annotations

import math

import numpy as np


def _validated_controls(control_points: np.ndarray) -> np.ndarray:
    """Return finite two-dimensional actuator controls with a usable degree."""
    controls = np.asarray(control_points, dtype=np.float64)
    if controls.ndim != 2:
        raise ValueError("control_points must be a two-dimensional array")
    if controls.shape[0] < 1:
        raise ValueError("control_points must contain at least one actuator")
    if controls.shape[1] < 2:
        raise ValueError("control_points must contain at least two control points")
    if not np.isfinite(controls).all():
        raise ValueError("control_points must be finite")
    return controls


def _basis(coefficient_count: int, normalized_time: float) -> np.ndarray:
    """Return the degree-``coefficient_count - 1`` Bernstein basis vector."""
    if not np.isfinite(normalized_time):
        raise ValueError("normalized_time must be finite")
    s = float(np.clip(normalized_time, 0.0, 1.0))
    degree = coefficient_count - 1
    indices = np.arange(coefficient_count, dtype=np.float64)
    binomial = np.asarray(
        [math.comb(degree, int(index)) for index in indices], dtype=np.float64
    )
    return binomial * s**indices * (1.0 - s) ** (degree - indices)


def evaluate_bernstein_controls(
    control_points: np.ndarray,
    normalized_time: float,
) -> np.ndarray:
    """Evaluate continuous bounded actuator controls on a normalized horizon.

    ``normalized_time`` is clamped to ``[0, 1]`` deliberately, making the
    endpoint control values valid at the edges of an ODE solver's evaluation
    interval.  The returned array has one torque per actuator row.
    """
    controls = _validated_controls(control_points)
    return controls @ _basis(controls.shape[1], normalized_time)


def bernstein_curvature_penalty(
    control_points: np.ndarray,
    *,
    weight: float,
) -> np.ndarray:
    """Return weighted second-difference residuals ordered by actuator row."""
    controls = _validated_controls(control_points)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("weight must be finite and non-negative")
    if weight == 0.0:
        return np.zeros(0, dtype=np.float64)
    curvature = controls[:, 2:] - 2.0 * controls[:, 1:-1] + controls[:, :-2]
    return math.sqrt(weight) * curvature.reshape(-1)


def bernstein_effort_penalty(
    control_points: np.ndarray,
    *,
    weight: float,
    scale: float,
) -> np.ndarray:
    """Return weighted, scaled control-point effort residuals."""
    controls = _validated_controls(control_points)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("weight must be finite and non-negative")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    if weight == 0.0:
        return np.zeros(0, dtype=np.float64)
    return math.sqrt(weight) * controls.reshape(-1) / scale
