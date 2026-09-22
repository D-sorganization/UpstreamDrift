"""Pure bounded Bernstein controls shared by constrained motion fitters.

Each actuator is represented by a row of Bernstein control points.  The
non-negative basis sums to one on the normalized horizon, so every evaluated
control remains within the row's control-point bounds.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult, least_squares


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


@dataclass(frozen=True)
class BernsteinFitResidualPolicy:
    """Shared weights and assembly convention for a bounded control fit."""

    curvature_weight: float
    effort_weight: float
    effort_scale: float

    def assemble(
        self,
        tracking_residual: np.ndarray,
        control_points: np.ndarray,
    ) -> np.ndarray:
        """Combine tracking and regularization residuals for one control vector.

        The tracking vector is supplied by model-specific forward kinematics;
        Bernstein smoothness and effort retain one numerical convention.
        """
        tracking = np.asarray(tracking_residual, dtype=np.float64)
        if tracking.ndim != 1 or not np.isfinite(tracking).all():
            raise ValueError("tracking_residual must be a finite one-dimensional array")
        curvature = bernstein_curvature_penalty(
            control_points, weight=self.curvature_weight
        )
        effort = bernstein_effort_penalty(
            control_points, weight=self.effort_weight, scale=self.effort_scale
        )
        return np.concatenate((tracking, curvature, effort))


def solve_bounded_bernstein_least_squares(
    residual_function: Callable[[np.ndarray], np.ndarray],
    initial_parameters: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    max_evaluations: int,
) -> OptimizeResult:
    """Solve one bounded Bernstein fit with the program's shared tolerances."""
    initial = np.asarray(initial_parameters, dtype=np.float64)
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if initial.ndim != 1 or not np.isfinite(initial).all():
        raise ValueError("initial_parameters must be a finite one-dimensional array")
    if lower.shape != initial.shape or upper.shape != initial.shape:
        raise ValueError("bounds must match initial_parameters")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("bounds must be finite")
    if np.any(lower >= upper):
        raise ValueError("each lower bound must be less than its upper bound")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")
    return least_squares(
        residual_function,
        initial,
        bounds=(lower, upper),
        max_nfev=max(max_evaluations, 5),
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
    )
