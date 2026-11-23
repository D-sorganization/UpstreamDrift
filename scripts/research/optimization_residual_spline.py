"""Bounded cubic feedforward residuals for optimization research (issue #15493).

This parameterization reuses SciPy's BSpline, not a new dynamics model. Its
coefficient box bounds the residual continuously, including between samples.
Total applied torque, work, tracking and replay still require separate gates.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import BSpline


def sample_residual(
    *,
    times: np.ndarray,
    breakpoints: np.ndarray,
    coefficients: np.ndarray,
    torque_radius: np.ndarray,
    minimum_spacing_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return residual torque (N m) and analytic slew (N m/s), shape (T, J).

    Preconditions: finite, increasing breakpoints in seconds separated by at
    least minimum_spacing_s > 0; coefficients shape (len(breakpoints)+2, J);
    nonnegative per-joint torque_radius shape (J,); abs(coefficients) <= radius.
    Query times are a nonempty vector within the closed support, in any order.
    Endpoint knots are repeated four times; interior knots are simple (C2).

    Postconditions: finite values and slew from the same clamped cubic, with
    no extrapolation or clipping and no input mutation. The convex-hull bound
    applies only to the feedforward perturbation, not feedback or total torque.
    Slew is reported, not limited. Invalid proposals raise ValueError.
    """
    if not math.isfinite(minimum_spacing_s) or minimum_spacing_s <= 0:
        raise ValueError("minimum_spacing_s must be finite and positive")
    arrays = {}
    for name, value in (
        ("times", times),
        ("breakpoints", breakpoints),
        ("coefficients", coefficients),
        ("torque_radius", torque_radius),
    ):
        array = np.asarray(value, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        arrays[name] = array
    t, b = arrays["times"], arrays["breakpoints"]
    c, radius = arrays["coefficients"], arrays["torque_radius"]
    if b.ndim != 1 or b.size < 2:
        raise ValueError("breakpoints must be a vector with at least two entries")
    with np.errstate(over="ignore", invalid="ignore"):
        spacing = np.diff(b)
        span = b[-1] - b[0]
    if (
        not np.isfinite(span)
        or not np.isfinite(spacing).all()
        or np.any(spacing < minimum_spacing_s)
    ):
        raise ValueError("breakpoint spacing must be finite and >= minimum_spacing_s")
    if radius.ndim != 1 or radius.size == 0 or np.any(radius < 0):
        raise ValueError("torque_radius must be a nonempty nonnegative vector")
    if c.shape != (b.size + 2, radius.size):
        raise ValueError("coefficients must have shape (len(breakpoints)+2, joints)")
    if np.any(np.abs(c) > radius):
        raise ValueError("coefficients exceed the per-joint residual trust radius")
    if t.ndim != 1 or t.size == 0:
        raise ValueError("times must be a nonempty vector")
    if np.any(t < b[0]) or np.any(t > b[-1]):
        raise ValueError("times must lie within the closed spline support")
    knots = np.concatenate((np.repeat(b[0], 4), b[1:-1], np.repeat(b[-1], 4)))
    spline = BSpline(knots, c, 3, extrapolate=False)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value = spline(t)
        slew = spline(t, nu=1)
    if not np.isfinite(value).all() or not np.isfinite(slew).all():
        raise ValueError("spline value and slew must remain finite")
    return value, slew
