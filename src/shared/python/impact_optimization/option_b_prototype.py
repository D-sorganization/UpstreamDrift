"""Research-only fixed-gain control and bounded B-spline perturbations.

The caller must bind vector channels, state units, model, and control clock.
Outputs are ideal command and clipped input, not measured actuator torque.
No rollout, physical feasibility, or between-sample guarantee is implied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline

Array = NDArray[np.float64]


def _finite(value: object, name: str, ndim: int) -> Array:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError
        out = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if out.ndim != ndim or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite and {ndim}-dimensional")
    out.setflags(write=False)
    return out


class SplineResidual:
    """Clamped cubic B-spline constrained by a scaled coefficient L2 ball.

    ``breakpoints_s`` include endpoints and obey minimum spacing; the open
    cubic knot vector repeats each endpoint four times. The trust ball applies
    to the full coefficient matrix after division by per-channel ``scale_nm``.
    It is not a work, slew, total-torque or anatomical constraint.
    """

    def __init__(
        self,
        *,
        breakpoints_s: Array,
        coefficients_nm: Array,
        scale_nm: Array,
        trust_radius: float,
        min_spacing_s: float,
    ) -> None:
        points = _finite(breakpoints_s, "breakpoints", 1)
        coeff = _finite(coefficients_nm, "coefficients", 2)
        scale = _finite(scale_nm, "scale", 1)
        if not np.isfinite(min_spacing_s) or min_spacing_s <= 0.0:
            raise ValueError("minimum knot spacing must be finite and positive")
        if len(points) < 2 or np.any(np.diff(points) < min_spacing_s):
            raise ValueError("breakpoints must be strictly spaced")
        if not np.isfinite(trust_radius) or trust_radius <= 0.0:
            raise ValueError("trust radius must be finite and positive")
        if not len(scale) or np.any(scale <= 0.0):
            raise ValueError("scale must be positive")
        if coeff.shape != (len(points) + 2, len(scale)):
            raise ValueError("cubic coefficient shape does not match knots/channels")
        with np.errstate(over="ignore", invalid="ignore"):
            norm = float(np.linalg.norm(coeff / scale))
        if not np.isfinite(norm) or norm > trust_radius:
            raise ValueError("scaled coefficient trust radius exceeded")
        knots = np.concatenate(
            (np.repeat(points[0], 4), points[1:-1], np.repeat(points[-1], 4))
        )
        self._spline = BSpline(knots, coeff, 3, extrapolate=False)
        self._start, self._end = float(points[0]), float(points[-1])
        self._channels = len(scale)

    def value(self, time_s: float) -> Array:
        if not np.isfinite(time_s) or not self._start <= time_s <= self._end:
            raise ValueError("time must lie within the spline interval")
        result = np.asarray(self._spline(time_s), dtype=float)
        if result.shape != (self._channels,) or not np.isfinite(result).all():
            raise ValueError("spline result must be finite")
        return result


@dataclass(frozen=True)
class ControlDecision:
    """Total command and ideal clipped input for one caller-selected hold."""

    commanded_nm: Array
    applied_nm: Array
    saturated: NDArray[np.bool_]


class FixedGainController:
    """Option B at a single sample; the caller implements the declared hold."""

    def __init__(
        self,
        *,
        times_s: Array,
        q_ref: Array,
        qd_ref: Array,
        tau_ff_ref: Array,
        kp: Array,
        kd: Array,
        lower_nm: Array,
        upper_nm: Array,
        residual: SplineResidual,
    ) -> None:
        self._times = _finite(times_s, "times", 1)
        self._q = _finite(q_ref, "q_ref", 2)
        self._qd = _finite(qd_ref, "qd_ref", 2)
        self._ff = _finite(tau_ff_ref, "tau_ff_ref", 2)
        self._kp = _finite(kp, "kp", 1)
        self._kd = _finite(kd, "kd", 1)
        self._lower = _finite(lower_nm, "lower", 1)
        self._upper = _finite(upper_nm, "upper", 1)
        n = len(self._kp)
        if len(self._times) < 2 or np.any(np.diff(self._times) <= 0.0):
            raise ValueError("reference times must be strictly increasing")
        if not n or any(
            x.shape != (len(self._times), n) for x in (self._q, self._qd, self._ff)
        ):
            raise ValueError("reference tables must match times and channels")
        if any(x.shape != (n,) for x in (self._kd, self._lower, self._upper)):
            raise ValueError("gain and limit vectors must match channels")
        if np.any(self._kp < 0.0) or np.any(self._kd < 0.0):
            raise ValueError("fixed gains must be nonnegative")
        if np.any(self._lower >= self._upper):
            raise ValueError("saturation lower bound must be below upper bound")
        if not isinstance(residual, SplineResidual) or residual._channels != n:
            raise ValueError("spline channels must match controller")
        if residual._start != self._times[0] or residual._end != self._times[-1]:
            raise ValueError("reference and spline horizons must match")
        self._residual = residual

    def evaluate(self, time_s: float, q: Array, qd: Array) -> ControlDecision:
        """Return FF + spline + fixed PD, then clip the complete command."""
        if not np.isfinite(time_s) or not self._times[0] <= time_s <= self._times[-1]:
            raise ValueError("time must lie within reference horizon")
        q_now, qd_now = _finite(q, "q", 1), _finite(qd, "qd", 1)
        n = len(self._kp)
        if q_now.shape != (n,) or qd_now.shape != (n,):
            raise ValueError("state channels must match controller")

        def sample(table: Array) -> Array:
            return np.array(
                [np.interp(time_s, self._times, table[:, j]) for j in range(n)]
            )

        with np.errstate(over="ignore", invalid="ignore"):
            commanded = (
                sample(self._ff)
                + self._residual.value(time_s)
                + self._kp * (sample(self._q) - q_now)
                + self._kd * (sample(self._qd) - qd_now)
            )
        if not np.isfinite(commanded).all():
            raise ValueError("total command must be finite")
        applied = np.clip(commanded, self._lower, self._upper)
        return ControlDecision(commanded, applied, commanded != applied)
