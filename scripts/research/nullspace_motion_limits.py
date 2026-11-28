"""Audit declared motion priors for #15494; no allocation or replay claim.

The caller supplies model-specific, unwrapped angular coordinates and sourced
limits. Piecewise-linear interpolation makes constant ROM and affine posture
limits testable at endpoints and speed limits testable on whole segments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.loop_nullspace.evidence import check_shape, readonly_float


def _margins(values: NDArray[np.float64], location: str) -> dict[str, Any]:
    """Describe signed margins without clipping or mixing unlike units."""
    index, axis = np.unravel_index(int(np.argmin(values)), values.shape)
    return {
        "minimum_margin": np.min(values, axis=0).tolist(),
        "narrowest": {
            location: int(index),
            "axis": int(axis),
            "margin": float(values[index, axis]),
        },
        "violations": [
            {location: int(i), "axis": int(j), "margin": float(values[i, j])}
            for i, j in zip(*np.nonzero(values < 0), strict=True)
        ],
    }


@dataclass(frozen=True)
class MotionLimitPriors:
    """Caller-sourced ROM, speed and optional coupled posture declarations.

    The audit validates these declarations against the supplied motion; no
    anatomical defaults or qualification are implied by constructing them.
    """

    lower_rad: ArrayLike
    upper_rad: ArrayLike
    max_speed_rad_s: ArrayLike
    safety_margin_rad: float = 0.0
    coupled_matrix: ArrayLike | None = None
    coupled_upper_rad: ArrayLike | None = None


def _validate_declarations(
    times_s: ArrayLike,
    q_rad: ArrayLike,
    coordinate_order: Sequence[str],
    limit_source: str,
    priors: MotionLimitPriors,
) -> tuple[NDArray[np.float64], ...]:
    """Validate declarations and return finite arrays with compatible shapes."""
    lower_rad, upper_rad = priors.lower_rad, priors.upper_rad
    max_speed_rad_s = priors.max_speed_rad_s
    safety_margin_rad = priors.safety_margin_rad
    coupled_matrix, coupled_upper_rad = priors.coupled_matrix, priors.coupled_upper_rad
    times = readonly_float(times_s, "times_s")
    q = readonly_float(q_rad, "q_rad")
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times_s must be 1-D with at least two samples")
    if q.ndim != 2 or q.shape[0] != times.size or q.shape[1] == 0:
        raise ValueError("q_rad must be (samples, nonempty coordinates)")
    joints = q.shape[1]
    if isinstance(coordinate_order, (str, bytes)) or len(coordinate_order) != joints:
        raise ValueError("coordinate_order must name every coordinate")
    if any(not isinstance(x, str) or not x.strip() for x in coordinate_order):
        raise ValueError("coordinate names must be nonempty strings")
    if len(set(coordinate_order)) != joints:
        raise ValueError("coordinate names must be unique")
    if not isinstance(limit_source, str) or not limit_source.strip():
        raise ValueError("limit_source must declare the source of these priors")
    lo = readonly_float(lower_rad, "lower_rad")
    hi = readonly_float(upper_rad, "upper_rad")
    speed_limit = readonly_float(max_speed_rad_s, "max_speed_rad_s")
    for values, name in (
        (lo, "lower_rad"),
        (hi, "upper_rad"),
        (speed_limit, "max_speed_rad_s"),
    ):
        check_shape(values, (joints,), name)
    if not np.isfinite(safety_margin_rad) or safety_margin_rad < 0:
        raise ValueError("safety_margin_rad must be finite and nonnegative")
    if np.any(speed_limit <= 0):
        raise ValueError("speed limits must be positive")
    if (coupled_matrix is None) != (coupled_upper_rad is None):
        raise ValueError("coupled_matrix and coupled_upper_rad are a pair")

    return times, q, lo, hi, speed_limit


def audit_motion_limits(
    times_s: ArrayLike,
    q_rad: ArrayLike,
    *,
    coordinate_order: Sequence[str],
    priors: MotionLimitPriors,
    limit_source: str,
) -> dict[str, Any]:
    """Check unwrapped angular motion against explicit constant priors.

    Preconditions: finite q of shape (samples, joints), increasing seconds,
    unique coordinate names, nonempty source, ordered per-joint ROM and
    positive speed bounds. Optional coupled inequalities are A q <= b;
    A is dimensionless, b is in radians, and rows must be nonzero. Both
    coupled inputs must be supplied together. No anatomical defaults exist.

    A nonnegative margin shrinks ROM at both ends and offsets each coupled
    plane inward by that distance in radians; an empty ROM is rejected.
    Postconditions: signed minimum margins, all violating samples/segments,
    units, priors and interpolation are retained in strict-JSON-compatible
    data. A pass applies only to the declared piecewise-linear path, not
    spline/native replay or acceleration at knots. Limits are caller-sourced,
    not independently verified; no clinical safety or feasibility is certified.

    Raises ValueError for invalid declarations or nonfinite derived arithmetic.
    """
    times, q, lo, hi, speed_limit = _validate_declarations(
        times_s, q_rad, coordinate_order, limit_source, priors
    )
    safety_margin_rad = priors.safety_margin_rad
    coupled_matrix, coupled_upper_rad = priors.coupled_matrix, priors.coupled_upper_rad
    joints = q.shape[1]
    coupled = None
    coupled_priors = None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            dt = np.diff(times)
            if np.any(dt <= 0):
                raise ValueError("times_s must be strictly increasing")
            if np.any(lo + safety_margin_rad >= hi - safety_margin_rad):
                raise ValueError("ROM must have a nonempty interior after margin")
            rom = _margins(np.minimum(q - lo, hi - q) - safety_margin_rad, "sample")
            speeds = np.abs(np.diff(q, axis=0) / dt[:, None])
            speed = _margins(speed_limit - speeds, "segment")
            if coupled_matrix is not None and coupled_upper_rad is not None:
                a = readonly_float(coupled_matrix, "coupled_matrix")
                b = readonly_float(coupled_upper_rad, "coupled_upper_rad")
                if a.ndim != 2 or a.shape[1] != joints or a.shape[0] == 0:
                    raise ValueError("coupled_matrix must be (nonempty rows, joints)")
                check_shape(b, (a.shape[0],), "coupled_upper_rad")
                norms = np.linalg.norm(a, axis=1)
                if np.any(norms == 0):
                    raise ValueError("coupled rows must be nonzero")
                slack = b / norms - q @ (a / norms[:, None]).T - safety_margin_rad
                coupled = _margins(slack, "sample")
                coupled_priors = {"matrix": a.tolist(), "upper_rad": b.tolist()}
    except FloatingPointError as exc:
        raise ValueError("motion-limit arithmetic must remain finite") from exc

    return {
        "schema": "nullspace-motion-limits/1",
        "times_s": times.tolist(),
        "coordinate_order": list(coordinate_order),
        "limit_source": limit_source,
        "priors": {
            "lower_rad": lo.tolist(),
            "upper_rad": hi.tolist(),
            "max_speed_rad_s": speed_limit.tolist(),
            "safety_margin_rad": float(safety_margin_rad),
            "coupled": coupled_priors,
        },
        "units": {"rom": "rad", "coupled_posture": "rad", "speed": "rad/s"},
        "interpolation": "piecewise_linear_unwrapped_radians",
        "rom": rom,
        "speed": speed,
        "peak_speed_rad_s": np.max(speeds, axis=0).tolist(),
        "coupled_posture": coupled,
        "declared_limits_satisfied": all(
            not item["violations"] for item in (rom, speed, coupled) if item is not None
        ),
        "forward_replay_verified": False,
        "physiological_feasibility_verified": False,
    }
