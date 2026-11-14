"""Inverse kinematics convergence types, contracts, and trajectory solver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from src.shared.python.motion_matching.marker_calibration import Offsets, PoseFn
from src.shared.python.motion_matching.tour_capture_contract import TourCapture
from src.shared.python.motion_matching.trajectory_ik_options import (
    CoordinateBounds,
    TrajectoryIkOptions,
    resolve_trajectory_options,
)

Array: TypeAlias = NDArray[np.float64]


class IKTermination(str, Enum):
    """Termination condition for an inverse kinematics solve."""

    CONVERGED = "converged"
    ITERATION_CAP = "iteration_cap"
    DAMPING_DIVERGED = "damping_diverged"
    NON_FINITE = "non_finite"
    NO_MARKERS = "no_markers"
    AT_BOUND = "at_bound"


@dataclass(frozen=True)
class IKFrameReport:
    """Convergence and quality metrics for a single trajectory IK frame."""

    frame: int
    termination: IKTermination
    iterations: int
    rms_error_m: float
    valid_marker_count: int
    converged: bool


@dataclass(frozen=True)
class IKTrajectory:
    """Solved joint trajectory and per-frame convergence reports."""

    q: Array
    reports: list[IKFrameReport]

    def __iter__(self) -> Iterator[Array | list[IKFrameReport]]:
        return iter((self.q, self.reports))


def classify_ik_termination(
    sol: Any,
    has_bounds: bool = False,
    low: Array | None = None,
    high: Array | None = None,
) -> tuple[IKTermination, bool]:
    """Classify the termination condition of a SciPy least_squares result."""
    if not np.isfinite(sol.x).all():
        return IKTermination.NON_FINITE, False
    if getattr(sol, "active_mask", None) is not None and np.any(sol.active_mask != 0):
        return IKTermination.AT_BOUND, False
    if has_bounds and low is not None and high is not None:
        if np.any(np.isclose(sol.x, low, 1e-6) | np.isclose(sol.x, high, 1e-6)):
            return IKTermination.AT_BOUND, False
    return (
        (IKTermination.CONVERGED, True)
        if sol.success
        else (IKTermination.ITERATION_CAP, False)
    )


def _trajectory_bounds(
    initial_q: Array,
    bounds: CoordinateBounds | tuple[Sequence[float], Sequence[float]] | None,
) -> tuple[Array, Array]:
    """Validate explicit finite bounds before any engine geometry evaluation."""
    if bounds is None:
        return np.full_like(initial_q, -np.inf), np.full_like(initial_q, np.inf)
    low, high = (np.asarray(v, dtype=float) for v in bounds)
    if not (
        low.shape == initial_q.shape == high.shape
        and np.isfinite(low).all()
        and np.isfinite(high).all()
        and np.all(low < high)
        and np.all((low <= initial_q) & (initial_q <= high))
    ):
        raise ValueError(
            "Coordinate bounds must be finite, ordered and contain initial_q"
        )
    return low.copy(), high.copy()


def solve_full_body_ik_trajectory_with_report(
    pose_fn: PoseFn,
    offsets: Offsets,
    capture: TourCapture,
    initial_q: Array,
    *,
    options: TrajectoryIkOptions | None = None,
    **option_overrides: Any,
) -> IKTrajectory:
    """Solve trajectory IK with per-frame convergence reports."""
    q0 = np.asarray(initial_q, dtype=float)
    if q0.ndim != 1 or not np.isfinite(q0).all():
        raise ValueError("initial_q must be a finite 1D array")
    if capture.frames < 1:
        raise ValueError("capture must contain at least 1 frame")
    missing = [lbl for lbl in capture.labels if lbl not in offsets]
    if missing:
        raise ValueError(f"Capture labels missing from offsets: {missing}")
    opts = resolve_trajectory_options(options, option_overrides)
    c_fn, c_wt, reg_wt = opts.closure_fn, opts.closure_weight, opts.reg_weight
    max_nfev, has_b = opts.max_nfev, opts.coordinate_bounds is not None
    low, high = _trajectory_bounds(q0, opts.coordinate_bounds)
    method = "trf" if (opts.method == "trf" or has_b) else (opts.method or "lm")

    q_out = np.zeros((capture.frames, q0.size), dtype=float)
    q_curr, reports = q0.copy(), []
    b_names = [offsets[lbl][0] for lbl in capture.labels]
    b_offs = [np.asarray(offsets[lbl][1], dtype=float) for lbl in capture.labels]

    for f in range(capture.frames):
        v_idx = [i for i, v in enumerate(capture.valid[f]) if v]
        if not v_idx:
            q_out[f] = q_curr
            reports.append(
                IKFrameReport(f, IKTermination.NO_MARKERS, 0, float("nan"), 0, False)
            )
            continue

        tgt, q_ref = capture.points_m[f], q_curr.copy()

        def residual(q: Array, v=v_idx, t=tgt, qr=q_ref) -> Array:
            p = pose_fn(q)
            diffs = [p[b_names[i]][0] @ b_offs[i] + p[b_names[i]][1] - t[i] for i in v]
            parts = [np.concatenate(diffs)]
            if c_fn is not None and c_wt > 0.0:
                parts.append(c_wt * c_fn(q))
            if reg_wt > 0.0:
                parts.append(reg_wt * (q - qr))
            return np.concatenate(parts) if len(parts) > 1 else parts[0]

        ftol = 1e-8 if (method == "lm" and not has_b) else None
        sol = least_squares(
            residual,
            q_curr,
            method=method,
            bounds=(low, high),
            max_nfev=max_nfev,
            ftol=ftol,
        )
        rms = float(np.sqrt(np.mean(sol.fun[: len(v_idx) * 3] ** 2)))
        term, conv = classify_ik_termination(sol, has_bounds=has_b, low=low, high=high)
        q_curr = sol.x.copy() if np.isfinite(sol.x).all() else q_curr
        q_out[f] = q_curr
        nev = int(getattr(sol, "nfev", 0))
        reports.append(IKFrameReport(f, term, nev, rms, len(v_idx), conv))

    return IKTrajectory(q=q_out, reports=reports)


def solve_full_body_ik_trajectory(*args: Any, **kwargs: Any) -> Array:
    """Solve trajectory IK; explicit finite source bounds opt into bounded TRF."""
    return solve_full_body_ik_trajectory_with_report(*args, **kwargs).q
