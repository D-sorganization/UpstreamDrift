"""Engine-agnostic inverse kinematics and marker trajectory tracking (FB-4).

Provides least-squares inverse kinematics solving across motion capture trajectories
given an engine's forward kinematics callable (``pose_fn``) and optional loop-closure
constraint callable (``closure_fn``). Warm-starts sequential frames to guarantee fast
convergence across dense kinematic captures.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from src.shared.python.motion_matching.marker_calibration import Offsets, Pose
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

Array = NDArray[np.float64]
PoseFn = Callable[[Array], Mapping[str, Pose]]
ClosureFn = Callable[[Array], Array]


def solve_full_body_ik_trajectory(
    pose_fn: PoseFn,
    offsets: Offsets,
    capture: TourCapture,
    initial_q: Array,
    *,
    closure_fn: ClosureFn | None = None,
    closure_weight: float = 10.0,
    reg_weight: float = 1e-3,
    max_nfev: int = 50,
) -> Array:
    """Solve inverse kinematics across all frames of a capture trajectory.

    Preconditions:
    - ``initial_q`` is a 1D finite array of coordinates.
    - ``capture.frames >= 1``.
    - Every label in ``capture.labels`` is present in ``offsets``.
    - ``closure_weight >= 0`` and ``reg_weight >= 0``.

    Postconditions:
    - Returns array of shape ``(capture.frames, initial_q.size)``.
    - All elements of the returned array are finite.
    """
    q0 = np.asarray(initial_q, dtype=float)
    if q0.ndim != 1 or not np.isfinite(q0).all():
        raise ValueError("initial_q must be a finite 1D array")
    if capture.frames < 1:
        raise ValueError("capture must contain at least 1 frame")
    missing = [label for label in capture.labels if label not in offsets]
    if missing:
        raise ValueError(f"Capture labels missing from offsets: {missing}")
    if closure_weight < 0:
        raise ValueError("closure_weight must be non-negative")
    if reg_weight < 0:
        raise ValueError("reg_weight must be non-negative")

    n_coords = q0.size
    q_out = np.zeros((capture.frames, n_coords), dtype=float)
    q_curr = q0.copy()

    # Pre-extract marker info for fast inner loop evaluation
    marker_indices = list(range(len(capture.labels)))
    marker_bodies = [offsets[label][0] for label in capture.labels]
    marker_offsets = [
        np.asarray(offsets[label][1], dtype=float) for label in capture.labels
    ]

    for f in range(capture.frames):
        valid_indices = [i for i in marker_indices if capture.valid[f, i]]
        if not valid_indices:
            q_out[f] = q_curr
            continue

        target_points = capture.points_m[f]
        q_ref = q_curr.copy()

        def residual(
            q_eval: Array,
            v_idx: list[int] = valid_indices,
            targets: Array = target_points,
            ref_q: Array = q_ref,
        ) -> Array:
            poses = pose_fn(q_eval)
            diffs = []
            for i in v_idx:
                body = marker_bodies[i]
                offset = marker_offsets[i]
                r, t = poses[body]
                p_pred = r @ offset + t
                diffs.append(p_pred - targets[i])

            res = np.concatenate(diffs)
            if closure_fn is not None and closure_weight > 0.0:
                closure_err = closure_fn(q_eval)
                res = np.concatenate([res, closure_weight * closure_err])
            if reg_weight > 0.0:
                res = np.concatenate([res, reg_weight * (q_eval - ref_q)])
            return res

        sol = least_squares(residual, q_curr, method="lm", max_nfev=max_nfev)
        if np.isfinite(sol.x).all():
            q_curr = sol.x.copy()
        q_out[f] = q_curr

    return q_out


def compute_marker_rms_trajectory(
    pose_fn: PoseFn,
    offsets: Offsets,
    capture: TourCapture,
    q_trajectory: Array,
) -> tuple[Array, dict[str, float], float]:
    """Compute per-frame, per-marker, and overall RMS errors for a solved trajectory.

    Returns:
    - ``rms_per_frame_m``: Array of shape ``(capture.frames,)``.
    - ``rms_per_marker_m``: dict mapping each marker label to its RMS error in metres.
    - ``total_rms_m``: Scalar float representing total root-mean-square error.
    """
    q_traj = np.asarray(q_trajectory, dtype=float)
    if q_traj.shape[0] != capture.frames or q_traj.ndim != 2:
        raise ValueError("q_trajectory rows must match capture frame count")

    per_frame_errors: list[float] = []
    all_errors: list[float] = []
    per_marker_errors: dict[str, list[float]] = {label: [] for label in capture.labels}

    for f in range(capture.frames):
        poses = pose_fn(q_traj[f])
        frame_sq_errors: list[float] = []
        for i, label in enumerate(capture.labels):
            if capture.valid[f, i]:
                body, offset = offsets[label]
                r, t = poses[body]
                p_pred = r @ np.asarray(offset, dtype=float) + t
                err = float(np.linalg.norm(p_pred - capture.points_m[f, i]))
                all_errors.append(err)
                frame_sq_errors.append(err**2)
                per_marker_errors[label].append(err)

        if frame_sq_errors:
            per_frame_errors.append(float(np.sqrt(np.mean(frame_sq_errors))))
        else:
            per_frame_errors.append(0.0)

    per_marker_rms = {
        label: float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0
        for label, errs in per_marker_errors.items()
    }
    total_rms = float(np.sqrt(np.mean(np.square(all_errors)))) if all_errors else 0.0

    return np.asarray(per_frame_errors, dtype=float), per_marker_rms, total_rms
