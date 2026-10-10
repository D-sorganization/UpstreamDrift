"""Gaze-schedule neck tracking for the forward-dynamics replay (OSV-3 scope 3, #11729).

The replay's computed-torque tracker is the neck PD/torque controller. This
module gives it a gaze-schedule neck reference (the three neck coordinates are
solved so the head gaze axis follows the schedule of
:mod:`gaze_residual`) and reports how well the head follows the schedule, both
for the tracked reference and for the replayed motion. The gaze-driven neck is a
modelled behaviour, never measured data: head markers do not drive it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from src.shared.python.motion_matching import gaze
from src.shared.python.motion_matching.pipeline.constants import TRACKING_CUTOFF_HZ
from src.shared.python.motion_matching.pipeline.gaze_residual import (
    HEAD_FRAME,
    GazePlan,
    plan_from_pass,
    schedule_directions,
)
from src.shared.python.motion_matching.pipeline.reference import smooth_reference
from src.shared.python.motion_matching.range_of_motion import UPPER_RANGES_DEG

Array = NDArray[np.float64]

NECK_REFERENCES = ("ik", "gaze")
#: Schedule evaluations in :func:`apply_fd_neck` (the first from the reference
#: neck, the second from the solved neck).
EYE_COUPLING_PASSES = 2
_BOUND_TOL_RAD = 1e-6
_WINDOWS = ("address_to_impact", "hold", "release", "after_release")


@dataclass(frozen=True)
class NeckSolve:
    """Per-frame neck solve: ``q`` plus clamping and residual statistics."""

    q: Array
    clamped_frames: int
    max_residual_deg: float
    rms_residual_deg: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "clamped_frames": int(self.clamped_frames),
            "max_residual_deg": float(self.max_residual_deg),
            "rms_residual_deg": float(self.rms_residual_deg),
        }


def _angle_deg(a: Array, b: Array) -> Array:
    """Row-wise angle in degrees between vectors ``a`` and ``b`` (frames, 3)."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def _neck_columns(kin: Any) -> list[int]:
    order = list(kin.coordinate_order)
    missing = [n for n in gaze.NECK_COORDINATES if n not in order]
    if missing:
        raise ValueError(f"kinematics lacks neck coordinate {missing[0]!r}")
    return [order.index(n) for n in gaze.NECK_COORDINATES]


def _neck_bounds(
    ranges_deg: dict[str, tuple[float, float]] | None,
) -> tuple[Array, Array]:
    table = UPPER_RANGES_DEG if ranges_deg is None else ranges_deg
    lo = np.radians([table[n][0] for n in gaze.NECK_COORDINATES])
    hi = np.radians([table[n][1] for n in gaze.NECK_COORDINATES])
    return lo, hi


def solve_neck_schedule(
    kin: Any,
    q: Array,
    directions: Array,
    axis_head: Array,
    *,
    ranges_deg: dict[str, tuple[float, float]] | None = None,
    prior_weight: float = 1e-2,
) -> NeckSolve:
    """Neck coordinates that point the head ``axis_head`` along ``directions``.

    Per frame a bounded least-squares solve over the three neck coordinates
    uses the kinematics' own forward kinematics for the head rotation, so it is
    exact for any joint frames. A weak prior toward the reference neck angles
    fixes the free roll about the gaze axis with minimal change.

    Postconditions: the result equals ``q`` except in the neck columns; the
    neck stays inside ``ranges_deg`` (default ``UPPER_RANGES_DEG``); a frame is
    clamped when any neck coordinate is within 1e-6 rad of a bound; residuals
    are the angle in degrees between the head axis and the direction.
    """
    q = np.asarray(q, dtype=float)
    directions = np.asarray(directions, dtype=float)
    axis = np.asarray(axis_head, dtype=float)
    if q.ndim != 2 or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite 2-D array (frames, coordinates)")
    if directions.shape != (q.shape[0], 3) or not np.all(np.isfinite(directions)):
        raise ValueError(
            f"directions must be finite with shape (frames={q.shape[0]}, 3), "
            f"got {directions.shape}"
        )
    if np.any(np.linalg.norm(directions, axis=1) < 1e-12):
        raise ValueError("directions must be nonzero")
    if axis.shape != (3,) or np.linalg.norm(axis) < 1e-12:
        raise ValueError("axis_head must be a nonzero 3-vector")
    if not np.isfinite(prior_weight) or prior_weight < 0:
        raise ValueError(f"prior_weight must be finite and >= 0, got {prior_weight}")
    cols = _neck_columns(kin)
    lo, hi = _neck_bounds(ranges_deg)
    axis = axis / np.linalg.norm(axis)
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    out = q.copy()
    clamped = 0
    residuals = np.zeros(len(q))
    x = np.clip(q[0, cols], lo, hi)
    for k in range(len(q)):
        row = q[k].copy()
        ref = q[k, cols].copy()

        def resid(xk: Array, row: Array = row, ref: Array = ref, k: int = k) -> Array:
            row[cols] = xk
            rot = kin.body_poses(row, [HEAD_FRAME])[HEAD_FRAME][0]
            return np.concatenate([rot @ axis - unit[k], prior_weight * (xk - ref)])

        x = np.clip(x, lo, hi)
        x = least_squares(resid, x, bounds=(lo, hi)).x
        x = np.clip(x, lo, hi)
        out[k, cols] = x
        clamped += int(np.any((x - lo < _BOUND_TOL_RAD) | (hi - x < _BOUND_TOL_RAD)))
        row[cols] = x
        rot = kin.body_poses(row, [HEAD_FRAME])[HEAD_FRAME][0]
        residuals[k] = float(_angle_deg((rot @ axis)[None], unit[k][None])[0])
    return NeckSolve(
        q=out,
        clamped_frames=clamped,
        max_residual_deg=float(residuals.max()),
        rms_residual_deg=float(np.sqrt(np.mean(residuals**2))),
    )


def _plan(lane: Any, kin: Any, q_ref: Array) -> GazePlan:
    """The lane's gaze plan, or the one the IK receipt uses (same impact)."""
    plan = lane.gaze_plan
    if plan is None:
        plan, _ = plan_from_pass(lane, kin, q_ref)
    return plan


def _window_stats(err: Array, mask: Array) -> dict[str, Any]:
    n = int(mask.sum())
    if n == 0:
        return {"frames": 0, "rms_deg": None, "max_deg": None}
    sel = err[mask]
    return {
        "frames": n,
        "rms_deg": float(np.sqrt(np.mean(sel**2))),
        "max_deg": float(sel.max()),
    }


def schedule_tracking(plan: GazePlan, kin: Any, q: Array, times: Any) -> dict[str, Any]:
    """Head-axis error against the gaze schedule, per phase window.

    Windows (by time): ``address_to_impact`` (t <= t_impact), ``hold`` (up to
    t_impact + t_hold), ``release`` (up to t_impact + t_hold + t_rel) and
    ``after_release``. An empty window reports ``None`` values, never zero.
    """
    t = np.asarray(times, dtype=float)
    directions, _eyes, head_r, head_t = schedule_directions(plan, kin, q, t)
    err = _angle_deg(head_r @ plan.gaze_axis_head, directions)
    t_hold_end = plan.impact_time_s + plan.t_hold_s
    t_rel_end = t_hold_end + plan.t_rel_s
    masks = {
        "address_to_impact": t <= plan.impact_time_s,
        "hold": (t > plan.impact_time_s) & (t <= t_hold_end),
        "release": (t > t_hold_end) & (t <= t_rel_end),
        "after_release": t > t_rel_end,
    }
    out: dict[str, Any] = {w: _window_stats(err, masks[w]) for w in _WINDOWS}
    out["address_to_impact_metrics"] = gaze.head_stability_metrics(
        head_r, head_t, plan.ball_m, 0, plan.impact_index, axis=plan.gaze_axis_head
    ).as_dict()
    return out


def apply_fd_neck(
    mode: str, lane: Any, kin: Any, q_ref: Array, q_track: Array
) -> tuple[Array, NeckSolve | None]:
    """Neck reference for the replay: ``ik`` keeps it, ``gaze`` follows the schedule.

    For ``gaze`` the neck columns of ``q_track`` are solved against the
    schedule, low-passed at the tracking cutoff and clipped back into the neck
    range. The returned :class:`NeckSolve` residuals are measured before that
    smoothing.
    """
    if mode not in NECK_REFERENCES:
        raise ValueError(
            f"unknown neck reference {mode!r}; use one of {NECK_REFERENCES}"
        )
    if mode == "ik":
        return q_track, None
    plan = _plan(lane, kin, q_ref)
    solved = q_track
    # The eye point turns with the head, so the scheduled direction depends on
    # the neck solution: re-evaluate it once from the first solve.
    for _ in range(EYE_COUPLING_PASSES):
        directions, *_ = schedule_directions(plan, kin, solved, lane.times)
        solve = solve_neck_schedule(kin, q_track, directions, plan.gaze_axis_head)
        solved = solve.q
    cols = _neck_columns(kin)
    lo, hi = _neck_bounds(None)
    smooth = smooth_reference(solve.q[:, cols], lane.rate_hz, TRACKING_CUTOFF_HZ)
    out = solve.q.copy()
    out[:, cols] = np.clip(smooth, lo, hi)
    return out, solve


def fd_head_gaze_report(
    lane: Any,
    kin: Any,
    q_ref: Array,
    q_track: Array,
    sim_q: Array,
    neck_solve: NeckSolve | None,
    mode: str,
) -> dict[str, Any]:
    """Receipt block: schedule tracking of the tracked reference and the replay."""
    label = "gaze_schedule" if mode == "gaze" else "ik"
    try:
        plan = _plan(lane, kin, q_ref)
        return {
            "available": True,
            "neck_reference": label,
            "neck_solve": None if neck_solve is None else neck_solve.as_dict(),
            "plan": plan.as_dict(),
            "tracked_reference": schedule_tracking(plan, kin, q_track, lane.times),
            "replay": schedule_tracking(plan, kin, sim_q, lane.times),
            "note": (
                "replay is computed-torque tracking of the tracked reference"
                + (
                    "; the gaze-driven neck is a modelled behaviour and head "
                    "markers do not drive the replay neck"
                    if mode == "gaze"
                    else "; the neck follows the marker-driven IK reference"
                )
            ),
        }
    except ValueError as exc:
        return {"available": False, "neck_reference": label, "reason": str(exc)}
