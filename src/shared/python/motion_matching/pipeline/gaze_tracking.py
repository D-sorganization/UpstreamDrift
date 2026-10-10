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

#: ``gaze-closed`` (OSV-3d) keeps the ``gaze`` reference as feedforward and
#: re-solves the neck target from the simulated torso during the replay.
NECK_REFERENCES = ("ik", "gaze", "gaze-closed")
_REPORT_LABELS = {
    "ik": "ik",
    "gaze": "gaze_schedule",
    "gaze-closed": "gaze_schedule_closed_loop",
}
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


@dataclass(frozen=True)
class _FrameSolve:
    """One bounded neck solve: point the head ``axis`` along a unit direction."""

    kin: Any
    cols: list[int]
    axis: Array
    bounds: tuple[Array, Array]
    prior_weight: float

    def _head_axis(self, row: Array) -> Array:
        rot = self.kin.body_poses(row, [HEAD_FRAME])[HEAD_FRAME][0]
        return rot @ self.axis

    def solve(
        self, q_row: Array, unit: Array, ref: Array, x0: Array
    ) -> tuple[Array, float, bool]:
        """(neck, residual in degrees, clamped) with the rest of ``q_row`` fixed."""
        lo, hi = self.bounds
        row = np.array(q_row, dtype=float)

        def resid(xk: Array) -> Array:
            row[self.cols] = xk
            return np.concatenate(
                [self._head_axis(row) - unit, self.prior_weight * (xk - ref)]
            )

        x = least_squares(resid, np.clip(x0, lo, hi), bounds=(lo, hi)).x
        x = np.clip(x, lo, hi)
        row[self.cols] = x
        residual = float(_angle_deg(self._head_axis(row)[None], unit[None])[0])
        clamped = bool(np.any((x - lo < _BOUND_TOL_RAD) | (hi - x < _BOUND_TOL_RAD)))
        return x, residual, clamped


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
    bounds = _neck_bounds(ranges_deg)
    axis = axis / np.linalg.norm(axis)
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    out = q.copy()
    clamped = 0
    residuals = np.zeros(len(q))
    x = np.clip(q[0, cols], *bounds)
    frame = _FrameSolve(kin, cols, axis, bounds, prior_weight)
    for k in range(len(q)):
        x, residuals[k], hit = frame.solve(q[k], unit[k], q[k, cols], x)
        out[k, cols] = x
        clamped += int(hit)
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


def _window_stats(err: Array, mask: NDArray[np.bool_]) -> dict[str, Any]:
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

    For ``gaze`` (and ``gaze-closed``, whose feedforward it is) the neck
    columns of ``q_track`` are solved against the schedule, low-passed at the
    tracking cutoff and clipped back into the neck range. The returned
    :class:`NeckSolve` residuals are measured before that smoothing.
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


class GazeNeckFeedback:
    """Closed-loop gaze neck: a kkt ``reference_hook`` (OSV-3d, #11729).

    Every ``update_period_s`` the neck target is re-solved on the *simulated*
    state, so the head axis points along the schedule direction seen from the
    simulated eye whatever the replay torso does; between updates the target
    is held. A weak prior toward the feedforward neck (``q_t``) fixes the roll
    about the gaze axis. Only the neck columns of ``q_t`` are replaced, and
    they stay inside ``UPPER_RANGES_DEG``. This is a modelled behaviour, not
    measured data.
    """

    def __init__(
        self,
        plan: GazePlan,
        kin: Any,
        times: Array,
        *,
        update_period_s: float = 1.0 / 240.0,
        prior_weight: float = 1e-2,
    ) -> None:
        if not np.isfinite(update_period_s) or update_period_s <= 0:
            raise ValueError(
                f"update_period_s must be finite and > 0, got {update_period_s}"
            )
        self.plan = plan
        self.kin = kin
        self.update_period_s = float(update_period_s)
        self._t0 = float(np.asarray(times, dtype=float)[0])
        self._cols = _neck_columns(kin)
        axis = np.asarray(plan.gaze_axis_head, dtype=float)
        self._frame = _FrameSolve(
            kin,
            self._cols,
            axis / np.linalg.norm(axis),
            _neck_bounds(None),
            prior_weight,
        )
        self._slot: int | None = None
        self._neck: Array | None = None
        self._residuals: list[float] = []
        self._clamped = 0

    def _direction(self, t: float, row: Array) -> Array:
        directions, *_ = schedule_directions(self.plan, self.kin, row[None], [t])
        return directions[0] / np.linalg.norm(directions[0])

    def _update(self, t: float, q: Array, q_t: Array) -> Array:
        row = q.copy()
        ref = q_t[self._cols]
        x = q[self._cols] if self._neck is None else self._neck
        for _ in range(EYE_COUPLING_PASSES):  # the eye turns with the head
            x, residual, clamped = self._frame.solve(
                row, self._direction(t, row), ref, x
            )
            row[self._cols] = x
        self._residuals.append(residual)
        self._clamped += int(clamped)
        return x

    def __call__(self, t: float, q: Array, q_t: Array) -> Array:
        q = np.asarray(q, dtype=float)
        q_t = np.asarray(q_t, dtype=float)
        if q.shape != q_t.shape or q.ndim != 1:
            raise ValueError(
                f"q and q_t must share a 1-D shape, got {q.shape} and {q_t.shape}"
            )
        if not (np.isfinite(q).all() and np.isfinite(q_t).all()):
            raise ValueError("q and q_t must be finite")
        slot = int(np.floor((float(t) - self._t0) / self.update_period_s + 1e-9))
        if self._neck is None or slot != self._slot:
            self._neck = self._update(float(t), q, q_t)
            self._slot = slot
        out = q_t.copy()
        out[self._cols] = self._neck
        return out

    def as_dict(self) -> dict[str, Any]:
        """Update count, clamped updates and residuals (``None`` before any)."""
        res = np.asarray(self._residuals)
        return {
            "updates": len(res),
            "clamped_updates": int(self._clamped),
            "max_residual_deg": float(res.max()) if res.size else None,
            "rms_residual_deg": (float(np.sqrt(np.mean(res**2))) if res.size else None),
            "update_period_s": self.update_period_s,
        }


def fd_neck_feedback(
    mode: str, lane: Any, kin: Any, q_ref: Array
) -> GazeNeckFeedback | None:
    """The replay's reference hook for ``gaze-closed``, else ``None``.

    The hook updates once per capture frame (``1 / lane.rate_hz``). Build a
    fresh one for every replay: it holds state.
    """
    if mode not in NECK_REFERENCES:
        raise ValueError(
            f"unknown neck reference {mode!r}; use one of {NECK_REFERENCES}"
        )
    if mode != "gaze-closed":
        return None
    return GazeNeckFeedback(
        _plan(lane, kin, q_ref),
        kin,
        lane.times,
        update_period_s=1.0 / float(lane.rate_hz),
    )


_NOTES = {
    "ik": "; the neck follows the marker-driven IK reference",
    "gaze": (
        "; the gaze-driven neck is a modelled behaviour and head markers do "
        "not drive the replay neck"
    ),
    "gaze-closed": (
        "; the gaze-driven neck is a modelled behaviour re-solved on the "
        "simulated torso during the replay, and head markers do not drive it"
    ),
}


def fd_head_gaze_report(
    lane: Any,
    kin: Any,
    q_ref: Array,
    q_track: Array,
    sim_q: Array,
    neck_solve: NeckSolve | None,
    mode: str,
    *,
    feedback: GazeNeckFeedback | None = None,
) -> dict[str, Any]:
    """Receipt block: schedule tracking of the tracked reference and the replay.

    ``feedback`` (``gaze-closed``) adds the closed-loop update statistics.
    """
    label = _REPORT_LABELS.get(mode, "ik")
    try:
        plan = _plan(lane, kin, q_ref)
        out = {
            "available": True,
            "neck_reference": label,
            "neck_solve": None if neck_solve is None else neck_solve.as_dict(),
            "plan": plan.as_dict(),
            "tracked_reference": schedule_tracking(plan, kin, q_track, lane.times),
            "replay": schedule_tracking(plan, kin, sim_q, lane.times),
            "note": "replay is computed-torque tracking of the tracked reference"
            + _NOTES.get(mode, _NOTES["ik"]),
        }
        if feedback is not None:
            out["feedback"] = feedback.as_dict()
        return out
    except ValueError as exc:
        return {"available": False, "neck_reference": label, "reason": str(exc)}
