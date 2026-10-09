"""Soft gaze residual for the matching pipeline (OSV-3, #11729).

Builds, from a marker-faithful trajectory, a per-frame axis target that points
the head forward axis along the gaze schedule (ball until impact + hold, then a
minimum-jerk release to the target line), plus the receipt block with the head
stability metrics. The residual is soft: head markers still count and the
default weight 0 leaves the marker-faithful result untouched. The gaze-
regularised head is never measured data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.ball import (
    BALL_RADIUS_M,
    ball_position_at_address,
)
from src.shared.python.motion_matching import gaze
from src.shared.python.motion_matching.club_face_target import (
    merge_axis_targets as merge_axis_targets,  # one shared per-frame union
)

Array = NDArray[np.float64]

HEAD_FRAME = "Head"
CLUB_FRAME = "Clubhead"
TORSO_FRAME = "Torso"
GRIP_FRAME = "Grip"
# Neck IK is a per-frame bounded solve; the receipt summary samples every Nth.
NECK_IK_STRIDE = 6


@dataclass(frozen=True)
class GazePlan:
    """Geometry the schedule needs: ball, impact and target-line direction."""

    ball_m: Array
    impact_index: int
    impact_time_s: float
    target_dir: Array
    t_hold_s: float
    t_rel_s: float
    face_normal: Array
    face_centre_m: Array
    gaze_axis_head: Array

    def as_dict(self) -> dict[str, Any]:
        return {
            "ball_at_address_m": [float(v) for v in self.ball_m],
            "ball_radius_m": BALL_RADIUS_M,
            "impact_index": int(self.impact_index),
            "impact_time_s": float(self.impact_time_s),
            "impact_rule": "lowest clubhead point within 0.05 s of peak speed",
            "target_line_direction": [float(v) for v in self.target_dir],
            "t_hold_s": self.t_hold_s,
            "t_release_s": self.t_rel_s,
            "eye_offset_head_m": list(gaze.EYE_OFFSET_HEAD_M),
            "gaze_axis_head": [float(v) for v in self.gaze_axis_head],
            "gaze_axis_rule": "head-frame direction of the address line of sight to the ball",
        }


def frame_poses(kin: Any, q: Array, frame: str) -> tuple[Array, Array]:
    """Rotations (frames, 3, 3) and translations (frames, 3) of one spec frame."""
    rots, trans = [], []
    for row in np.asarray(q, dtype=float):
        rot, pos = kin.body_poses(row, [frame])[frame]
        rots.append(rot)
        trans.append(pos)
    return np.asarray(rots), np.asarray(trans)


def face_normal_local(
    club_r: Array, club_t: Array, grip_t: Array, times: Array, k: int
) -> Array:
    """Face normal in the clubhead frame from the impact path (square face).

    The Simscape clubhead frame carries its roll about the shaft arbitrarily,
    so the normal is the impact-frame clubhead velocity with its shaft
    component removed, expressed in the clubhead frame. Limitation: it assumes
    the face is square to the path at impact (recorded in the reference).
    """
    if not 0 < k < len(times) - 1:
        raise ValueError("impact frame must have neighbours for a velocity")
    shaft = club_r.T @ (grip_t - club_t[k])
    shaft = shaft / np.linalg.norm(shaft)
    vel = club_r.T @ ((club_t[k + 1] - club_t[k - 1]) / (times[k + 1] - times[k - 1]))
    perp = vel - (vel @ shaft) * shaft
    if np.linalg.norm(perp) < 1e-9:
        raise ValueError("clubhead has no path component normal to the shaft")
    return perp / np.linalg.norm(perp)


def plan_gaze(
    kin: Any,
    q: Array,
    times: Sequence[float] | Array,
    *,
    ground_height_m: float,
    face_offset_m: float,
    t_hold_s: float = gaze.DEFAULT_T_HOLD_S,
    t_rel_s: float = gaze.DEFAULT_T_RELEASE_S,
) -> GazePlan:
    """Ball, impact frame and target line from a trajectory ``q``.

    The ball is :func:`ball_position_at_address` for the address (frame 0)
    face; the target line is the horizontal face normal at address (square
    face). Impact comes from the clubhead trajectory, never a fixed time.
    """
    t = np.asarray(times, dtype=float)
    club_r, club_t = frame_poses(kin, q, CLUB_FRAME)
    k = gaze.impact_index(t, club_t)
    _, grip_t = frame_poses(kin, q, GRIP_FRAME)
    local_normal = face_normal_local(club_r[k], club_t, grip_t[k], t, k)
    normal = club_r[0] @ local_normal
    centre = club_t[0] + normal * face_offset_m
    ball = ball_position_at_address(centre, normal, ground_height_m=ground_height_m)
    head_r, head_t = frame_poses(kin, q[:1], HEAD_FRAME)
    sight = ball - gaze.eye_point(head_r[0], head_t[0])
    axis = head_r[0].T @ sight
    axis = axis / np.linalg.norm(axis)
    horizontal = np.array([normal[0], normal[1], 0.0])
    if np.linalg.norm(horizontal) < 1e-6:
        raise ValueError("address face normal has no horizontal component")
    return GazePlan(
        ball_m=ball,
        impact_index=k,
        impact_time_s=float(t[k]),
        target_dir=horizontal / np.linalg.norm(horizontal),
        t_hold_s=t_hold_s,
        t_rel_s=t_rel_s,
        face_normal=normal,
        face_centre_m=centre,
        gaze_axis_head=axis,
    )


def schedule_directions(
    plan: GazePlan, kin: Any, q: Array, times: Sequence[float] | Array
) -> tuple[Array, Array, Array, Array]:
    """(directions, eyes, head_R, head_t) of the schedule evaluated along ``q``."""
    head_r, head_t = frame_poses(kin, q, HEAD_FRAME)
    eyes = gaze.eye_point(head_r, head_t)
    directions = gaze.gaze_target_direction(
        np.asarray(times, dtype=float),
        eyes,
        plan.ball_m,
        plan.target_dir,
        t_impact=plan.impact_time_s,
        t_hold=plan.t_hold_s,
        t_rel=plan.t_rel_s,
    )
    return directions, eyes, head_r, head_t


def gaze_axis_targets(
    plan: GazePlan,
    kin: Any,
    q: Array,
    times: Sequence[float] | Array,
    weight: float,
) -> list[dict[str, tuple[Sequence[float], Sequence[float], float]]]:
    """Per-frame head axis targets ``{Head: (forward axis, direction, weight)}``."""
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("gaze weight must be finite and nonnegative")
    directions, *_ = schedule_directions(plan, kin, q, times)
    return [
        {HEAD_FRAME: (tuple(plan.gaze_axis_head), tuple(d), float(weight))}
        for d in directions
    ]


def gaze_report(
    plan: GazePlan,
    kin: Any,
    q: Array,
    times: Sequence[float] | Array,
    weight: float,
    coordinate_order: Sequence[str],
) -> dict[str, Any]:
    """Receipt block: weight, definitions, address-to-impact metrics, neck IK."""
    directions, _eyes, head_r, head_t = schedule_directions(plan, kin, q, times)
    metrics = gaze.head_stability_metrics(
        head_r, head_t, plan.ball_m, 0, plan.impact_index, axis=plan.gaze_axis_head
    )
    nominal = gaze.head_stability_metrics(
        head_r, head_t, plan.ball_m, 0, plan.impact_index
    )
    torso_r, _ = frame_poses(kin, q, TORSO_FRAME)
    clamped = 0
    worst = 0.0
    sampled = range(0, len(directions), NECK_IK_STRIDE)
    for k in sampled:
        res = gaze.neck_ik(torso_r[k], directions[k], axis=plan.gaze_axis_head)
        clamped += int(res.clamped.any())
        worst = max(worst, res.residual_deg)
    neck_cols = [coordinate_order.index(n) for n in gaze.NECK_COORDINATES]
    neck = np.degrees(np.asarray(q)[:, neck_cols])
    return {
        "gaze_weight": float(weight),
        "regularised": bool(weight > 0),
        "note": (
            "gaze-regularised head is a soft prior, not measured"
            if weight > 0
            else "marker-faithful head"
        ),
        "plan": plan.as_dict(),
        "address_to_impact": metrics.as_dict(),
        "address_to_impact_nominal_axis": {
            "theta_gaze_max_deg": nominal.theta_gaze_max_deg,
            "theta_gaze_rms_deg": nominal.theta_gaze_rms_deg,
            "note": "head +x axis as the gaze axis (not calibrated at address)",
        },
        "neck_ik_schedule": {
            "frames": len(sampled),
            "stride": NECK_IK_STRIDE,
            "frames_with_clamping": clamped,
            "max_residual_deg": float(worst),
        },
        "neck_range_deg": {
            name: [float(neck[:, i].min()), float(neck[:, i].max())]
            for i, name in enumerate(gaze.NECK_COORDINATES)
        },
    }


def head_gaze_receipt(lane: Any, kin: Any, q: Array) -> dict[str, Any]:
    """Receipt block for a finished lane (plans from ``q`` when no gaze pass ran).

    Unavailable is never zero: a model without the head, clubhead or grip frame
    (native Simscape geometry) returns ``{"available": False, "reason": ...}``.
    """
    try:
        plan = lane.gaze_plan
        if plan is None:
            plan = plan_gaze(
                kin,
                q,
                lane.times,
                ground_height_m=lane.ground.height_m,
                face_offset_m=lane.gaze_face_offset_m,
            )
        return gaze_report(
            plan, kin, q, lane.times, lane.gaze_weight, tuple(kin.coordinate_order)
        )
    except ValueError as exc:
        return {
            "available": False,
            "gaze_weight": float(lane.gaze_weight),
            "reason": str(exc),
        }
