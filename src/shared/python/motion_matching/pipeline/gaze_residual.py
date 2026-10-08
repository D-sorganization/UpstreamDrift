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

Array = NDArray[np.float64]

HEAD_FRAME = "Head"
CLUB_FRAME = "Clubhead"
TORSO_FRAME = "Torso"
# Clubhead frame: +x is the face normal (club_models.ClubSpec.head_half_size_m
# lists x first as the face-normal extent), face centre at +x by that half size.
FACE_NORMAL_LOCAL: tuple[float, float, float] = (1.0, 0.0, 0.0)


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
            "gaze_axis_head": list(gaze.GAZE_AXIS_HEAD),
        }


def frame_poses(kin: Any, q: Array, frame: str) -> tuple[Array, Array]:
    """Rotations (frames, 3, 3) and translations (frames, 3) of one spec frame."""
    rots, trans = [], []
    for row in np.asarray(q, dtype=float):
        rot, pos = kin.body_poses(row, [frame])[frame]
        rots.append(rot)
        trans.append(pos)
    return np.asarray(rots), np.asarray(trans)


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
    normal = club_r[0] @ np.asarray(FACE_NORMAL_LOCAL)
    centre = club_t[0] + normal * face_offset_m
    ball = ball_position_at_address(centre, normal, ground_height_m=ground_height_m)
    horizontal = np.array([normal[0], normal[1], 0.0])
    if np.linalg.norm(horizontal) < 1e-6:
        raise ValueError("address face normal has no horizontal component")
    k = gaze.impact_index(t, club_t)
    return GazePlan(
        ball_m=ball,
        impact_index=k,
        impact_time_s=float(t[k]),
        target_dir=horizontal / np.linalg.norm(horizontal),
        t_hold_s=t_hold_s,
        t_rel_s=t_rel_s,
        face_normal=normal,
        face_centre_m=centre,
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
    return [{HEAD_FRAME: (gaze.GAZE_AXIS_HEAD, tuple(d), float(weight))} for d in directions]


def merge_axis_targets(
    base: Sequence[Mapping[str, Any] | None] | None,
    extra: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Union of two per-frame axis target lists (``extra`` wins on a clash)."""
    if base is not None and len(base) != len(extra):
        raise ValueError("axis target lists need one entry per frame")
    return [
        {**(base[k] or {}), **extra[k]} if base is not None else dict(extra[k])
        for k in range(len(extra))
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
        head_r, head_t, plan.ball_m, 0, plan.impact_index
    )
    torso_r, _ = frame_poses(kin, q, TORSO_FRAME)
    clamped = 0
    worst = 0.0
    for k in range(len(directions)):
        res = gaze.neck_ik(torso_r[k], directions[k])
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
        "neck_ik_schedule": {
            "frames": int(len(directions)),
            "frames_with_clamping": clamped,
            "max_residual_deg": float(worst),
        },
        "neck_range_deg": {
            name: [float(neck[:, i].min()), float(neck[:, i].max())]
            for i, name in enumerate(gaze.NECK_COORDINATES)
        },
    }


def club_face_offset_m(document: Mapping[str, Any]) -> float:
    """Face-normal half extent of the document's club head (face centre offset)."""
    from src.shared.python.motion_matching.club_models import CLUBS

    name = str(document.get("club", {}).get("name", ""))
    return float(CLUBS[name].head_half_size_m[0]) if name in CLUBS else 0.0


def head_gaze_receipt(lane: Any, kin: Any, q: Array) -> dict[str, Any]:
    """Receipt block for a finished lane (plans from ``q`` when no gaze pass ran)."""
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
