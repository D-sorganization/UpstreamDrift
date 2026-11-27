"""GB-3 build stages: palm wrap, thumbs, and trail-hand linking."""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from src.shared.python.grip_build.club_sdf import ClubSDF
from src.shared.python.grip_build.contact_protocol import ContactDistanceModel
from src.shared.python.grip_build.geometry import capsule_capsule_signed_distance
from src.shared.python.grip_build.hand_chain import HandChain
from src.shared.python.grip_build.receipt import BuildStageRecord, BuildStatus
from src.shared.python.grip_build.thumb import (
    ThumbJointAngles,
    thumb_capsules,
    thumb_ip_cap_rad,
)
from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_spec.types import GripSpec

_AWAY_AZIMUTH_RAD: Final[float] = math.radians(-90.0)
_PAD_PAIR_NAMES: Final[tuple[str, ...]] = (
    "lead_heel_pad_butt",
    "trail_palm_lead_thumb",
)


def away_target_azimuth_rad(desired_deg: float, *, handedness: str = "right") -> float:
    """Convert grip_pose-style thumb azimuth (0 = away, + toward toe) to x-y angle."""
    direction = grip_pose_thumb_direction_rad(math.radians(desired_deg), handedness)
    return math.atan2(float(direction[1]), float(direction[0]))


def grip_pose_thumb_direction_rad(angle_deg: float, handedness: str) -> np.ndarray:
    target = 1.0 if handedness == "right" else -1.0
    away = np.array([0.0, -target, 0.0], dtype=np.float64)
    toe = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    a = math.radians(angle_deg)
    return math.cos(a) * away - math.sin(a) * toe


def grip_pose_azimuth_from_direction(direction: np.ndarray, handedness: str) -> float:
    target = 1.0 if handedness == "right" else -1.0
    away = np.array([0.0, -target, 0.0], dtype=np.float64)
    toe = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d[2] = 0.0
    norm = float(np.linalg.norm(d))
    if norm < 1e-12:
        return 0.0
    d /= norm
    sin_a = -float(d @ toe)
    cos_a = float(d @ away)
    return math.degrees(math.atan2(sin_a, cos_a))


def palm_wrap_stage(
    chain: HandChain,
    club: ClubSDF,
    model: ContactDistanceModel,
    *,
    bone_tol: float,
    pad_tol: float,
) -> tuple[BuildStageRecord, BuildStatus]:
    lo, hi = 0.0, math.radians(55.0)
    best_angle = 0.0
    best_clearance = -math.inf
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        chain.palm_wrap_rad = mid
        heel = chain.palm_heel_point()
        clearance = model.club_clearance(heel, heel, chain.palm_heel_radius_m(), club)
        if clearance >= -pad_tol:
            best_angle = mid
            best_clearance = clearance
            hi = mid
        else:
            lo = mid
    chain.palm_wrap_rad = best_angle
    status = BuildStatus.OK if best_clearance >= -pad_tol else BuildStatus.GAP
    return (
        BuildStageRecord(
            stage_id="palm_wrap",
            finger="palm",
            joint="wrap",
            flexion_rad=best_angle,
            status=status,
            min_clearance_m=best_clearance,
            target_tolerance_m=pad_tol,
        ),
        status,
    )


def thumb_press_stage(
    chain: HandChain,
    club: ClubSDF,
    model: ContactDistanceModel,
    *,
    thumb_radial: np.ndarray,
    ip_cap_rad: float,
    bone_tol: float,
    pad_tol: float,
    stage_id: str,
) -> tuple[BuildStageRecord, BuildStatus, ThumbJointAngles]:
    angles = ThumbJointAngles()
    lo, hi = 0.0, math.radians(70.0)
    best = ThumbJointAngles()
    best_clearance = -math.inf
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        trial = ThumbJointAngles(cmc_rad=mid * 0.35, mcp_rad=mid * 0.5, ip_rad=0.0)
        caps = thumb_capsules(
            chain.mesh_params,
            0.0,
            trial,
            chain.grip_radius_m,
            radial_unit=thumb_radial,
            palm_wrap_rad=chain.palm_wrap_rad,
            axial_shift_m=chain.axial_shift_m,
        )
        clearance = min(model.club_clearance(p0, p1, r, club) for p0, p1, r in caps)
        if clearance >= -pad_tol:
            best = trial
            best_clearance = clearance
            lo = mid
        else:
            hi = mid
    best.ip_rad = min(best.mcp_rad * 0.15, ip_cap_rad)
    caps = thumb_capsules(
        chain.mesh_params,
        0.0,
        best,
        chain.grip_radius_m,
        radial_unit=thumb_radial,
        palm_wrap_rad=chain.palm_wrap_rad,
        axial_shift_m=chain.axial_shift_m,
    )
    best_clearance = min(model.club_clearance(p0, p1, r, club) for p0, p1, r in caps)
    chain.thumb = best
    chain.thumb_azimuth_rad = math.atan2(float(thumb_radial[1]), float(thumb_radial[0]))
    status = BuildStatus.OK if best_clearance >= -pad_tol else BuildStatus.GAP
    return (
        BuildStageRecord(
            stage_id=stage_id,
            finger="thumb",
            joint="press",
            flexion_rad=best.cmc_rad + best.mcp_rad,
            status=status,
            min_clearance_m=best_clearance,
            target_tolerance_m=pad_tol,
        ),
        status,
        best,
    )


def trail_link_clearances(
    spec: GripSpec,
    lead: HandChain,
    trail: HandChain,
    model: ContactDistanceModel,
    *,
    trail_club: ClubSDF,
) -> dict[str, float]:
    """Hand-hand clearances for overlap, interlock, or ten-finger topology pairs."""
    lead_caps = {
        finger: lead.finger_capsules(finger) for finger in ("index", "middle", "little")
    }
    trail_lit = trail.finger_capsules("little")
    p0, p1, r = trail_lit[-1]
    clearances: dict[str, float] = {}
    if spec.style is GripStyle.OVERLAP:
        idx_caps = lead_caps["index"]
        mid_caps = lead_caps["middle"]
        gaps = [
            model.finger_clearance(p0, p1, r, *other) for other in idx_caps + mid_caps
        ]
        clearances["overlap_little_finger_lead_index"] = min(gaps)
    elif spec.style is GripStyle.INTERLOCK:
        mid_caps = lead_caps["middle"]
        idx_caps = lead_caps["index"]
        gaps_mid = [model.finger_clearance(p0, p1, r, *other) for other in mid_caps]
        gaps_idx = [model.finger_clearance(p0, p1, r, *other) for other in idx_caps]
        clearances["interlock_little_finger_lead_middle"] = min(gaps_mid)
        clearances["interlock_lead_index_wrap"] = min(gaps_idx)
    else:
        clearances["ten_finger_little_finger_grip"] = model.club_clearance(
            p0, p1, r, trail_club
        )
    thumb_caps = thumb_capsules(
        lead.mesh_params,
        lead.thumb_azimuth_rad,
        lead.thumb,
        lead.grip_radius_m,
        palm_wrap_rad=lead.palm_wrap_rad,
    )
    pocket = trail.trail_palm_pocket_point()
    lt, rt, rr = thumb_caps[-1]
    clearances["trail_palm_lead_thumb"] = capsule_capsule_signed_distance(
        pocket, pocket, trail.palm_pocket_radius_m(), lt, rt, rr
    )
    return clearances


def active_pairs_from_clearances(
    clearances: dict[str, float],
    *,
    pad_tol: float,
    bone_tol: float,
) -> tuple[str, ...]:
    active: list[str] = []
    for name, gap in clearances.items():
        if name.startswith("_"):
            continue
        tol = bone_tol if "interlock" in name or "overlap" in name else pad_tol
        if gap >= -tol:
            active.append(name)
    for pad_name in _PAD_PAIR_NAMES:
        if pad_name in clearances and clearances[pad_name] >= -pad_tol:
            active.append(pad_name)
    return tuple(dict.fromkeys(active))
