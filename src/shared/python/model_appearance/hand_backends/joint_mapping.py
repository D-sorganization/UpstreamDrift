"""Map :class:`GripSpec` fields onto MyoHand joint angles (rad)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Literal

from src.shared.python.contracts import require
from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.hand_size import (
    DEFAULT_HAND_LENGTH_M,
    hand_mesh_scale,
    resolve_hand_breadth_m,
)
from src.shared.python.model_appearance.grip_hand_mesh import (
    DEFAULT_HAND_MESH_PARAMS,
    HandMeshParams,
)
from src.shared.python.model_appearance.grip_pose import Hand

HandSide = Literal["lead", "trail"]


def _hand_for_side(side: HandSide, spec: GripSpec) -> Hand:
    if side == "lead":
        return Hand.LEAD
    if side == "trail":
        return Hand.TRAIL
    raise ValueError(f"side must be 'lead' or 'trail', got {side!r}")


def _mesh_params(spec: GripSpec) -> HandMeshParams:
    length = spec.hand_size.length_m
    breadth = spec.hand_size.breadth_m
    scale = hand_mesh_scale(length)
    ref_breadth = resolve_hand_breadth_m(DEFAULT_HAND_LENGTH_M)
    _ = breadth / ref_breadth if ref_breadth > 0.0 else 1.0
    return replace(DEFAULT_HAND_MESH_PARAMS, scale=scale)


def grip_spec_to_myohand_joint_angles(
    spec: GripSpec,
    side: HandSide,
    *,
    grip_radius_m: float,
    params: HandMeshParams | None = None,
    joint_angle_overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Resolve the 23 canonical MyoHand joints for one hand.

    Preconditions:
        ``spec`` is a :class:`GripSpec`; ``grip_radius_m`` in [0.008, 0.03].
    Postconditions:
        Returns a dict keyed by canonical joint name (no ``rh_`` prefix) in radians.
    """
    from src.engines.physics_engines.myosuite.python.myohand_bilateral_adapter import (
        pose_to_myohand_joint_targets,
    )

    require(0.008 <= grip_radius_m <= 0.03, "grip_radius_m out of range")
    pose = spec.to_grip_pose()
    mesh_params = _mesh_params(spec) if params is None else params
    hand = _hand_for_side(side, spec)
    base = pose_to_myohand_joint_targets(
        pose,
        trigger_finger=spec.trigger_finger,
        grip_radius_m=grip_radius_m,
        params=mesh_params,
    )[hand]

    hand_spec = spec.lead if side == "lead" else spec.trail
    palm_depth = float(hand_spec.palm_finger_depth)
    flex = (palm_depth - 0.5) * 0.7
    base["flexion"] = _clamp_like_adapter("flexion", base["flexion"] + flex * 0.35)
    base["deviation"] = _clamp_like_adapter(
        "deviation", base["deviation"] + flex * 0.2 * (1.0 if side == "lead" else -0.5)
    )

    thumb_lo, thumb_hi = -0.55, 0.15
    straight = float(hand_spec.thumb_straightness)
    thumb_delta = thumb_lo + (thumb_hi - thumb_lo) * straight
    base["mp_flexion"] = _clamp_like_adapter("mp_flexion", thumb_delta)
    base["ip_flexion"] = _clamp_like_adapter("ip_flexion", thumb_delta * 0.85)

    index_station = float(hand_spec.index_contact_station)
    if index_station > 1e-6:
        base["mcp2_flexion"] = _clamp_like_adapter(
            "mcp2_flexion", base["mcp2_flexion"] - 0.25 * index_station
        )

    for joint, override in hand_spec.joint_coupling_overrides:
        base[joint] = _clamp_like_adapter(joint, float(override))

    if joint_angle_overrides:
        for name, val in joint_angle_overrides.items():
            key = name.removeprefix("rh_")
            base[key] = _clamp_like_adapter(key, float(val))

    return base


def _clamp_like_adapter(joint: str, value: float) -> float:
    import numpy as np

    from src.engines.physics_engines.myosuite.python.myohand_bilateral_adapter import (
        _JOINT_LIMITS_RAD,
    )

    lo, hi = _JOINT_LIMITS_RAD[joint]
    return float(np.clip(value, lo, hi))


def anatomical_mirror_for_side(spec: GripSpec, side: HandSide) -> bool:
    """Whether to sagittally mirror the anatomical right-hand chain."""
    if spec.handedness == "right":
        return side == "lead"
    return side == "trail"


__all__ = [
    "anatomical_mirror_for_side",
    "grip_spec_to_myohand_joint_angles",
]
