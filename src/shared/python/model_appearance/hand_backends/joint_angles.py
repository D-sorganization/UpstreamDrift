"""Joint angles shared by every hand visual backend."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

from src.shared.python.grip_spec.types import GripSpec

HandSide = Literal["lead", "trail"]
_FINGER_NAMES = ("index", "middle", "ring", "little")
_FINGER_KEYS = _FINGER_NAMES + ("thumb",)


@dataclass(frozen=True)
class FingerFlexion:
    """MCP / PIP / DIP flexion in radians (positive = curl toward palm)."""

    mcp_rad: float = 0.0
    pip_rad: float = 0.0
    dip_rad: float = 0.0


@dataclass(frozen=True)
class HandJointAngles:
    """Full hand skeleton pose for visual backends."""

    fingers: dict[str, FingerFlexion] = field(default_factory=dict)
    thumb_mcp_rad: float = 0.0
    thumb_ip_rad: float = 0.0
    palm_arch_rad: float = 0.0
    metacarpal_spread_rad: float = 0.0

    def __post_init__(self) -> None:
        for finger in _FINGER_NAMES:
            if finger not in self.fingers:
                object.__setattr__(
                    self, "fingers", {**self.fingers, finger: FingerFlexion()}
                )

    @classmethod
    def from_legacy_curl(
        cls,
        *,
        finger_curl_scale: float,
        palm_arch_scale: float,
        thumb_straightness: float = 0.65,
    ) -> HandJointAngles:
        """Build a curled pose from normalised 0–1 scales (settings-driven tests)."""
        mcp = math.radians(55.0 * finger_curl_scale)
        pip = math.radians(70.0 * finger_curl_scale)
        dip = math.radians(45.0 * finger_curl_scale)
        fingers = {
            name: FingerFlexion(mcp_rad=mcp, pip_rad=pip, dip_rad=dip)
            for name in _FINGER_NAMES
        }
        arch = math.radians(22.0 * palm_arch_scale)
        spread = math.radians(8.0 * finger_curl_scale)
        thumb_mcp = math.radians(35.0 * (1.0 - thumb_straightness))
        thumb_ip = math.radians(20.0 * (1.0 - thumb_straightness))
        return cls(
            fingers=fingers,
            thumb_mcp_rad=thumb_mcp,
            thumb_ip_rad=thumb_ip,
            palm_arch_rad=arch,
            metacarpal_spread_rad=spread,
        )


def _hand_grip_spec(spec: GripSpec, side: HandSide):
    return spec.lead if side == "lead" else spec.trail


def joint_angles_from_grip_spec(spec: GripSpec, side: HandSide) -> HandJointAngles:
    """Angles that approximate the legacy capsule baker for this ``GripSpec``."""
    hand = _hand_grip_spec(spec, side)
    depth = float(hand.palm_finger_depth)
    curl_scale = 0.45 + 0.45 * depth
    arch_scale = 0.35 + 0.5 * depth
    angles = HandJointAngles.from_legacy_curl(
        finger_curl_scale=curl_scale,
        palm_arch_scale=arch_scale,
        thumb_straightness=float(hand.thumb_straightness),
    )
    if side == "trail":
        # Trail fingers curl the opposite azimuth direction; mirror flexion pattern.
        mirrored = {}
        for name, flex in angles.fingers.items():
            mirrored[name] = replace(flex, mcp_rad=flex.mcp_rad * 1.02)
        return replace(
            angles,
            fingers=mirrored,
            metacarpal_spread_rad=-angles.metacarpal_spread_rad,
        )
    return angles


def mirror_hand_joint_angles(angles: HandJointAngles) -> HandJointAngles:
    """Mirror across the hand-grip frame ``y=0`` plane (left/right on club)."""
    fingers = {
        name: replace(
            flex,
            mcp_rad=flex.mcp_rad,
            pip_rad=flex.pip_rad,
            dip_rad=flex.dip_rad,
        )
        for name, flex in angles.fingers.items()
    }
    return replace(
        angles,
        fingers=fingers,
        metacarpal_spread_rad=-angles.metacarpal_spread_rad,
    )


legacy_neutral_joint_angles = joint_angles_from_grip_spec


__all__ = [
    "FingerFlexion",
    "HandJointAngles",
    "HandSide",
    "joint_angles_from_grip_spec",
    "legacy_neutral_joint_angles",
    "mirror_hand_joint_angles",
]
