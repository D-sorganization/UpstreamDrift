"""Shared hand-visual types for epic #15150.

Sibling lanes (procedural skin, shadow, MyoHand, skinned human) grew the same
three-argument ``build_hand_visual(grip_spec, side, joint_angles)`` shape.
This module is the SDF lane's copy of that contract so a later merge can keep
one definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.shared.python.grip_spec.types import GripSpec

HandRole = Literal["lead", "trail"]
HAND_ROLES: tuple[HandRole, ...] = ("lead", "trail")


@dataclass(frozen=True)
class HandJointAngles:
    """Curl inputs shared by every visual backend.

    ``palm_finger_depth`` and ``thumb_straightness`` are the GripSpec fields
    in ``[0, 1]``. ``club_cross_angle_deg`` is carried for the contract; the
    SDF poser does not yet shear the palm by it (see the SDF handoff).
    """

    palm_finger_depth: float
    thumb_straightness: float
    club_cross_angle_deg: float

    @classmethod
    def from_grip_spec(cls, spec: GripSpec, role: HandRole) -> HandJointAngles:
        """Read the per-hand authoring fields for ``role``."""
        if role not in HAND_ROLES:
            raise ValueError(f"role must be one of {HAND_ROLES}, got {role!r}")
        hand = spec.lead if role == "lead" else spec.trail
        return cls(
            palm_finger_depth=float(hand.palm_finger_depth),
            thumb_straightness=float(hand.thumb_straightness),
            club_cross_angle_deg=float(hand.club_cross_angle_deg),
        )


__all__ = ["HAND_ROLES", "HandJointAngles", "HandRole"]
