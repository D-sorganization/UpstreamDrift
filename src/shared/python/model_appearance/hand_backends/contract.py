"""Grip-spec articulation inputs for the SDF hand backend (#15155).

The shared backend types live in :mod:`hand_backends.types` (ADR-0057 A2);
this module keeps only the SDF poser's own inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends.types import (
    HAND_SIDES,
    HandSide,
)

HandRole = HandSide
HAND_ROLES: tuple[HandRole, ...] = HAND_SIDES


@dataclass(frozen=True)
class GripArticulation:
    """Curl inputs read from :class:`GripSpec` for one hand.

    ``palm_finger_depth`` and ``thumb_straightness`` are the GripSpec fields
    in ``[0, 1]``. ``club_cross_angle_deg`` is carried for the contract; the
    SDF poser does not yet shear the palm by it (see the SDF handoff).
    """

    palm_finger_depth: float
    thumb_straightness: float
    club_cross_angle_deg: float

    @classmethod
    def from_grip_spec(cls, spec: GripSpec, role: HandRole) -> GripArticulation:
        """Read the per-hand authoring fields for ``role``."""
        if role not in HAND_ROLES:
            raise ValueError(f"role must be one of {HAND_ROLES}, got {role!r}")
        hand = spec.lead if role == "lead" else spec.trail
        return cls(
            palm_finger_depth=float(hand.palm_finger_depth),
            thumb_straightness=float(hand.thumb_straightness),
            club_cross_angle_deg=float(hand.club_cross_angle_deg),
        )


__all__ = ["HAND_ROLES", "GripArticulation", "HandRole"]
