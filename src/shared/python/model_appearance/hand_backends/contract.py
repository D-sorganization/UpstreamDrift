"""Grip-spec articulation inputs for the procedural skin backend (#15122).

The shared backend types live in :mod:`hand_backends.types` (ADR-0057 A2);
this module keeps only the procedural backend's own posing inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends.types import HandSide


@dataclass(frozen=True)
class GripArticulation:
    """Visual articulation inputs read from :class:`GripSpec` for one hand."""

    palm_finger_depth: float
    thumb_straightness: float
    club_cross_angle_deg: float

    @classmethod
    def from_grip_spec(cls, spec: GripSpec, role: HandSide) -> GripArticulation:
        hand = spec.lead if role == "lead" else spec.trail
        return cls(
            palm_finger_depth=float(hand.palm_finger_depth),
            thumb_straightness=float(hand.thumb_straightness),
            club_cross_angle_deg=float(hand.club_cross_angle_deg),
        )


__all__ = ["GripArticulation"]
