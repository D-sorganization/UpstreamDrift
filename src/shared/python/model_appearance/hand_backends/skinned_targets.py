"""Grip-frame targets of one skinned hand, as a right hand (epic #15150).

Split from ``skinned_human`` to keep that module within the size budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends.joint_angles import HandSide
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    SkinnedHumanSettings,
)


@dataclass(frozen=True)
class RightHandTargets:
    """Grip-frame targets for the right-hand build of one side."""

    palm_azimuth_rad: float
    thumb_azimuth_rad: float
    cross_angle_deg: float
    index_station: float
    pinky_palm_depth_m: float
    thumb_straightness: float
    mirrored: bool


def targets_for(
    spec: GripSpec, side: HandSide, settings: SkinnedHumanSettings
) -> RightHandTargets:
    """Express one hand's spec as targets for a right hand.

    Azimuth 0 is the top of the grip and increases away from the target. For a
    right-handed golfer the trail hand is the right hand; the lead hand is
    built as its mirror image (``y -> -y``), so its targets are negated here.
    """
    hand = spec.lead if side == "lead" else spec.trail
    v_azimuth = hand.v_azimuth_deg if side == "trail" else -hand.v_azimuth_deg
    is_right = (side == "trail") == (spec.handedness == "right")
    station = hand.index_contact_station or settings.index_station_at_spec_zero
    return RightHandTargets(
        palm_azimuth_rad=math.radians(
            v_azimuth
            + settings.palm_contact_azimuth_offset_deg
            + settings.palm_centre_wrap_deg
        ),
        thumb_azimuth_rad=math.radians(-hand.thumb_azimuth_deg),
        cross_angle_deg=hand.club_cross_angle_deg,
        index_station=station,
        pinky_palm_depth_m=hand.pinky_side_palm_depth_m,
        thumb_straightness=hand.thumb_straightness,
        mirrored=not is_right,
    )


def turned(
    targets: RightHandTargets, turn_rad: float, straightness: float | None = None
) -> RightHandTargets:
    """``targets`` with the thumb azimuth turned by ``turn_rad``."""
    return replace(
        targets,
        thumb_azimuth_rad=targets.thumb_azimuth_rad + turn_rad,
        thumb_straightness=(
            targets.thumb_straightness if straightness is None else straightness
        ),
    )


__all__ = ["RightHandTargets", "targets_for", "turned"]
