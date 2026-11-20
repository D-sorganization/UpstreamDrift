"""Measures of the trail hand's grip at address (epic #15150).

The owner judges the drawn trail hand against a reference grip by four things:
how far its wrist is arched, how square its fingers wrap to the shaft, whether
the middle and ring knuckles lie on the shaft's line behind it, and how high
the heel of its palm sits on the lead thumb. :func:`trail_grip_measures` reads
all four from a posed pair, in the pair's grip frame: ``z`` along the shaft
from butt to head, ``+x`` the top of the grip and ``+y`` the trail side, away
from the target. Appearance only; nothing here reads a model's joint angles.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from src.shared.python.model_appearance.hand_backends.joint_angles import FINGERS
from src.shared.python.model_appearance.hand_backends.skinned_landmarks import (
    KNUCKLE_SEGMENT,
    WRIST_BONE,
    hand_landmarks,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualResult

#: The bone whose joint is the base of the thumb, in the heel of the palm.
THUMB_BASE_BONE = "thumb_meta"
_FOREARM_BONE = "forearm"
#: Bones whose skin is the palm, the heel included.
_PALM_BONES = (WRIST_BONE, *(f"meta_{finger}" for finger in FINGERS))


@dataclass(frozen=True)
class TrailGripMeasures:
    """The trail hand against the reference grip (degrees and metres).

    ``wrist_extension_deg``: the forearm tipped toward the back of the hand.
    ``wrap_tilt_deg``: per finger, how far the plane it curls in is off square
    to the shaft. ``knuckle_line_to_shaft_deg``: the line from the forefinger's
    knuckle to the little finger's, against the shaft. ``knuckle_lateral_m``
    and ``knuckle_behind_m``: per finger, the knuckle's distance from the plane
    through the shaft and the target line (toward the top of the grip
    positive) and its distance behind the shaft, away from the target.
    ``heel_to_lead_thumb_base_m``: least distance from the trail palm's skin to
    the lead thumb's base joint. ``wrist_below_lead_thumb_base_m``: the trail
    wrist joint's station along the shaft past that joint, toward the head.
    """

    wrist_extension_deg: float
    wrap_tilt_deg: Mapping[str, float]
    knuckle_line_to_shaft_deg: float
    knuckle_lateral_m: Mapping[str, float]
    knuckle_behind_m: Mapping[str, float]
    heel_to_lead_thumb_base_m: float
    wrist_below_lead_thumb_base_m: float


def _to_shaft_deg(direction: np.ndarray) -> float:
    """Angle between a line and the shaft, 0 to 90 degrees."""
    along = abs(float(direction[2])) / float(np.linalg.norm(direction))
    return math.degrees(math.acos(min(1.0, along)))


def _wrist_extension_deg(hand: HandVisualResult) -> float:
    """Extension of the posed wrist: the forearm's turn on the hand is ulnar
    deviation about the palm's normal after extension about the wrist's width."""
    asset, transforms = hand.rest_asset, hand.bone_transforms
    if asset is None or transforms is None:
        raise ValueError("the hand carries no skinning data")
    forearm, wrist = (
        transforms[asset.bone_index(bone)][:3, :3]
        for bone in (_FOREARM_BONE, WRIST_BONE)
    )
    turn = wrist.T @ forearm
    return math.degrees(math.atan2(float(turn[2, 1]), float(turn[2, 2])))


def trail_grip_measures(
    lead: HandVisualResult, trail: HandVisualResult
) -> TrailGripMeasures:
    """Measure the trail hand of a posed pair against the reference grip.

    Preconditions: both are :class:`HandVisualResult` of a skinned pair in one
    grip frame, the lead hand first (``TypeError`` / ``ValueError`` otherwise).
    Postconditions: angles to the shaft lie in 0 to 90 degrees; distances are
    finite; nothing is changed.
    """
    for hand in (lead, trail):
        if not isinstance(hand, HandVisualResult):
            raise TypeError("trail_grip_measures needs two HandVisualResult")
    if (lead.side, trail.side) != ("lead", "trail"):
        raise ValueError("give the lead hand, then the trail hand")
    marks, asset = hand_landmarks(trail), trail.rest_asset
    if asset is None:
        raise ValueError("the trail hand carries no skinning data")
    knuckles = {f: marks.joint(f"{f}_{KNUCKLE_SEGMENT}").origin_m for f in FINGERS}
    tilts = {}
    for finger in FINGERS:
        hinge = marks.joint(f"{finger}_mid").hinge_axis
        if hinge is None:
            raise ValueError(f"the {finger} finger has no hinge to measure")
        tilts[finger] = _to_shaft_deg(hinge)
    base = hand_landmarks(lead).joint(THUMB_BASE_BONE).origin_m
    palm = trail.mesh.vertices[asset.dominant_vertex_ids(_PALM_BONES)]
    wrist = marks.joint(WRIST_BONE).origin_m
    return TrailGripMeasures(
        wrist_extension_deg=_wrist_extension_deg(trail),
        wrap_tilt_deg=tilts,
        knuckle_line_to_shaft_deg=_to_shaft_deg(knuckles["little"] - knuckles["index"]),
        knuckle_lateral_m={f: float(p[0]) for f, p in knuckles.items()},
        knuckle_behind_m={f: float(p[1]) for f, p in knuckles.items()},
        heel_to_lead_thumb_base_m=float(np.linalg.norm(palm - base, axis=1).min()),
        wrist_below_lead_thumb_base_m=float(wrist[2] - base[2]),
    )


__all__ = ["THUMB_BASE_BONE", "TrailGripMeasures", "trail_grip_measures"]
