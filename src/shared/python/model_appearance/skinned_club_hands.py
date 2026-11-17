"""Skinned human hands on a club, in a frame squared to the clubface (#15251).

:mod:`grip_hands` turns the capsule hands from the wrist followers, which left
the top of the grip under the hands 46 degrees round the shaft from the top of
the grip the clubface defines. Here the posed pair of
:mod:`hand_backends.skinned_pair` is placed in a frame taken from the clubface
alone: ``+x`` is the top of the grip (at right angles to the face normal),
``+z`` runs from butt to head and the origin is on the grip axis, the lead
hand's distance below the butt. Visual only: no force or contact model reads it.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.club_assembly import ClubAssembly
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hands import (
    HAND_PARTS,
    face_square_axes,
)
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.golf_grip import load_golf_grip
from src.shared.python.model_appearance.hand_backends.qa import (
    load_hand_visual_quality,
    wrist_joint_m,
)
from src.shared.python.model_appearance.hand_backends.skinned_pair import (
    build_grip_hands,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualResult


Array = NDArray[np.float64]

_GRIP_AXIS = np.array([0.0, 1.0, 0.0])  # butt -> head in the club-body frame


def owner_grip_spec(roll_deg: float = 0.0) -> GripSpec:
    """The legacy grip with the owner's settings from ``hand_visual_quality``.

    The lead thumb sits ``owner_thumb_azimuth_deg.lead`` to the away-from-target
    side of the top of the grip and the trail thumb ``.trail`` to the target
    side; the club leaves the lead palm ``owner_lead_heel_pad_depth_m`` below
    the little-finger knuckle; the lead hand is ``owner_lead_below_cap_edge_m``
    below the edge of the butt cap. ``roll_deg`` turns both hands about the shaft.
    """
    base = GripSpec.legacy()
    quality = load_hand_visual_quality()
    thumbs = quality["owner_thumb_azimuth_deg"]
    lead_turn, trail_turn = (
        {"shaft_roll_deg": float(roll_deg), "thumb_azimuth_deg": float(thumbs[side])}
        for side in ("lead", "trail")
    )
    heel = float(quality["owner_lead_heel_pad_depth_m"])
    below = float(quality["owner_lead_below_cap_edge_m"])
    return replace(
        base,
        lead=replace(base.lead, pinky_side_palm_depth_m=heel, **lead_turn),
        trail=replace(base.trail, **trail_turn),
        lead_below_butt_m=below + load_golf_grip().cap_height_m,
    )


def skinned_hand_frame(club: ClubAssembly) -> Array:
    """4x4 pose of the skinned pair's grip frame in the club-body frame.

    Postcondition: a right-handed rotation whose ``z`` is the grip axis and
    whose ``x`` is at right angles to the clubface normal.
    """
    top, side = face_square_axes(club)
    frame = np.eye(4)
    frame[:3, 0], frame[:3, 1], frame[:3, 2] = top, side, _GRIP_AXIS
    butt = np.array([0.0, -club.length_m, club.axis_offset_m])
    frame[:3, 3] = butt + owner_grip_spec().lead_below_butt_m * _GRIP_AXIS
    return frame


@lru_cache(maxsize=1)
def _pair() -> dict[str, HandVisualResult]:
    """The posed pair in its own grip frame; solving it takes seconds."""
    radius = float(load_hand_visual_quality()["grip_radius_m"])
    return build_grip_hands(owner_grip_spec(), grip_radius_m=radius)


def skinned_hand_meshes(club: ClubAssembly) -> dict[str, Mesh]:
    """``{"lead_hand", "trail_hand"}`` skinned meshes in the club-body frame.

    The hands are posed on the standard golf grip of ``golf_grip.json``, not on
    the club's drawn grip tube, which is up to about a millimetre fatter.
    """
    frame, pair = skinned_hand_frame(club), _pair()
    return {
        label: Mesh(
            pair[hand.value].mesh.vertices @ frame[:3, :3].T + frame[:3, 3],
            pair[hand.value].mesh.faces,
        )
        for hand, label in HAND_PARTS.items()
    }


def wrist_joints_m(club: ClubAssembly) -> dict[str, Array]:
    """Wrist joint centre of each hand (``lead``, ``trail``) in the club-body frame.

    This is where a forearm attaches to the posed hand.
    """
    frame, pair = skinned_hand_frame(club), _pair()
    return {
        hand.value: frame[:3, :3] @ wrist_joint_m(pair[hand.value]) + frame[:3, 3]
        for hand in (Hand.LEAD, Hand.TRAIL)
    }
