"""Hand anatomical frame from the physical club body and physical grip.

The hand frame is ``R_club @ N @ M(alpha, psi, gamma)``: the world rotation of
the club body, the engine's native hand-to-club rotation ``N`` and the declared
physical grip offset (:func:`grip_contract.follower_rotation`). Visual grip
parameters (:class:`GripPose` style, thumb offset, lifeline) and appearance
never enter: only :class:`HandAngles` (physical) are accepted.

Convention note: the direction of ``N`` follows ``child_to_follower`` of the
wrist joint as documented in ``grip_contract``; an engine adapter that supplies
``N`` is responsible for that direction.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.biomechanics.joint_conventions import AnatomicalFrame
from src.shared.python.model_appearance.grip_pose import GripPose
from src.shared.python.motion_matching.grip_contract import (
    GripSpec,
    HandAngles,
    follower_rotation,
)


def hand_frame_from_club(
    club_rotation_world: object,
    club_origin_world: object,
    native_rotation: object,
    grip: HandAngles,
    calibration_id: str,
) -> AnatomicalFrame:
    """Hand :class:`AnatomicalFrame` for a club pose and one hand's physical grip.

    Raises:
        TypeError: if ``grip`` is a visual ``GripPose``/``GripSpec`` or anything
            other than :class:`HandAngles` (select a hand with
            :func:`physical_hand`).
        ValueError: for a malformed native rotation or club pose.
    """
    if isinstance(grip, GripPose):
        raise TypeError(
            "hand frames come from the physical grip (HandAngles), never a "
            "visual GripPose"
        )
    if not isinstance(grip, HandAngles):
        raise TypeError(
            f"grip must be HandAngles (physical), got {type(grip).__name__}"
        )
    offset = follower_rotation(np.asarray(native_rotation, float), grip)
    return AnatomicalFrame(
        np.asarray(club_rotation_world, float) @ offset,
        np.asarray(club_origin_world, float),
        calibration_id,
    )


def physical_hand(spec: GripSpec, role: str) -> HandAngles:
    """The physical ``HandAngles`` of the ``"lead"`` or ``"trail"`` hand."""
    if role not in ("lead", "trail"):
        raise ValueError(f"role must be 'lead' or 'trail', got {role!r}")
    return spec.lead if role == "lead" else spec.trail
