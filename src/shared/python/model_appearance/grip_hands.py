"""Both curled-finger hands placed on the club's grip (OSV-2 slice 2b, #11728).

:mod:`grip_hand_mesh` bakes each hand in its own hand-grip frame. This module
places the two hands on the visual grip tube of a :class:`ClubAssembly`, in the
club-body frame every engine already uses for the shaft, grip and head
(origin at the sole point on the shaft axis, grip toward ``-y``, shaft axis
offset ``axis_offset_m`` along ``+z``). An engine that draws
:func:`assembly_meshes_with_hands` on the club body therefore shows both hands
wrapped on the grip, whatever its hand bodies look like.

The hands ride the club body, not the hand bodies: the golfer's hands are
welded to the grip (the spec closure), so the hand-to-grip transform is
constant and drawing them on the club keeps them exactly on the grip.
Visual only: nothing here touches a mass, an inertia or a constraint.

Hand-grip frame in the club-body frame
--------------------------------------
``z`` is the grip axis ``g`` from the butt toward the head (club ``+y``);
``x`` is :mod:`grip_pose` azimuth 0, ``-n`` with ``n`` the face normal
projected off the shaft (the rendered head, so the club's ``face_roll_deg``
squares the hands with the face); ``y = z x x`` is azimuth 90 deg, ``-t``
with ``t`` the heel-to-toe axis. The origin is on the grip axis
``GripPose.distance_below_butt_m(hand)`` below the butt.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.club_assembly import (
    GRIP_BUTT_RADIUS_M,
    GRIP_TIP_RADIUS_M,
    ClubAssembly,
    assembly_meshes,
    clubface_vector,
)
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hand_mesh import build_grip_hand_mesh
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)

Array = NDArray[np.float64]

#: Mesh label of each hand, as drawn next to ``shaft``, ``grip`` and ``head``.
HAND_PARTS: dict[Hand, str] = {Hand.LEAD: "lead_hand", Hand.TRAIL: "trail_hand"}
_GRIP_AXIS = np.array([0.0, 1.0, 0.0])  # butt toward head in the club frame


def _check_club(club: object) -> None:
    if not isinstance(club, ClubAssembly):
        raise TypeError(f"club must be a ClubAssembly, got {type(club).__name__}")


def _check_pose(pose: object) -> None:
    if not isinstance(pose, GripPose):
        raise TypeError(f"pose must be a GripPose, got {type(pose).__name__}")


def grip_radius_at(club: ClubAssembly, distance_below_butt_m: float) -> float:
    """Radius of the visual grip tube ``distance_below_butt_m`` below the butt.

    The tube tapers linearly from ``GRIP_BUTT_RADIUS_M`` at the butt to
    ``GRIP_TIP_RADIUS_M`` at the grip's lower end. Preconditions: ``club`` a
    :class:`ClubAssembly`; the distance finite and on the grip
    (``[0, grip_length_m]``), else ``ValueError``.
    """
    _check_club(club)
    d = float(distance_below_butt_m)
    if not (math.isfinite(d) and 0.0 <= d <= club.grip_length_m):
        raise ValueError(
            f"distance {d} m below the butt is off the grip [0, {club.grip_length_m}] m"
        )
    f = d / club.grip_length_m
    return GRIP_BUTT_RADIUS_M + f * (GRIP_TIP_RADIUS_M - GRIP_BUTT_RADIUS_M)


def grip_hand_frame(
    club: ClubAssembly, hand: Hand, pose: GripPose = DEFAULT_GRIP_POSE
) -> Array:
    """4x4 transform from ``hand``'s hand-grip frame to the club-body frame.

    Postconditions: a proper rotation whose ``z`` column is the grip axis
    (butt toward head) and whose ``x`` column is minus the face normal
    projected off the shaft; the origin on the grip axis at the pose depth.
    """
    _check_club(club)
    if not isinstance(hand, Hand):
        raise TypeError(f"hand must be a Hand, got {hand!r}")
    _check_pose(pose)
    normal = np.asarray(clubface_vector(club), dtype=float)
    normal = normal - float(normal @ _GRIP_AXIS) * _GRIP_AXIS
    size = float(np.linalg.norm(normal))
    if size < 1e-6:
        raise ValueError("the club face normal is parallel to the shaft")
    x = -normal / size
    frame = np.eye(4)
    frame[:3, 0], frame[:3, 1], frame[:3, 2] = x, np.cross(_GRIP_AXIS, x), _GRIP_AXIS
    butt = np.array([0.0, -club.length_m, club.axis_offset_m])
    frame[:3, 3] = butt + pose.distance_below_butt_m(hand) * _GRIP_AXIS
    return frame


def _placed_hand(club: ClubAssembly, hand: Hand, pose: GripPose) -> Mesh:
    """One hand in the club frame, wrapped on the thickest grip radius it spans."""
    depth = pose.distance_below_butt_m(hand)
    nominal = build_grip_hand_mesh(
        hand, grip_radius_m=grip_radius_at(club, depth), pose=pose
    )
    nearest, _ = nominal.on_grip_extent_below_butt_m()
    radius = grip_radius_at(club, min(max(nearest, 0.0), club.grip_length_m))
    baked = build_grip_hand_mesh(hand, grip_radius_m=radius, pose=pose).mesh
    frame = grip_hand_frame(club, hand, pose)
    vertices = baked.vertices @ frame[:3, :3].T + frame[:3, 3]
    return Mesh(vertices, baked.faces)


def grip_hand_meshes(
    club: ClubAssembly, pose: GripPose = DEFAULT_GRIP_POSE
) -> dict[str, Mesh]:
    """``{"lead_hand", "trail_hand"}`` meshes wrapped on ``club``'s grip.

    Each hand is baked on the grip radius at its butt-most on-grip point (the
    tube tapers toward the head), so no pad enters the tube and every pad
    clears it by at most ``grip_hand_mesh.MAX_PAD_CLEARANCE_M`` plus the
    taper across the hand (0.3 mm). Postcondition: closed, outward-wound
    meshes in the club-body frame.
    """
    _check_club(club)
    _check_pose(pose)
    return {label: _placed_hand(club, hand, pose) for hand, label in HAND_PARTS.items()}


def assembly_meshes_with_hands(
    club: ClubAssembly, pose: GripPose = DEFAULT_GRIP_POSE
) -> dict[str, Mesh]:
    """Shaft, grip and head of :func:`assembly_meshes`, then both grip hands."""
    return {**assembly_meshes(club), **grip_hand_meshes(club, pose)}
