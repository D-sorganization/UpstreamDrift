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
``x`` is :mod:`grip_pose` azimuth 0, ``+t`` with ``t`` the heel-to-toe axis
projected off the shaft, which is the top of the grip toward the golfer's chin
(the shaft leans toward the golfer, so the top is the toe side, not ``-n``);
``y = z x x`` is azimuth 90 deg, ``-n`` with ``n`` the face normal projected
off the shaft (the rendered head, so the club's ``face_roll_deg`` squares the
hands with the face), which is the trail side (away from the target). HRG-A1
(#13162) replaced the earlier ``x = -n`` convention, which rolled both hands
about 90 deg under the grip. The origin is on the grip axis
``GripPose.distance_below_butt_m(hand)`` below the butt.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from dataclasses import replace
from typing import Any, Final

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
from src.shared.python.model_appearance.grip_hand_mesh import (
    DEFAULT_HAND_MESH_PARAMS,
    HandMeshParams,
    build_grip_hand_mesh,
)
from src.shared.python.hand_size import (
    DEFAULT_HAND_LENGTH_M,
    hand_mesh_scale,
)
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)

from src.shared.python.model_appearance.hand_spec import (
    ResolvedHandSpec,
    wrist_follower_dorsal_axis,
)

Array = NDArray[np.float64]

#: Mesh label of each hand, as drawn next to ``shaft``, ``grip`` and ``head``.
HAND_PARTS: dict[Hand, str] = {Hand.LEAD: "lead_hand", Hand.TRAIL: "trail_hand"}
_GRIP_AXIS = np.array([0.0, 1.0, 0.0])  # butt toward head in the club frame

#: Half the knuckle sweep from the V crease to MCP, placing the dorsum midpoint (deg).
#: Fixed geometric offset of the hand mesh's metacarpal block relative to the V crease.
HAND_MESH_DORSUM_FROM_V_DEG: Final[float] = (
    DEFAULT_HAND_MESH_PARAMS.mcp_from_v_deg / 2.0
)  # 55.0

#: Fixed offset (deg) from the wrist follower frame dorsal axis to the hand mesh dorsum.
#: The drawn dorsum aligns directly with the wrist follower dorsal axis (SHOW-A15 #14747).
WRIST_TO_DORSUM_OFFSET_DEG: Final[float] = 0.0


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


def _club_axes(club: ClubAssembly) -> tuple[Array, Array]:
    normal = np.asarray(clubface_vector(club), dtype=float)
    normal = normal - float(normal @ _GRIP_AXIS) * _GRIP_AXIS
    size = float(np.linalg.norm(normal))
    if size < 1e-6:
        raise ValueError("the club face normal is parallel to the shaft")
    x = np.cross(_GRIP_AXIS, normal / size)  # +t: the top of the grip (chin side)
    return x, np.cross(_GRIP_AXIS, x)


def _resolve_dorsal_axis(
    hand: Hand,
    pose: GripPose,
    wrist_follower: Array | Mapping[str, Any] | None,
    hands: ResolvedHandSpec | None,
) -> Array | None:
    if isinstance(wrist_follower, Mapping):
        return wrist_follower_dorsal_axis(wrist_follower, hand, pose)
    if wrist_follower is not None:
        arr = np.asarray(wrist_follower, dtype=float)
        dorsal = arr[:3, 0] if arr.ndim == 2 else arr[:3]
    elif hands is not None and hand in hands.wrist_dorsal_axes:
        dorsal = np.asarray(hands.wrist_dorsal_axes[hand], dtype=float)
    else:
        return None
    proj = dorsal - float(dorsal @ _GRIP_AXIS) * _GRIP_AXIS
    size = float(np.linalg.norm(proj))
    if size < 1e-6:
        raise ValueError("wrist follower dorsal axis is parallel to the shaft")
    return proj / size


def grip_hand_frame(
    club: ClubAssembly,
    hand: Hand,
    pose: GripPose = DEFAULT_GRIP_POSE,
    *,
    wrist_follower: Array | Mapping[str, Any] | None = None,
    hands: ResolvedHandSpec | None = None,
) -> Array:
    """4x4 transform from ``hand``'s hand-grip frame to the club-body frame.

    When ``wrist_follower`` or ``hands`` (with ``wrist_dorsal_axes``) is present,
    derives ``x`` and ``y`` from the wrist follower's dorsal axis projected off
    the grip axis ``z`` (#13996); otherwise falls back to ``_club_axes(club)``.
    """
    _check_club(club)
    if not isinstance(hand, Hand):
        raise TypeError(f"hand must be a Hand, got {hand!r}")
    _check_pose(pose)
    dorsal = _resolve_dorsal_axis(hand, pose, wrist_follower, hands)
    if dorsal is not None:
        x, y = dorsal, np.cross(_GRIP_AXIS, dorsal)
    else:
        x, y = _club_axes(club)
    frame = np.eye(4)
    frame[:3, 0], frame[:3, 1], frame[:3, 2] = x, y, _GRIP_AXIS
    butt = np.array([0.0, -club.length_m, club.axis_offset_m])
    frame[:3, 3] = butt + pose.distance_below_butt_m(hand) * _GRIP_AXIS
    return frame


def _mesh_params(hand_length_m: float | None) -> HandMeshParams:
    length = DEFAULT_HAND_LENGTH_M if hand_length_m is None else float(hand_length_m)
    return replace(DEFAULT_HAND_MESH_PARAMS, scale=hand_mesh_scale(length))


def hand_mesh_to_wrist_rotation(
    hand: Hand,
    pose: GripPose = DEFAULT_GRIP_POSE,
    params: HandMeshParams = DEFAULT_HAND_MESH_PARAMS,
) -> Array:
    """3x3 rotation about grip z aligning the hand mesh's dorsum with the wrist frame x-axis.

    The hand mesh generated by :func:`build_grip_hand_mesh` places its V crease at
    ``placement.v_azimuth_deg`` and its dorsum (metacarpal block) midpoint at
    ``placement.v_azimuth_deg + curl * HAND_MESH_DORSUM_FROM_V_DEG``. Rotating
    by ``-(placement.v_azimuth_deg + curl * HAND_MESH_DORSUM_FROM_V_DEG)`` cancels
    the visual ``grip_pose`` azimuth and aligns the drawn dorsum directly with the
    wrist follower frame's dorsal axis (+x), ensuring the physics wrist frame is the
    single source of truth (SHOW-A15 #14747).
    """
    curl = -1 if hand is Hand.LEAD else 1
    placement = pose.placement(hand)
    if params is None:
        params = _mesh_params(None)
    dorsum_offset = params.mcp_from_v_deg / 2.0
    mesh_dorsum_deg = placement.v_azimuth_deg + curl * dorsum_offset
    d_rot_deg = WRIST_TO_DORSUM_OFFSET_DEG - mesh_dorsum_deg
    rad = math.radians(d_rot_deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _placed_hand(
    club: ClubAssembly,
    hand: Hand,
    pose: GripPose,
    *,
    hand_length_m: float | None = None,
    hands: ResolvedHandSpec | None = None,
) -> Mesh:
    """One hand in the club frame, wrapped on the thickest grip radius it spans."""
    params = _mesh_params(hand_length_m)
    depth = pose.distance_below_butt_m(hand)
    nominal = build_grip_hand_mesh(
        hand, grip_radius_m=grip_radius_at(club, depth), pose=pose, params=params
    )
    nearest, _ = nominal.on_grip_extent_below_butt_m()
    radius = grip_radius_at(club, min(max(nearest, 0.0), club.grip_length_m))
    baked = build_grip_hand_mesh(
        hand, grip_radius_m=radius, pose=pose, params=params
    ).mesh
    frame = grip_hand_frame(club, hand, pose, hands=hands)
    dorsal = _resolve_dorsal_axis(hand, pose, None, hands)
    if dorsal is not None:
        rot = frame[:3, :3] @ hand_mesh_to_wrist_rotation(hand, pose, params)
    else:
        rot = frame[:3, :3]
    vertices = baked.vertices @ rot.T + frame[:3, 3]
    return Mesh(vertices, baked.faces)


def drawn_dorsum_direction(
    club: ClubAssembly,
    hand: Hand,
    pose: GripPose = DEFAULT_GRIP_POSE,
    *,
    hand_length_m: float | None = None,
    hands: ResolvedHandSpec | None = None,
) -> Array:
    """Club-frame unit direction of ``hand``'s drawn dorsum (metacarpal block) normal.

    Derives the normal from the placed dorsum component shell, projected off the
    shaft axis (SHOW-A15 #14747).
    """
    _check_club(club)
    _check_pose(pose)
    if not isinstance(hand, Hand):
        raise TypeError(f"hand must be a Hand, got {hand!r}")
    if hands is not None:
        pose, hand_length_m = hands.pose, hands.hand_length_m
    params = _mesh_params(hand_length_m)
    depth = pose.distance_below_butt_m(hand)
    nominal = build_grip_hand_mesh(
        hand, grip_radius_m=grip_radius_at(club, depth), pose=pose, params=params
    )
    nearest, _ = nominal.on_grip_extent_below_butt_m()
    radius = grip_radius_at(club, min(max(nearest, 0.0), club.grip_length_m))
    baked_grip = build_grip_hand_mesh(
        hand, grip_radius_m=radius, pose=pose, params=params
    )
    dorsum_parts = [c for c in baked_grip.components if c.finger == "dorsum"]
    if not dorsum_parts:
        raise ValueError(f"no dorsum component on {hand.name} hand mesh")
    frame = grip_hand_frame(club, hand, pose, hands=hands)
    dorsal = _resolve_dorsal_axis(hand, pose, None, hands)
    if dorsal is not None:
        rot = frame[:3, :3] @ hand_mesh_to_wrist_rotation(hand, pose, params)
    else:
        rot = frame[:3, :3]
    club_d_verts = dorsum_parts[0].mesh.vertices @ rot.T
    mean_dir = club_d_verts.mean(axis=0)
    proj = mean_dir - float(mean_dir @ _GRIP_AXIS) * _GRIP_AXIS
    size = float(np.linalg.norm(proj))
    if size < 1e-9:
        raise ValueError("dorsum direction is parallel to the shaft")
    return proj / size


def grip_hand_meshes(
    club: ClubAssembly,
    pose: GripPose = DEFAULT_GRIP_POSE,
    *,
    hand_length_m: float | None = None,
    hands: ResolvedHandSpec | None = None,
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
    return {
        label: _placed_hand(club, hand, pose, hand_length_m=hand_length_m, hands=hands)
        for hand, label in HAND_PARTS.items()
    }


def assembly_meshes_with_hands(
    club: ClubAssembly,
    pose: GripPose = DEFAULT_GRIP_POSE,
    *,
    hand_length_m: float | None = None,
    hands: ResolvedHandSpec | None = None,
) -> dict[str, Mesh]:
    """Shaft, grip and head of :func:`assembly_meshes`, then both grip hands.

    ``hands`` (#13995), when given, supplies the pose and hand length and
    overrides ``pose`` and ``hand_length_m``.
    """
    if hands is not None:
        pose, hand_length_m = hands.pose, hands.hand_length_m
    return {
        **assembly_meshes(club),
        **grip_hand_meshes(club, pose, hand_length_m=hand_length_m, hands=hands),
    }
