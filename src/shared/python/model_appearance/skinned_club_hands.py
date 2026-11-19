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

from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
from typing import Any

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
    the little-finger knuckle; it lies ``owner_trail_cross_angle_deg`` off
    square across the trail hand, whose palm then rests on the lead thumb
    (#15265); the lead hand is ``owner_lead_below_cap_edge_m``
    below the edge of the butt cap. ``roll_deg`` turns both hands about the shaft.
    """
    base = GripSpec.legacy()
    quality: Mapping[str, Any] = load_hand_visual_quality()
    thumbs = quality["owner_thumb_azimuth_deg"]
    heel = float(quality["owner_lead_heel_pad_depth_m"])
    below = float(quality["owner_lead_below_cap_edge_m"])
    return replace(
        base,
        lead=replace(
            base.lead,
            pinky_side_palm_depth_m=heel,
            shaft_roll_deg=float(roll_deg),
            thumb_azimuth_deg=float(thumbs["lead"]),
        ),
        trail=replace(
            base.trail,
            shaft_roll_deg=float(roll_deg),
            thumb_azimuth_deg=float(thumbs["trail"]),
            club_cross_angle_deg=float(quality["owner_trail_cross_angle_deg"]),
        ),
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
    quality: Mapping[str, Any] = load_hand_visual_quality()
    radius = float(quality["grip_radius_m"])
    return build_grip_hands(owner_grip_spec(), grip_radius_m=radius)


def forearm_cuff_m() -> float:
    """Length of forearm skin kept on a hand drawn on the club (a setting)."""
    quality: Mapping[str, Any] = load_hand_visual_quality()
    return float(quality["club_hand_forearm_cuff_m"])


def _posed(result: HandVisualResult, bone: str) -> tuple[Array, Array]:
    """``bone``'s 4x4 from the rest frame to the pair's grip frame, and its rest head."""
    asset, transforms = result.rest_asset, result.bone_transforms
    if asset is None or transforms is None or result.frame_from_rest is None:
        raise ValueError("result carries no skinning audit data")
    i = asset.bone_index(bone)
    return result.frame_from_rest @ transforms[i], asset.bone_heads[i]


def _joint_m(result: HandVisualResult, bone: str) -> Array:
    """Posed head of ``bone`` in the pair's grip frame."""
    pose, head = _posed(result, bone)
    return np.asarray((pose @ np.append(head, 1.0))[:3], dtype=float)


def forearm_reach_m(club: ClubAssembly) -> dict[str, tuple[Array, Array]]:
    """Per hand: (wrist joint, unit direction from wrist to elbow), club frame."""
    frame, pair = skinned_hand_frame(club), _pair()
    reach = {}
    for side, result in pair.items():
        wrist = wrist_joint_m(result)
        along = _joint_m(result, "forearm") - wrist
        along = frame[:3, :3] @ (along / np.linalg.norm(along))
        reach[side] = (frame[:3, :3] @ wrist + frame[:3, 3], along)
    return reach


def forearm_axes(club: ClubAssembly) -> dict[str, Array]:
    """Per hand: 3x3 of the posed forearm's radial, distal, palmar axes, club frame.

    Distal (column 1) runs along the forearm bone from the elbow to the wrist;
    radial (column 0) is the bone's thumb-side axis squared to it; palmar
    (column 2) completes the triad, which is left-handed for the lead hand of
    a right-handed golfer (a mirrored right hand).
    """
    frame, pair, reach = skinned_hand_frame(club), _pair(), forearm_reach_m(club)
    axes = {}
    for side, result in pair.items():
        bone = frame[:3, :3] @ _posed(result, "forearm")[0][:3, :3]
        distal = -reach[side][1]
        radial = bone[:, 0] - (bone[:, 0] @ distal) * distal
        radial /= np.linalg.norm(radial)
        palmar = np.sign(np.linalg.det(bone)) * np.cross(radial, distal)
        axes[side] = np.column_stack([radial, distal, palmar])
    return axes


def _capped(vertices: Array, faces: NDArray[np.int64]) -> tuple[Array, NDArray]:
    """Close every open edge loop of a mesh with a fan about the loop's centre."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    paired = {(int(a), int(b)) for a, b in edges}
    following = {a: b for a, b in paired if (b, a) not in paired}
    points, fans = [vertices], []
    count = len(vertices)
    while following:
        start, nxt = following.popitem()
        loop = [start]
        while nxt != start:
            loop.append(nxt)
            nxt = following.pop(nxt)
        points.append(vertices[loop].mean(axis=0, keepdims=True))
        fans += [(loop[(k + 1) % len(loop)], loop[k], count) for k in range(len(loop))]
        count += 1
    if not fans:
        return vertices, faces
    return np.concatenate(points), np.concatenate([faces, np.array(fans)])


def _flat_rim(
    vertices: Array, faces: NDArray[np.int64], point: Array, normal: Array
) -> Array:
    """``vertices`` with every open-edge vertex moved along ``normal`` onto the
    plane through ``point``, so a cut that dropped whole triangles ends in one
    flat ring instead of a ragged edge."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    paired = {(int(a), int(b)) for a, b in edges}
    rim = sorted({a for a, b in paired if (b, a) not in paired})
    out = vertices.copy()
    out[rim] += np.outer((point - out[rim]) @ normal, normal)
    return out


def skinned_hand_meshes(
    club: ClubAssembly, *, cut_at_wrist: bool = True
) -> dict[str, Mesh]:
    """``{"lead_hand", "trail_hand"}`` skinned meshes in the club-body frame.

    The hands are posed on the standard golf grip of ``golf_grip.json``, not on
    the club's drawn grip tube, which is up to about a millimetre fatter. With
    ``cut_at_wrist`` every triangle that reaches more than
    :func:`forearm_cuff_m` up the forearm from the wrist joint is dropped, so a
    model's own forearm continues from the hand; the cut edge is moved onto
    that plane and closed flat.
    """
    frame, pair = skinned_hand_frame(club), _pair()
    reach = forearm_reach_m(club)
    meshes = {}
    for hand, label in HAND_PARTS.items():
        mesh = pair[hand.value].mesh
        vertices = mesh.vertices @ frame[:3, :3].T + frame[:3, 3]
        faces = mesh.faces
        if cut_at_wrist:
            wrist, along = reach[hand.value]
            keep = (vertices - wrist) @ along <= forearm_cuff_m()
            faces = faces[keep[faces].all(axis=1)]
            rim = wrist + forearm_cuff_m() * along
            vertices, faces = _capped(_flat_rim(vertices, faces, rim, along), faces)
        meshes[label] = Mesh(vertices, faces)
    return meshes


def wrist_joints_m(club: ClubAssembly) -> dict[str, Array]:
    """Wrist joint centre of each hand (``lead``, ``trail``) in the club-body frame.

    This is where a forearm attaches to the posed hand.
    """
    frame, pair = skinned_hand_frame(club), _pair()
    return {
        hand.value: frame[:3, :3] @ wrist_joint_m(pair[hand.value]) + frame[:3, 3]
        for hand in (Hand.LEAD, Hand.TRAIL)
    }
