"""Trail-palm-on-lead-thumb contact patch frame (HRG-H1, #13189).

In a right-handed grip the trail (right) palm's lifeline pad presses on the lead
(left) thumb (``GripPose.trail_lifeline_over_lead_thumb``). This module provides
the one patch point and normal of that pair, in the club-body frame. It is the
only place the patch geometry is decided: a later grip-configuration source
(HRG-GR2, ``GripSpec``) replaces its inputs, not its callers.

Definition (binding)
--------------------
Work in the lead hand-grip frame (``grip_hand_frame``: origin on the grip axis
at the lead hand centre, ``z`` butt toward head, ``x`` the toe axis, ``y = z x x``).

* ``Q``, the palm reference, is the vertex centroid of the *trail* hand's palm
  shell, moved along the grip axis by the hand spacing of the pose.
* The lead thumb is the capsule of ``HandMeshParams`` (radius ``thumb_radius_m``,
  axis at the thumb azimuth, radius ``grip_radius + pad_clearance + r``, between
  the declared stations ``thumb_stations_m[0]`` and ``[2]``).
* ``A`` is the point of the thumb axis segment nearest ``Q``; the normal is
  ``n = (Q - A) / |Q - A|`` (from the thumb toward the palm), and the patch
  point is ``P = A + r n``: the thumb surface point nearest the palm.

A left-handed golfer's grip is the reflection of the right-handed one in the
plane of the shaft and the face normal (the hand-grip frame ``x`` axis is the
mirror normal), so the left-handed patch is the exact reflection of the
right-handed one. ``handedness`` is ``subject_handedness(spec)`` (default ``"right"``).
Visual-geometry reference only: nothing here touches a mass, an
inertia or a constraint.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.handedness import (
    lead_side,
    trail_side,
    validate_handedness,
)
from src.shared.python.grip_contact.interface import (
    appearance_to_shared_grip_frame,
    shared_to_appearance_grip_frame,
)
from src.shared.python.model_appearance.grip_hand_mesh import (
    DEFAULT_HAND_MESH_PARAMS,
    GRIP_RADIUS_RANGE_M,
    HandMeshParams,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)

Array = NDArray[np.float64]
Vec3 = tuple[float, float, float]

#: Tolerance on the orthonormality of the lead hand-grip frame rotation.
ROTATION_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ThumbPatch:
    """Palm-on-thumb patch in the club-body frame.

    ``normal_club`` is the unit normal from the lead thumb toward the trail palm,
    so a compressive palm force on the thumb acts along ``-normal_club``.
    ``thumb_axis_hand_m`` is the thumb axis point of the patch in the lead
    hand-grip frame (its ``x, y`` are the axis position, ``z`` the station).
    """

    point_club_m: Vec3
    normal_club: Vec3
    palm_reference_club_m: Vec3
    thumb_axis_hand_m: Vec3
    station_m: float
    thumb_radius_m: float
    handedness: str
    lead_side: str
    trail_side: str


def _vec3(x: Sequence[float] | Array) -> Vec3:
    return (float(x[0]), float(x[1]), float(x[2]))


def _check(
    lead_frame: object,
    grip_radius_m: object,
    pose: object,
    handedness: object,
    params: object,
) -> Array:
    validate_handedness(handedness)
    frame = np.asarray(lead_frame, dtype=float)
    if frame.shape != (4, 4) or not np.isfinite(frame).all():
        raise ValueError("lead_frame must be a finite 4x4 transform")
    rot = frame[:3, :3]
    if not (
        np.allclose(rot.T @ rot, np.eye(3), atol=ROTATION_TOLERANCE)
        and math.isclose(float(np.linalg.det(rot)), 1.0, abs_tol=ROTATION_TOLERANCE)
    ):
        raise ValueError("lead_frame rotation must be a proper rotation")
    lo, hi = GRIP_RADIUS_RANGE_M
    if isinstance(grip_radius_m, bool) or not isinstance(grip_radius_m, (int, float)):
        raise TypeError(f"grip_radius_m must be a number, got {grip_radius_m!r}")
    if not (math.isfinite(grip_radius_m) and lo <= grip_radius_m <= hi):
        raise ValueError(
            f"grip_radius_m must be in [{lo}, {hi}] m, got {grip_radius_m}"
        )
    if not isinstance(pose, GripPose):
        raise TypeError(f"pose must be a GripPose, got {type(pose).__name__}")
    if not isinstance(params, HandMeshParams):
        raise TypeError(f"params must be HandMeshParams, got {type(params).__name__}")
    return frame


def _palm_reference(
    grip_radius_m: float, pose: GripPose, params: HandMeshParams
) -> Array:
    """Trail palm centroid in the lead hand-grip frame (right-handed grip)."""
    trail = build_grip_hand_mesh(
        Hand.TRAIL, grip_radius_m=grip_radius_m, pose=pose, params=params
    )
    palm = next(c for c in trail.components if c.finger == "palm")
    shift = pose.distance_below_butt_m(Hand.TRAIL) - pose.distance_below_butt_m(
        Hand.LEAD
    )
    return palm.mesh.vertices.mean(axis=0) + np.array([0.0, 0.0, shift])


def _patch_in_shared_frame(
    grip_radius_m: float, pose: GripPose, params: HandMeshParams
) -> tuple[Array, Array, Array, Array]:
    """(point, normal, palm reference, thumb axis point) in the shared grip frame (x = grip axis)."""
    k = params.scale
    radius = k * params.thumb_radius_m
    azimuth = math.radians(pose.placement(Hand.LEAD).thumb_offset_deg)
    offset = grip_radius_m + params.pad_clearance_m + radius
    axis_yz = offset * np.array([math.cos(azimuth), math.sin(azimuth)])
    palm_app = _palm_reference(grip_radius_m, pose, params)
    palm = appearance_to_shared_grip_frame(palm_app)
    x_lo, x_hi = k * params.thumb_stations_m[0], k * params.thumb_stations_m[2]
    station = float(np.clip(palm[0], x_lo, x_hi))
    axis_point = np.array([station, axis_yz[0], axis_yz[1]])
    toward = palm - axis_point
    normal = toward / np.linalg.norm(toward)
    return axis_point + radius * normal, normal, palm, axis_point


def _patch_in_hand_frame(
    grip_radius_m: float, pose: GripPose, params: HandMeshParams
) -> tuple[Array, Array, Array, Array]:
    """(point, normal, palm reference, thumb axis point) of a right-handed grip."""
    p_shared, n_shared, palm_shared, axis_shared = _patch_in_shared_frame(
        grip_radius_m, pose, params
    )
    return (
        shared_to_appearance_grip_frame(p_shared),
        shared_to_appearance_grip_frame(n_shared),
        shared_to_appearance_grip_frame(palm_shared),
        shared_to_appearance_grip_frame(axis_shared),
    )


def thumb_palm_patch(
    lead_frame: Array,
    *,
    grip_radius_m: float,
    pose: GripPose = DEFAULT_GRIP_POSE,
    handedness: str = "right",
    params: HandMeshParams = DEFAULT_HAND_MESH_PARAMS,
) -> ThumbPatch:
    """Palm-on-thumb patch point and normal in the club-body frame.

    ``lead_frame`` is the 4x4 lead hand-grip to club-body transform
    (``grip_hands.grip_hand_frame(club, Hand.LEAD, pose)``) and ``grip_radius_m``
    the grip radius the hands were baked on. Preconditions: a finite 4x4 with a
    proper rotation, a radius in ``GRIP_RADIUS_RANGE_M``, a :class:`GripPose`,
    :class:`HandMeshParams` and ``handedness`` ``"right"`` or ``"left"``
    (``ValueError``/``TypeError`` otherwise). Postconditions: the point lies on
    the lead thumb surface, ``normal_club`` is a unit vector toward the trail
    palm, and the left-handed patch is the reflection of the right-handed one.
    """
    frame = _check(lead_frame, grip_radius_m, pose, handedness, params)
    point, normal, palm, axis_point = _patch_in_hand_frame(
        float(grip_radius_m), pose, params
    )
    if handedness == "left":
        mirror = np.array([-1.0, 1.0, 1.0])
        point, normal, palm, axis_point = (
            v * mirror for v in (point, normal, palm, axis_point)
        )
    rot, origin = frame[:3, :3], frame[:3, 3]
    return ThumbPatch(
        point_club_m=_vec3(rot @ point + origin),
        normal_club=_vec3(rot @ normal),
        palm_reference_club_m=_vec3(rot @ palm + origin),
        thumb_axis_hand_m=_vec3(axis_point),
        station_m=float(axis_point[2]),
        thumb_radius_m=params.scale * params.thumb_radius_m,
        handedness=handedness,
        lead_side=lead_side(handedness),
        trail_side=trail_side(handedness),
    )
