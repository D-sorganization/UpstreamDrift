"""Golf-language address grip roll, measured face-on (SHOW-A1, #14550).

The owner reads the grip off the face-on view: **0 deg = the hand's palm plane is
vertical** (the neutral grip), **+ = strong** (the back of the lead hand turns
toward the sky, clockwise from the golfer's view). ``grip_contract`` /
``grip_fit`` describe the same hands as ``(alpha, psi, gamma)`` relative to the
*native hand frame*; that is a model-frame quantity and is not what the owner
sees (the flagship read weak, alpha = -25 deg, while the render shows ~90 deg).

Definition
----------
* ``shaft`` is the butt-to-head axis, club-frame ``y``.
* ``(e_h, e_v)`` span the plane perpendicular to the shaft: ``e_h`` is the
  horizontal one pointing toward the target, ``e_v = shaft x e_h`` points up.
* The hand normal ``n`` is the back of the lead hand and the palm of the trail
  hand. Both are ``-dorsal``, with ``dorsal`` the wrist-follower x axis of the
  hand (``hand_spec.wrist_follower_dorsal_axis``, the axis the hand meshes are
  drawn from), re-expressed in the world by the native club frame.
* ``roll = atan2(n . e_v, n . e_h)``, in degrees, after projecting ``n`` off the
  shaft.

Mapping to HX (#13757): ``alpha`` is a rotation of the hand about the shaft, so
a change ``d_alpha`` changes the roll by exactly ``d_alpha``
(``test_roll_is_invariant_to_spin...``). ``psi`` and ``gamma`` tilt the hand
about axes perpendicular to the shaft and move only the *projected* normal's
length, not (to first order) its direction. The roll equals ``alpha`` plus a
constant set by the native frame and the club lie at address.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.impact_parameters.target_frame import TargetFrame
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_spec import wrist_follower_dorsal_axis

Array = NDArray[np.float64]

_TOL = 1e-9
_MIN_PROJECTION = 1e-6
_SHAFT_IN_CLUB = np.array([0.0, 1.0, 0.0])


def _unit3(value: Any, name: str) -> Array:
    arr = np.asarray(value, dtype=float)
    require(
        arr.shape == (3,) and bool(np.isfinite(arr).all()),
        f"{name} must be a finite 3-vector",
    )
    size = float(np.linalg.norm(arr))
    require(size > _MIN_PROJECTION, f"{name} must be non-zero")
    return arr / size


@dataclass(frozen=True)
class GripPoseState:
    """Pose of the grip: the native club frame and each hand's dorsal axis.

    ``club_rotation`` is the world <- native club frame rotation (column 1 is the
    shaft, butt to head). The dorsal axes are expressed in that club frame. They
    are rigid (they depend on the grip rotation, not on the wrist angles), so a
    state is fully determined by the club orientation at address.
    """

    club_rotation: Array
    lead_dorsal_club: Array
    trail_dorsal_club: Array
    target: TargetFrame = field(default_factory=TargetFrame)


def reference_axes(shaft: Any, target: TargetFrame) -> tuple[Array, Array]:
    """``(e_h, e_v)``: horizontal-toward-target and up, both perpendicular to shaft.

    Preconditions: a non-vertical shaft; a right-handed target frame.
    """
    require(
        target.handedness == "right",
        "left-handed mirroring of the grip roll is not specified yet",
    )
    s = _unit3(shaft, "shaft")
    up = _unit3(target.up, "up")
    e_h = np.cross(up, s)
    require(
        float(np.linalg.norm(e_h)) > _MIN_PROJECTION,
        "the shaft must not be vertical",
    )
    e_h = e_h / np.linalg.norm(e_h)
    if e_h @ _unit3(target.target_dir, "target_dir") < 0.0:
        e_h = -e_h
    e_v = np.cross(s, e_h)
    ensure(float(e_v @ up) > 0.0, "e_v must point up", e_v)
    return e_h, e_v


def _roll_deg(normal_world: Array, e_h: Array, e_v: Array, shaft: Array) -> float:
    projected = normal_world - float(normal_world @ shaft) * shaft
    require(
        float(np.linalg.norm(projected)) > _MIN_PROJECTION,
        "the hand normal must not lie along the shaft",
    )
    return math.degrees(math.atan2(float(projected @ e_v), float(projected @ e_h)))


def address_grip_roll_deg(state: GripPoseState) -> tuple[float, float]:
    """``(lead, trail)`` face-on hand roll in degrees: 0 = vertical, + = strong.

    Preconditions: ``club_rotation`` is a proper rotation (orthonormal, det +1);
    dorsal axes are non-zero and not along the shaft. Postcondition: both values
    lie in (-180, 180].
    """
    rot = np.asarray(state.club_rotation, dtype=float)
    require(
        rot.shape == (3, 3)
        and bool(np.isfinite(rot).all())
        and bool(np.allclose(rot @ rot.T, np.eye(3), atol=1e-6))
        and abs(float(np.linalg.det(rot)) - 1.0) < 1e-6,
        "club_rotation must be an orthonormal frame with determinant +1",
    )
    shaft = rot @ _SHAFT_IN_CLUB
    e_h, e_v = reference_axes(shaft, state.target)
    lead = _roll_deg(
        -(rot @ _unit3(state.lead_dorsal_club, "lead_dorsal_club")), e_h, e_v, shaft
    )
    trail = _roll_deg(
        -(rot @ _unit3(state.trail_dorsal_club, "trail_dorsal_club")), e_h, e_v, shaft
    )
    ensure(
        -180.0 < lead <= 180.0 and -180.0 < trail <= 180.0,
        "roll must lie in (-180, 180]",
        (lead, trail),
    )
    return lead, trail


def grip_pose_state_from_spec(
    spec_document: Mapping[str, Any],
    club_rotation: Any,
    target: TargetFrame | None = None,
) -> GripPoseState:
    """State of a full-body spec (v1 or v2) given its native club frame in the world.

    The dorsal axes come from the spec's wrist joints, the same source the hand
    meshes use, so the metric reads the model, not a re-derived copy. The caller
    supplies ``club_rotation`` from its engine (for MuJoCo, the orientation of
    the ``LW`` frame site). Precondition: both wrist joints exist in the spec.
    """
    lead = wrist_follower_dorsal_axis(spec_document, Hand.LEAD)
    trail = wrist_follower_dorsal_axis(spec_document, Hand.TRAIL)
    require(
        lead is not None and trail is not None,
        "the spec has no wrist-follower joints for both hands",
    )
    return GripPoseState(
        club_rotation=np.asarray(club_rotation, dtype=float),
        lead_dorsal_club=lead,
        trail_dorsal_club=trail,
        target=target or TargetFrame(),
    )
