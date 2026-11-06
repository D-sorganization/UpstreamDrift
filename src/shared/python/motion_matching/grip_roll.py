"""Golf-language address grip roll, measured face-on (SHOW-A1 #14550, A7 #14731).

The owner reads the grip off the face-on view as a clock: **0 deg = the back of
the wrist points straight up, on top of the shaft** (12 o'clock); **+ =
counterclockwise** on that clock, from 12 toward the target side (9 o'clock is
+90 deg). The default address grip is +15 deg. The first definition (A1) took
0 deg as "the palm plane is vertical", i.e. the back of the wrist at 9 o'clock,
so the shipped "15 deg" was 75 deg from this zero. ``grip_contract`` describes
the hands as ``(alpha, psi, gamma)`` relative to the native hand frame, which is
not what the owner sees.

Definition: ``shaft`` is club-frame ``y`` (butt to head); ``e_h`` is the
horizontal direction toward the target perpendicular to it and ``e_v = shaft x
e_h`` points up. The wrist axis ``b`` is ``-dorsal`` for both hands, where
``dorsal`` is the wrist-follower x axis (``hand_spec.wrist_follower_dorsal_axis``,
one club-frame vector for lead and trail) taken into the world by the native club
frame. The follower x points to the PALM side of the lead hand, so ``-x`` is the
BACK of the lead hand; the drawn lead hand is spun about the shaft by
``grip_hands.WRIST_FRAME_TO_MESH_DEG`` so its dorsum lies on that axis (D68 measured
the unspun mesh dorsum 35 deg off +x). The trail hand is read on the same axis; its
drawn back sits 110 deg from the lead's (-95 deg at the default). ``roll =
atan2(b.e_h, b.e_v)`` after projecting ``b`` off the shaft. One rule reads both hands.

Mapping to HX (#13757): alpha is a rotation about the shaft, so ``d_alpha``
changes the roll by exactly ``-d_alpha`` (``test_spin_about_the_shaft...``: the
clock counts the other way); psi and gamma tilt about axes perpendicular to the
shaft and are seen here only through the projected axis. The roll is a constant
fixed by the native frame and the club lie at address, minus alpha.
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
    e_h = np.asarray(e_h / np.linalg.norm(e_h), dtype=np.float64)
    if e_h @ _unit3(target.target_dir, "target_dir") < 0.0:
        e_h = -e_h
    e_v = np.cross(s, e_h)
    ensure(float(e_v @ up) > 0.0, "e_v must point up", e_v)
    return np.asarray(e_h, dtype=np.float64), np.asarray(e_v, dtype=np.float64)


def _roll_deg(axis_world: Array, e_h: Array, e_v: Array, shaft: Array) -> float:
    projected = axis_world - float(axis_world @ shaft) * shaft
    require(
        float(np.linalg.norm(projected)) > _MIN_PROJECTION,
        "the wrist axis must not lie along the shaft",
    )
    return math.degrees(math.atan2(float(projected @ e_h), float(projected @ e_v)))


def address_grip_roll_deg(state: GripPoseState) -> tuple[float, float]:
    """``(lead, trail)`` face-on wrist roll: 0 = back of wrist up, + = counterclockwise.

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
    if lead is None or trail is None:
        raise ValueError("the spec has no wrist-follower joints for both hands")
    return GripPoseState(
        club_rotation=np.asarray(club_rotation, dtype=float),
        lead_dorsal_club=lead,
        trail_dorsal_club=trail,
        target=target or TargetFrame(),
    )
