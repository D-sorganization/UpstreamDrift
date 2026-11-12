"""Golf-language address grip roll, measured face-on (SHOW-A1 #14550, A7 #14731).

The owner reads the grip off the face-on view as a clock: **0 deg = the back of
the wrist points straight up, on top of the shaft** (12 o'clock); **+ =
counterclockwise** on that clock, from 12 toward the target side (9 o'clock is
+90 deg). The default address grip is +15 deg. The first definition (A1) took
0 deg with the back of the wrist at 9 o'clock (horizontal palm plane),
so the shipped "15 deg" was 75 deg from this zero. ``grip_contract`` describes
the hands as ``(alpha, psi, gamma)`` relative to the native hand frame, which is
not what the owner sees.

Definition: ``shaft`` is club-frame ``y`` (butt to head); ``e_h`` is the
horizontal direction toward the target perpendicular to it and ``e_v = shaft x
e_h`` points up. The wrist axis ``b`` is ``-dorsal`` for both hands: the back of
the LEAD hand (the follower x axis points to the palm side of the lead hand; the
trail hand is read on the same axis, so its back is the opposite end, 180 deg
away, as in a real grip). ``dorsal`` is the wrist-follower x axis
(``hand_spec.wrist_follower_dorsal_axis``, one club-frame vector for lead and
trail) taken into the world by the native club frame. ``roll = atan2(b.e_h,
b.e_v)`` after projecting ``b`` off the shaft. One rule reads both hands.

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
from typing import Any, Final

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


#: Minimum evidence-based dorsal separation between lead and trail hands (degrees).
#:
#: Derivation from prehensile mechanics and golf-instruction evidence (SHOW-A21, #14797):
#: 1. Prehensile opposition geometry (Napier 1956, "The Prehensile Movements of
#:    the Human Hand", J. Bone Joint Surg. 38B:902-913):
#:    In a two-handed bilateral cylindrical grasp, stable prehension requires the
#:    two hands to apply opposing normal forces across the cylinder. The dorsum
#:    normal unit vectors d_lead and d_trail must face into opposite hemispheres
#:    (dot product d_lead . d_trail < 0, corresponding to dorsal separation > 90 deg).
#:    At separation = 90 deg (d_lead . d_trail = 0), the normals are orthogonal,
#:    marking the exact geometric boundary between opposing grasp (palms on opposite
#:    sides) and non-opposing grasp (palms facing into the same quadrant).
#:    Separation < 90 deg (d_lead . d_trail > 0) means the dorsal surfaces face into
#:    the same hemisphere, so the palms face the same side and cannot clamp the shaft
#:    against each other.
#: 2. Distinction between 3D wrist frame dorsum normal and 2D frontal-plane V crease:
#:    - The 3D wrist-follower dorsal axis (hand_spec.wrist_follower_dorsal_axis)
#:      is orthogonal to the metacarpal palmar plane in 3D space.
#:    - The 2D thumb-index "V" crease sits on the radial border of the hand,
#:      rotated ~35-75 deg relative to the wrist dorsum normal.
#:    - Because the shaft lies diagonally across the palm (PALM_DIAGONAL_DEG = 40 deg),
#:      wrist flexion/deviation couples with shaft rotation.
#:    - In textbook neutral (Hogan 1957; Leadbetter 1990; Cochran & Stobbs 1968),
#:      when the 2D V creases are parallel (trail V-offset = 0 deg, option c),
#:      the 3D wrist dorsal axes are directly opposed across the shaft:
#:      dorsal separation = 180.0 deg (and drawn-dorsum mesh separation is ~70 deg).
#: 3. Instructional variation across weak, neutral, and strong grips:
#:    - In standard grips (Hogan 1957, pp. 19-34; Leadbetter 1990, ch. 2), both hands
#:      rotate concordantly, keeping dorsal separation near 180 deg (~150-180 deg).
#:    - In tournament player variations (Cochran & Stobbs 1968, pp. 50-57), palm plane
#:      divergence rarely exceeds 30-45 deg (separation 135-180 deg), with an extreme
#:      functional tolerance of ~60-65 deg (separation >= 115 deg).
#: 4. Evaluation of G2 options (#14751):
#:    - Option (c) parallel V's: separation = 180.00 deg (d_lead . d_trail = -1.0, passes).
#:    - Option (b15) trail alpha 15: separation = 115.89 deg (d_lead . d_trail = -0.44, passes).
#:    - Flagship iron7: separation = 124.08 deg (passes).
#:    - Option (a) pure mirror: separation = 81.77 deg (d_lead . d_trail = +0.14 > 0,
#:      fails: backs face into the same hemisphere, violating prehensile opposition).
#:    - Copy grip (main driver bug): separation = 0.00 deg (d_lead . d_trail = +1.0,
#:      fails: parallel backs, same hand twice).
MIN_DORSAL_SEPARATION_DEG: Final[float] = 90.0


def dorsal_separation_deg(lead_dorsal: Any, trail_dorsal: Any) -> float:
    """Angular separation in degrees between lead and trail dorsal axes in [0, 180].

    Preconditions: both vectors are non-zero finite 3-vectors.
    """
    u_lead = _unit3(lead_dorsal, "lead_dorsal")
    u_trail = _unit3(trail_dorsal, "trail_dorsal")
    dot = float(np.clip(u_lead @ u_trail, -1.0, 1.0))
    return math.degrees(math.acos(dot))


def validate_dorsal_separation(
    lead_dorsal: Any,
    trail_dorsal: Any,
    *,
    min_deg: float = MIN_DORSAL_SEPARATION_DEG,
) -> float:
    """Validate that lead and trail dorsal separation meets or exceeds the evidence-based bound.

    Preconditions: both vectors are non-zero finite 3-vectors.
    Returns: measured dorsal separation in degrees.
    Raises: PreconditionError if separation < min_deg.
    """
    separation = dorsal_separation_deg(lead_dorsal, trail_dorsal)
    require(
        separation >= min_deg,
        f"dorsal separation {separation:.2f} deg must be >= {min_deg:.1f} deg "
        f"(evidence-based prehensile opposition bound, #14797)",
    )
    return separation


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

    @property
    def dorsal_separation_deg(self) -> float:
        """Angular separation in degrees between lead and trail dorsal axes."""
        return dorsal_separation_deg(self.lead_dorsal_club, self.trail_dorsal_club)

    def validate_dorsal_separation(
        self, min_deg: float = MIN_DORSAL_SEPARATION_DEG
    ) -> float:
        """Validate that dorsal separation meets or exceeds the evidence-based bound."""
        return validate_dorsal_separation(
            self.lead_dorsal_club, self.trail_dorsal_club, min_deg=min_deg
        )


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


def spec_dorsal_separation_deg(
    spec_document: Mapping[str, Any],
    pose: Any = None,
) -> float | None:
    """Dorsal separation in degrees between lead and trail wrist followers of a spec.

    Returns None if the spec lacks wrist followers for both hands.
    """
    kwargs = {"pose": pose} if pose is not None else {}
    lead = wrist_follower_dorsal_axis(spec_document, Hand.LEAD, **kwargs)
    trail = wrist_follower_dorsal_axis(spec_document, Hand.TRAIL, **kwargs)
    if lead is None or trail is None:
        return None
    return dorsal_separation_deg(lead, trail)
