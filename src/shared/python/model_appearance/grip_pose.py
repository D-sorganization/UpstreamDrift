"""Shared two-hand grip-pose definition (OSV-2, #11728).

One place defines where a right-handed golfer's hands sit on the grip, so that
every engine (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite) and every builder
takes the same numbers.  Nothing here depends on an engine.

Conventions
-----------
Distances are metres measured **down the shaft from the butt end** of the grip
(``distance_below_butt_m``); the tour-matching OpenSim club frame
(``club_geometry``) has its origin at the butt and the shaft along ``-y``, so
``club_y_m`` is the negated distance.

Angles are rotations about the grip axis ``g`` (butt toward head) in the plane
perpendicular to the shaft.  With ``n`` the face normal (toward the target) and
``t`` the toe axis (heel toward toe), azimuth 0 is the direction ``-n`` (away
from the target, toward the golfer's chin line) and azimuth increases toward
``-t`` (toward the golfer, i.e. toward the trail shoulder).

Values are nominal coaching-convention parameters for a neutral grip, not
measurements: the captures carry no hand markers, so the hand roll that the
data can constrain is fitted separately (``motion_matching.grip_fit``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require

#: Nominal point of the hand body (metres, hand frame) that sits on the grip:
#: about 6 cm distal to the wrist origin along the hand's ``-y``.  The OSV-9
#: Rajagopal models do not use it: their hand grip frames come from the
#: committed address calibration (``msk_club.HAND_GRIP_POINT_M``).
HAND_GRIP_ANCHOR_IN_HAND_M: tuple[float, float, float] = (0.0, -0.06, 0.0)

#: Hand-centre spacing along the shaft.  3 in (0.0762 m): the separation of the
#: trail closure placement and the lead wrist joint in the native Simscape spec
#: (``full_body_spec_v2.json``: ``closure.placement_b`` y = -1.0 m vs the lead
#: joint ``child_to_follower`` y = -1.0762 m in the club frame).
NATIVE_HAND_SPACING_M = 0.0762
#: Lead-hand grip point below the butt end.  The single source of truth is the
#: generated anthropometric spec (``GripInterface.from_spec`` of
#: ``full_body_spec_anthro_{driver,iron7}.json``, which the OSV-9 musculoskeletal
#: club and its committed address calibration use): 3.2 cm for both clubs.  It
#: is mirrored here, not read at import, because the grip pose is club-agnostic
#: and engine-free; ``tests/opensim/test_grip_closure_both_hands.py`` fails if
#: the two drift apart.
LEAD_BELOW_BUTT_M = 0.032
#: Angle of the shaft across the palm, from the hand's radial (thumb) axis
#: toward the fingers: in a golf grip the shaft runs diagonally from the heel
#: pad (ulnar, proximal) to the base of the index finger (radial, distal).
#: Moved here from the OpenSim calibration (#12306) so every engine takes the
#: same hand orientation on the grip (OSV-2 item 4, #11728).
PALM_DIAGONAL_DEG = 40.0
#: Tolerance (dimensionless) on the unit length and orthogonality of the
#: hand-frame axes passed to :meth:`GripPose.shaft_direction_in_hand`.
AXIS_TOLERANCE = 1e-9
#: Side of the grip each palm faces, as a sign on the target direction: the
#: trail palm faces the target, the lead palm faces away, so the palms are
#: parallel and opposed across the grip.
_PALM_TARGET_SIGN: dict[str, float] = {"lead": -1.0, "trail": 1.0}


class Hand(str, Enum):
    """The two hands of a right-handed golfer (lead = left, trail = right)."""

    LEAD = "lead"
    TRAIL = "trail"


class GripStyle(str, Enum):
    """How the little finger of the trail hand relates to the lead hand."""

    OVERLAP = "overlap"
    INTERLOCK = "interlock"
    TEN_FINGER = "ten_finger"


_FINGER_LINKS: dict[GripStyle, tuple[str, ...]] = {
    GripStyle.OVERLAP: ("trail_little_over_lead_index_middle_gap",),
    GripStyle.INTERLOCK: ("trail_little_between_lead_index_middle",),
    GripStyle.TEN_FINGER: (),
}


@dataclass(frozen=True)
class HandPlacement:
    """One hand on the grip.

    ``thumb_offset_deg`` is the azimuth of the thumb away from the top-centre
    of the grip toward the trail side (zero for a thumb straight down the top).
    """

    hand: Hand
    distance_below_butt_m: float
    v_azimuth_deg: float
    thumb_offset_deg: float


@dataclass(frozen=True)
class GripPose:
    """Neutral two-hand grip for a right-handed golfer.

    Preconditions (checked in ``__post_init__``): finite values,
    ``lead_below_butt_m`` in [0, 0.1], ``hand_spacing_m`` in (0, 0.2],
    ``lead_v_azimuth_deg`` in [0, 90], ``lead_thumb_offset_deg`` in [0, 45],
    ``style`` a :class:`GripStyle`.
    Postcondition: the trail hand is strictly below the lead hand on the grip.
    """

    style: GripStyle = GripStyle.OVERLAP
    lead_below_butt_m: float = LEAD_BELOW_BUTT_M
    hand_spacing_m: float = NATIVE_HAND_SPACING_M
    #: Lead "V" (thumb/index crease) between the chin (~10 deg) and the trail
    #: shoulder (~35 deg).
    lead_v_azimuth_deg: float = 20.0
    #: Trail V relative to the lead V; zero keeps the two parallel.
    trail_v_offset_deg: float = 0.0
    #: Short lead thumb, slightly to the trail side of centre.
    lead_thumb_offset_deg: float = 10.0
    #: Trail life line covers the lead thumb (true for all three styles).
    trail_lifeline_over_lead_thumb: bool = True
    #: Shaft angle across the palm from the radial axis toward the fingers.
    palm_diagonal_deg: float = PALM_DIAGONAL_DEG

    def __post_init__(self) -> None:
        require(
            isinstance(self.style, GripStyle),
            f"style must be a GripStyle, got {self.style!r}",
        )
        for name, lo, hi in (
            ("lead_below_butt_m", 0.0, 0.1),
            ("hand_spacing_m", 0.0, 0.2),
            ("lead_v_azimuth_deg", 0.0, 90.0),
            ("trail_v_offset_deg", -30.0, 30.0),
            ("lead_thumb_offset_deg", 0.0, 45.0),
            ("palm_diagonal_deg", 0.0, 90.0),
        ):
            value = getattr(self, name)
            require(
                isinstance(value, (int, float)) and math.isfinite(value),
                f"{name} must be a finite number, got {value!r}",
            )
            require(lo <= value <= hi, f"{name} must be in [{lo}, {hi}], got {value}")
        require(self.hand_spacing_m > 0.0, "hand_spacing_m must be positive")
        ensure(
            self.distance_below_butt_m(Hand.TRAIL)
            > self.distance_below_butt_m(Hand.LEAD),
            "trail hand must be below the lead hand",
        )

    def distance_below_butt_m(self, hand: Hand) -> float:
        """Distance of the hand centre below the butt end along the shaft."""
        require(isinstance(hand, Hand), f"hand must be a Hand, got {hand!r}")
        if hand is Hand.LEAD:
            return self.lead_below_butt_m
        return self.lead_below_butt_m + self.hand_spacing_m

    def club_y_m(self, hand: Hand) -> float:
        """Hand-centre y in the OpenSim club frame (origin at the butt, -y shaft)."""
        return -self.distance_below_butt_m(hand)

    def placement(self, hand: Hand) -> HandPlacement:
        """Full placement of one hand (position along the grip and angles)."""
        require(isinstance(hand, Hand), f"hand must be a Hand, got {hand!r}")
        if hand is Hand.LEAD:
            azimuth, thumb = self.lead_v_azimuth_deg, self.lead_thumb_offset_deg
        else:
            azimuth, thumb = self.lead_v_azimuth_deg + self.trail_v_offset_deg, 0.0
        return HandPlacement(hand, self.distance_below_butt_m(hand), azimuth, thumb)

    def linked_fingers(self) -> tuple[str, ...]:
        """The finger link between the hands, as a one-element tuple, or none."""
        return _FINGER_LINKS[self.style]

    def palm_target_sign(self, hand: Hand) -> float:
        """Sign on the target direction of the side ``hand``'s palm faces.

        ``+1`` for the trail palm (toward the target), ``-1`` for the lead palm
        (away from it). Raises ``ValueError`` for anything but a :class:`Hand`.
        """
        if not isinstance(hand, Hand):
            raise ValueError(f"hand must be a Hand, got {hand!r}")
        return _PALM_TARGET_SIGN[hand.value]

    def shaft_direction_in_hand(
        self,
        hand: Hand,
        radial_axis: Sequence[float] | NDArray[np.float64],
        distal_axis: Sequence[float] | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Head-end shaft direction in ``hand``'s own body frame.

        ``radial_axis`` (toward the thumb) and ``distal_axis`` (toward the
        fingertips) are the hand frame's anatomical axes, a fact of each
        engine's hand body; the angle between them is the shared
        :attr:`palm_diagonal_deg`. Preconditions: ``hand`` a :class:`Hand`;
        both axes finite unit 3-vectors, mutually orthogonal, else
        ``ValueError``. Postcondition: a unit vector at ``palm_diagonal_deg``
        from ``radial_axis``, tilted toward ``distal_axis``.
        """
        self.palm_target_sign(hand)  # validates the hand
        radial = _unit_axis(radial_axis, "radial_axis")
        distal = _unit_axis(distal_axis, "distal_axis")
        if abs(float(radial @ distal)) > AXIS_TOLERANCE:
            raise ValueError("radial_axis and distal_axis must be orthogonal")
        angle = math.radians(self.palm_diagonal_deg)
        return math.cos(angle) * radial + math.sin(angle) * distal

    @property
    def interlocked(self) -> bool:
        """True when the little finger passes between the lead fingers."""
        return self.style is GripStyle.INTERLOCK


def _unit_axis(
    axis: Sequence[float] | NDArray[np.float64], name: str
) -> NDArray[np.float64]:
    """``axis`` as a float array, checked to be a finite unit 3-vector."""
    vector = np.asarray(axis, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector, got shape {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite, got {vector}")
    if abs(float(np.linalg.norm(vector)) - 1.0) > AXIS_TOLERANCE:
        raise ValueError(f"{name} must be a unit vector, got {vector}")
    return vector


DEFAULT_GRIP_POSE = GripPose()
