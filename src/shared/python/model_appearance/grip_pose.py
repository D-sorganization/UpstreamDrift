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
from dataclasses import dataclass
from enum import Enum

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

    @property
    def interlocked(self) -> bool:
        """True when the little finger passes between the lead fingers."""
        return self.style is GripStyle.INTERLOCK


DEFAULT_GRIP_POSE = GripPose()
