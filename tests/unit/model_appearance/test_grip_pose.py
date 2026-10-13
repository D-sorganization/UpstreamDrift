"""Shared two-hand grip-pose definition (OSV-2 slice 1, #11728)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    HAND_GRIP_ANCHOR_IN_HAND_M,
    GripPose,
    GripStyle,
    Hand,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC_V2 = ROOT / "docs/development/full_body_models/full_body_spec_v2.json"


def test_lead_hand_is_above_trail_hand_along_the_grip() -> None:
    pose = DEFAULT_GRIP_POSE
    assert pose.distance_below_butt_m(Hand.LEAD) < pose.distance_below_butt_m(
        Hand.TRAIL
    )
    # OpenSim club frame: origin at the butt, shaft along -y, so lead is higher.
    assert pose.club_y_m(Hand.LEAD) > pose.club_y_m(Hand.TRAIL)
    assert pose.club_y_m(Hand.LEAD) < 0.0  # both hands are on the club


def test_lead_hand_sits_3p2_cm_below_the_butt_as_in_the_spec() -> None:
    assert DEFAULT_GRIP_POSE.distance_below_butt_m(Hand.LEAD) == pytest.approx(0.032)


def test_hand_spacing_matches_the_spec_closure_placements() -> None:
    """Trail closure placement_b.y vs the lead joint offset in the native spec."""
    spec = json.loads(SPEC_V2.read_bytes())
    closure = spec["closure"]
    trail_y = np.asarray(closure["placement_b"])[1, 3]
    joint = next(j for j in spec["joints"] if j["child"] == closure["body_b"])
    lead_y = np.asarray(joint["child_to_follower"])[1, 3]
    # Spec club frame: shaft toward the grip along -y, so the lead hand (nearer
    # the butt) is the more negative.
    assert lead_y < trail_y
    spacing = DEFAULT_GRIP_POSE.hand_spacing_m
    assert spacing == pytest.approx(trail_y - lead_y, abs=1e-3)


def test_single_source_for_the_opensim_hand_offsets() -> None:
    from src.engines.physics_engines.opensim.python.tour_matching import club_geometry

    assert DEFAULT_GRIP_POSE.club_y_m(Hand.LEAD) == club_geometry.LEAD_HAND_OFFSET_M
    assert DEFAULT_GRIP_POSE.club_y_m(Hand.TRAIL) == club_geometry.TRAIL_HAND_OFFSET_M


def test_neutral_v_angles() -> None:
    pose = DEFAULT_GRIP_POSE
    lead, trail = pose.placement(Hand.LEAD), pose.placement(Hand.TRAIL)
    # Lead V between the chin and the trail shoulder, within the coached band.
    assert 10.0 <= lead.v_azimuth_deg <= 35.0
    # Trail V parallel to the lead V.
    assert trail.v_azimuth_deg == pytest.approx(lead.v_azimuth_deg)
    # Lead thumb slightly to the trail side of centre; trail has no thumb offset
    # of its own because its life line covers the lead thumb.
    assert 0.0 < lead.thumb_offset_deg <= 20.0
    assert pose.trail_lifeline_over_lead_thumb is True


def test_default_style_is_overlap_and_options_exist() -> None:
    assert DEFAULT_GRIP_POSE.style is GripStyle.OVERLAP
    assert {s.value for s in GripStyle} == {"overlap", "interlock", "ten_finger"}
    assert DEFAULT_GRIP_POSE.linked_fingers() == (
        "trail_little_over_lead_index_middle_gap",
    )
    interlock = GripPose(style=GripStyle.INTERLOCK)
    assert interlock.linked_fingers() == ("trail_little_between_lead_index_middle",)
    assert interlock.interlocked is True and DEFAULT_GRIP_POSE.interlocked is False
    ten = GripPose(style=GripStyle.TEN_FINGER)
    assert ten.linked_fingers() == ()


def test_hand_grip_anchor_is_the_documented_hand_point() -> None:
    assert HAND_GRIP_ANCHOR_IN_HAND_M == (0.0, -0.06, 0.0)


def test_pose_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_GRIP_POSE.hand_spacing_m = 0.1  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lead_below_butt_m": -0.01},
        {"lead_below_butt_m": float("nan")},
        {"hand_spacing_m": 0.0},
        {"hand_spacing_m": 0.3},
        {"lead_v_azimuth_deg": 200.0},
        {"lead_thumb_offset_deg": -5.0},
        {"style": "overlap"},
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        GripPose(**kwargs)  # type: ignore[arg-type]


def test_placement_rejects_an_unknown_hand() -> None:
    with pytest.raises(ValueError, match="hand"):
        DEFAULT_GRIP_POSE.placement("left")  # type: ignore[arg-type]


def test_placements_are_finite() -> None:
    for hand in Hand:
        p = DEFAULT_GRIP_POSE.placement(hand)
        assert all(
            math.isfinite(v)
            for v in (p.distance_below_butt_m, p.v_azimuth_deg, p.thumb_offset_deg)
        )


def test_myosuite_club_sites_come_from_the_shared_pose() -> None:
    from src.engines.physics_engines.myosuite.python import golfer_scene
    from src.shared.python.motion_matching.club_models import CLUBS

    club = CLUBS["driver"]
    lead = golfer_scene._club_grip_site("l", club, Hand.LEAD)
    trail = golfer_scene._club_grip_site("r", club, Hand.TRAIL)

    def y(site: str) -> float:
        return float(site.split('pos="0 ')[1].split()[0])

    # Origin at the head, shaft toward -y: the lead hand is farther from the head.
    assert y(lead) < y(trail) < 0.0
    assert y(lead) == pytest.approx(-(club.length_m - 0.032))
    assert y(trail) - y(lead) == pytest.approx(DEFAULT_GRIP_POSE.hand_spacing_m)


def test_no_engine_redefines_the_hand_anchor_or_the_trail_offsets() -> None:
    """Engines import the shared numbers; no literal copies in engine code."""
    needles = ("(0.0, -0.06, 0.0)", "TRAIL_HAND_OFFSET_M: float = -", "-0.95 0")
    engines = ROOT / "src/engines"
    offenders = [
        str(p.relative_to(ROOT))
        for p in engines.rglob("*.py")
        if any(n in p.read_text(encoding="utf-8", errors="ignore") for n in needles)
    ]
    assert not offenders, offenders


# ---------------------------------------- wrist/hand orientation (item 4)
_RADIAL = np.array([0.0, 0.0, 1.0])
_DISTAL = np.array([0.0, -1.0, 0.0])


def test_shaft_lies_diagonally_across_the_palm() -> None:
    """Heel pad to index base: between the radial axis and the fingers."""
    assert 20.0 <= DEFAULT_GRIP_POSE.palm_diagonal_deg <= 60.0


@pytest.mark.parametrize("hand", list(Hand))
def test_shaft_direction_in_hand_is_the_palm_diagonal(hand: Hand) -> None:
    pose = DEFAULT_GRIP_POSE
    shaft = pose.shaft_direction_in_hand(hand, _RADIAL, _DISTAL)
    assert np.linalg.norm(shaft) == pytest.approx(1.0)
    from_radial = math.degrees(math.acos(float(shaft @ _RADIAL)))
    assert from_radial == pytest.approx(pose.palm_diagonal_deg)
    # tilted toward the fingers (distal), never toward the wrist
    assert float(shaft @ _DISTAL) > 0.0


def test_a_different_diagonal_turns_the_shaft_in_the_hand() -> None:
    steep = GripPose(palm_diagonal_deg=60.0)
    shaft = steep.shaft_direction_in_hand(Hand.LEAD, _RADIAL, _DISTAL)
    assert math.degrees(math.acos(float(shaft @ _RADIAL))) == pytest.approx(60.0)


def test_palms_face_each_other_across_the_grip() -> None:
    """Trail palm toward the target (+1), lead palm away from it (-1)."""
    assert DEFAULT_GRIP_POSE.palm_target_sign(Hand.TRAIL) == 1.0
    assert DEFAULT_GRIP_POSE.palm_target_sign(Hand.LEAD) == -1.0


@pytest.mark.parametrize(
    ("radial", "distal", "match"),
    [
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), "orthogonal"),
        ((0.0, 0.0, 0.0), (0.0, -1.0, 0.0), "unit"),
        ((0.0, 0.0, 1.0), (0.0, float("nan"), 0.0), "finite"),
        ((0.0, 1.0), (0.0, -1.0, 0.0), "3-vector"),
    ],
)
def test_shaft_direction_in_hand_contracts(
    radial: tuple[float, ...], distal: tuple[float, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        DEFAULT_GRIP_POSE.shaft_direction_in_hand(Hand.LEAD, radial, distal)


@pytest.mark.parametrize("value", [-1.0, 91.0, float("inf")])
def test_invalid_palm_diagonal_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="palm_diagonal_deg"):
        GripPose(palm_diagonal_deg=value)


def test_palm_target_sign_rejects_an_unknown_hand() -> None:
    with pytest.raises(ValueError, match="hand"):
        DEFAULT_GRIP_POSE.palm_target_sign("lead")  # type: ignore[arg-type]


def test_no_engine_defines_its_own_grip_orientation_angle() -> None:
    """Hand-on-grip angles live in grip_pose (or the grip_fit receipt) only."""
    import re

    pattern = re.compile(r"^(GRIP|WRIST|HAND|PALM)\w*_DEG\b\s*[:=]", re.MULTILINE)
    engines = ROOT / "src/engines"
    offenders = [
        str(p.relative_to(ROOT))
        for p in engines.rglob("*.py")
        if pattern.search(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not offenders, offenders
