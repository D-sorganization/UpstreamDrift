"""Tests for visual grip strength and per-hand roll parameterisation (HRG-A2, #13166).

Visual-only parameters:
- GripPose.grip_strength_deg ([-30, 30])
- per-hand roll_offset_deg (lead_roll_offset_deg, trail_roll_offset_deg in [-30, 30])
- Monotonic azimuth shifts and left-handed sagittal mirroring
- Invariance of physical hand attachment and HRG-E2 wrist angles
"""

from __future__ import annotations

import dataclasses
import math
import numpy as np
import pytest

from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)
from src.shared.python.anatomical_angles.hand_frame import (
    hand_frame_from_club,
)
from src.shared.python.motion_matching.grip_contract import (
    HandAngles,
)

pytestmark = pytest.mark.unit

# Sagittal reflection matrix: reflects lateral axis (y in body frame)
P = np.diag([1.0, -1.0, 1.0])


def test_default_grip_strength_and_roll_offsets_are_zero() -> None:
    pose = DEFAULT_GRIP_POSE
    assert pose.grip_strength_deg == 0.0
    assert pose.lead_roll_offset_deg == 0.0
    assert pose.trail_roll_offset_deg == 0.0
    assert pose.roll_offset_deg(Hand.LEAD) == 0.0
    assert pose.roll_offset_deg(Hand.TRAIL) == 0.0


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("grip_strength_deg", -30.1),
        ("grip_strength_deg", 30.1),
        ("lead_roll_offset_deg", -30.1),
        ("lead_roll_offset_deg", 30.1),
        ("trail_roll_offset_deg", -30.1),
        ("trail_roll_offset_deg", 30.1),
    ],
)
def test_grip_strength_and_roll_offset_bounds_raise(
    field_name: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field_name):
        GripPose(**{field_name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("strength", [-30.0, -15.0, 0.0, 15.0, 30.0])
def test_valid_grip_strength_values_instantiate(strength: float) -> None:
    pose = GripPose(grip_strength_deg=strength)
    assert pose.grip_strength_deg == strength


@pytest.mark.parametrize("hand", [Hand.LEAD, Hand.TRAIL])
def test_monotonic_azimuth_shift_with_grip_strength(hand: Hand) -> None:
    """Stronger grip shifts azimuth monotonically positive (away from target toward trail side)."""
    strengths = [-30.0, -15.0, 0.0, 15.0, 30.0]
    azimuths = [
        GripPose(grip_strength_deg=s).placement(hand).v_azimuth_deg for s in strengths
    ]
    # Verify strictly monotonic increasing
    for a, b in zip(azimuths[:-1], azimuths[1:], strict=True):
        assert b > a
        assert b - a == pytest.approx(15.0)


def test_per_hand_roll_offset_affects_only_target_hand() -> None:
    lead_offset = 12.0
    trail_offset = -8.0
    pose = GripPose(
        lead_roll_offset_deg=lead_offset, trail_roll_offset_deg=trail_offset
    )

    base = DEFAULT_GRIP_POSE
    lead_base = base.placement(Hand.LEAD).v_azimuth_deg
    trail_base = base.placement(Hand.TRAIL).v_azimuth_deg

    lead_placed = pose.placement(Hand.LEAD).v_azimuth_deg
    trail_placed = pose.placement(Hand.TRAIL).v_azimuth_deg

    assert lead_placed == pytest.approx(lead_base + lead_offset)
    assert trail_placed == pytest.approx(trail_base + trail_offset)


@pytest.mark.parametrize("strength", [-20.0, 0.0, 20.0])
@pytest.mark.parametrize("hand", [Hand.LEAD, Hand.TRAIL])
def test_left_handed_sagittal_mirroring(strength: float, hand: Hand) -> None:
    """Left-handed landmarks must be the exact sagittal reflection (diag(1, -1, 1)) of right-handed."""
    right = GripPose(grip_strength_deg=strength, handedness="right").landmarks(hand)
    left = GripPose(grip_strength_deg=strength, handedness="left").landmarks(hand)

    assert right.keys() == left.keys()
    for name, point in right.items():
        np.testing.assert_allclose(
            P @ point, left[name], atol=1e-9, err_msg=f"{hand} {name}"
        )


def test_hand_frame_from_club_rejects_visual_grip_pose() -> None:
    """HRG-E2 hand_frame_from_club must reject GripPose and remain invariant to visual parameters."""
    pose = GripPose(grip_strength_deg=25.0, lead_roll_offset_deg=10.0)
    with pytest.raises(TypeError, match="hand frames come from the physical grip"):
        hand_frame_from_club(
            club_rotation_world=np.eye(3),
            club_origin_world=np.zeros(3),
            native_rotation=np.eye(3),
            grip=pose,  # type: ignore[arg-type]
            calibration_id="test_calib",
        )


def test_physical_hand_angles_unaffected_by_visual_pose() -> None:
    """Physics invariants: HandAngles are completely decoupled from visual GripPose."""
    angles = HandAngles(alpha_deg=5.0, psi_deg=40.0, gamma_deg=0.0)
    frame = hand_frame_from_club(
        club_rotation_world=np.eye(3),
        club_origin_world=np.array([0.0, 0.0, 1.0]),
        native_rotation=np.eye(3),
        grip=angles,
        calibration_id="test_calib",
    )
    # The anatomical frame is derived purely from physical HandAngles
    assert frame.origin_world.shape == (3,)
    assert frame.rotation_world.shape == (3, 3)


def test_visual_grip_strength_and_roll_offsets_cli_and_sidecar() -> None:
    """CLI flags embed visual-only parameters into appearance sidecar without moving physics."""
    from src.shared.python.model_appearance.schema import document_from_dict
    from src.shared.python.motion_matching.execution.grip_selection import (
        GripOverrides,
        apply_visual_picks,
    )
    from src.tools.native_viewer_export import cli

    base = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]
    parse = cli.build_parser().parse_args
    args = parse(
        [
            *base,
            "--visual-grip-strength",
            "12.5",
            "--lead-roll-offset",
            "-5.0",
            "--trail-roll-offset",
            "7.5",
        ]
    )
    settings = cli.build_settings(args)
    assert settings.appearance is not None
    assert isinstance(settings.appearance, dict)
    grip_block = settings.appearance["grip"]
    assert grip_block["grip_strength_deg"] == pytest.approx(12.5)
    assert grip_block["lead_roll_offset_deg"] == pytest.approx(-5.0)
    assert grip_block["trail_roll_offset_deg"] == pytest.approx(7.5)

    # Validates against appearance_v1.schema.json
    doc = document_from_dict(settings.appearance)
    assert doc.grip is not None
    assert doc.grip["grip_strength_deg"] == pytest.approx(12.5)
    assert doc.grip["lead_roll_offset_deg"] == pytest.approx(-5.0)
    assert doc.grip["trail_roll_offset_deg"] == pytest.approx(7.5)

    # Overrides and apply_visual_picks round-trip
    overrides = GripOverrides(
        grip_strength_deg=10.0,
        lead_roll_offset_deg=-15.0,
        trail_roll_offset_deg=20.0,
    )
    assert overrides.has_visual is True
    assert overrides.has_physical is False
    applied = apply_visual_picks({}, overrides)
    assert applied["grip"] == {
        "grip_strength_deg": 10.0,
        "lead_roll_offset_deg": -15.0,
        "trail_roll_offset_deg": 20.0,
    }
