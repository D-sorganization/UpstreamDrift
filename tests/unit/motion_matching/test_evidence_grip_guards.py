"""Evidence-based grip guards (SHOW-A21, #14797).

Derives and validates two grip guards from golf-instruction references (Hogan 1957,
Leadbetter 1990, Cochran & Stobbs 1968), human forearm anatomy (Kapandji 1982,
AAOS 1965), prehensile opposition geometry (Napier 1956), and measured capture data:
1. GripSpec / GripPose trail V-offset bound: ``TRAIL_V_OFFSET_RANGE_DEG = (-75.0, 75.0)``
2. Lead-vs-trail dorsal separation guard: ``MIN_DORSAL_SEPARATION_DEG = 90.0``
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.model_appearance.grip_pose import (
    GripPose,
    TRAIL_V_OFFSET_RANGE_DEG,
)
from src.shared.python.motion_matching.grip_roll import (
    MIN_DORSAL_SEPARATION_DEG,
    GripPoseState,
    dorsal_separation_deg,
    validate_dorsal_separation,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Constant definitions and citation checks
# ---------------------------------------------------------------------------


def test_evidence_constants_are_defined_with_expected_values() -> None:
    """Guards are named constants with cited evidence ranges."""
    assert TRAIL_V_OFFSET_RANGE_DEG == (-75.0, 75.0)
    assert MIN_DORSAL_SEPARATION_DEG == 90.0


# ---------------------------------------------------------------------------
# Parametrized test table: legitimate grips (weak, neutral, strong, capture, b15, c)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,lead_alpha,trail_alpha,expected_v_offset,expected_dorsal_sep",
    [
        ("neutral_matched_v", 0.0, 0.0, 0.0, 180.0),
        ("weak_preset", -15.0, -15.0, 0.0, 180.0),
        ("strong_preset", 15.0, 15.0, 0.0, 180.0),
        ("ten_finger_preset", 5.0, 5.0, 0.0, 180.0),
        ("tour_driver_capture_fit", -25.02, 16.31, 41.33, 140.0),
        ("main_iron7_anthro", -13.91, 13.82, 27.73, 124.08),
        ("option_c_parallel_vs", -10.73, -10.73, 0.0, 180.0),
        ("option_b15_trail_alpha_15", -10.73, -74.84, -64.11, 115.89),
    ],
)
def test_legitimate_instructional_and_measured_grips_pass_guards(
    label: str,
    lead_alpha: float,
    trail_alpha: float,
    expected_v_offset: float,
    expected_dorsal_sep: float,
) -> None:
    """Weak, neutral, strong, tour capture and valid G2 options pass both guards."""
    v_offset = trail_alpha - lead_alpha
    assert v_offset == pytest.approx(expected_v_offset, abs=0.5)
    assert TRAIL_V_OFFSET_RANGE_DEG[0] <= v_offset <= TRAIL_V_OFFSET_RANGE_DEG[1]

    # Verify GripPose admits this trail V-offset
    pose = GripPose(trail_v_offset_deg=v_offset)
    assert pose.trail_v_offset_deg == pytest.approx(v_offset, abs=1e-6)

    # Dorsal separation meets or exceeds the 90 deg prehensile opposition bound
    assert expected_dorsal_sep >= MIN_DORSAL_SEPARATION_DEG
    rad = math.radians(expected_dorsal_sep)
    d_lead = np.array([1.0, 0.0, 0.0])
    d_trail = np.array([math.cos(rad), 0.0, math.sin(rad)])
    sep = dorsal_separation_deg(d_lead, d_trail)
    assert sep == pytest.approx(expected_dorsal_sep, abs=0.5)
    assert validate_dorsal_separation(d_lead, d_trail) == pytest.approx(sep, abs=1e-6)


# ---------------------------------------------------------------------------
# Parametrized test table: rejected / impossible poses (a, parallel backs, palms-out)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,v_offset,dorsal_sep,fails_v_offset,fails_dorsal_sep",
    [
        ("option_a_pure_mirror", -98.23, 81.77, True, True),
        ("copy_grip_main_bug_parallel_backs", 0.0, 0.0, False, True),
        ("hyper_supinated_palms_out", 85.0, 45.0, True, True),
        ("extreme_divergence_100_deg", 100.0, 80.0, True, True),
        ("orthogonal_boundary_fail", 0.0, 89.9, False, True),
    ],
)
def test_impossible_and_un_anatomical_poses_fail_guards(
    label: str,
    v_offset: float,
    dorsal_sep: float,
    fails_v_offset: bool,
    fails_dorsal_sep: bool,
) -> None:
    """Option (a), parallel backs (0 deg) and non-opposing palms (<90 deg) fail."""
    if fails_v_offset:
        assert not (
            TRAIL_V_OFFSET_RANGE_DEG[0] <= v_offset <= TRAIL_V_OFFSET_RANGE_DEG[1]
        )
        with pytest.raises(PreconditionError):
            GripPose(trail_v_offset_deg=v_offset)
    else:
        assert TRAIL_V_OFFSET_RANGE_DEG[0] <= v_offset <= TRAIL_V_OFFSET_RANGE_DEG[1]

    rad = math.radians(dorsal_sep)
    d_lead = np.array([1.0, 0.0, 0.0])
    d_trail = np.array([math.cos(rad), 0.0, math.sin(rad)])
    sep = dorsal_separation_deg(d_lead, d_trail)
    assert sep == pytest.approx(dorsal_sep, abs=0.1)

    if fails_dorsal_sep:
        assert sep < MIN_DORSAL_SEPARATION_DEG
        with pytest.raises(PreconditionError):
            validate_dorsal_separation(d_lead, d_trail)
    else:
        assert sep >= MIN_DORSAL_SEPARATION_DEG
        assert validate_dorsal_separation(d_lead, d_trail) == pytest.approx(
            sep, abs=1e-6
        )


# ---------------------------------------------------------------------------
# Specific evaluation of G2 options: (a), (b15), and (c)
# ---------------------------------------------------------------------------


def test_exact_recheck_of_g2_trail_options_against_evidence_guards() -> None:
    """Records exact numbers for (a), (b15), (c) under evidence-based guards (#14797).

    Option (c) parallel V's:
      - trail V-offset: 0.0 deg (inside [-75.0, 75.0], PASS)
      - dorsal separation: 180.00 deg (>= 90.0 deg, PASS)
    Option (b15) trail alpha 15:
      - trail V-offset: -64.11 deg (inside [-75.0, 75.0], PASS)
      - dorsal separation: 115.89 deg (>= 90.0 deg, PASS)
    Option (a) pure mirror:
      - trail V-offset: -98.23 deg (outside [-75.0, 75.0], FAIL)
      - dorsal separation: 81.77 deg (< 90.0 deg, FAIL)
    """
    # (c) parallel Vs:
    c_v_offset = 0.0
    c_dorsal_sep = 180.00
    assert TRAIL_V_OFFSET_RANGE_DEG[0] <= c_v_offset <= TRAIL_V_OFFSET_RANGE_DEG[1]
    assert c_dorsal_sep >= MIN_DORSAL_SEPARATION_DEG

    # (b15) trail alpha 15:
    b15_v_offset = -64.11
    b15_dorsal_sep = 115.89
    assert TRAIL_V_OFFSET_RANGE_DEG[0] <= b15_v_offset <= TRAIL_V_OFFSET_RANGE_DEG[1]
    assert b15_dorsal_sep >= MIN_DORSAL_SEPARATION_DEG

    # (a) pure mirror:
    a_v_offset = -98.23
    a_dorsal_sep = 81.77
    assert not (
        TRAIL_V_OFFSET_RANGE_DEG[0] <= a_v_offset <= TRAIL_V_OFFSET_RANGE_DEG[1]
    )
    assert a_dorsal_sep < MIN_DORSAL_SEPARATION_DEG


# ---------------------------------------------------------------------------
# GripPoseState property integration
# ---------------------------------------------------------------------------


def test_grip_pose_state_dorsal_separation_property_and_validation() -> None:
    """GripPoseState provides .dorsal_separation_deg and .validate_dorsal_separation()."""
    rot = np.eye(3)
    state_opposed = GripPoseState(
        club_rotation=rot,
        lead_dorsal_club=np.array([1.0, 0.0, 0.0]),
        trail_dorsal_club=np.array([-1.0, 0.0, 0.0]),
    )
    assert state_opposed.dorsal_separation_deg == pytest.approx(180.0, abs=1e-6)
    assert state_opposed.validate_dorsal_separation() == pytest.approx(180.0, abs=1e-6)

    state_parallel = GripPoseState(
        club_rotation=rot,
        lead_dorsal_club=np.array([1.0, 0.0, 0.0]),
        trail_dorsal_club=np.array([1.0, 0.0, 0.0]),
    )
    assert state_parallel.dorsal_separation_deg == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(PreconditionError):
        state_parallel.validate_dorsal_separation()
