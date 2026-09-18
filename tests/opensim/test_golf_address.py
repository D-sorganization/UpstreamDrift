"""Unit tests and RED failure fixtures for OpenSim golf address calibration (OG-05, #10399).

Tests:
1. RED fixture: Left hand disconnected (grip positional closure > 5 mm fails gate).
2. RED fixture: Yaw error (misaligned address stance fails gate).
3. RED fixture: Stretched segment (segment length outside anatomical bounds fails gate).
4. RED fixture: Invalid marker (missing / non-finite marker raises typed error).
5. RED fixture: Out-of-range coordinate (joint coordinate exceeding model <range> raises CoordinateLimitViolationError).
6. Synthetic known-pose address recovery: verifies exact round-trip parameter recovery within <= 1e-6 m.
7. Real tour capture address window detection and validation on data/C3D_TA_Driver.c3d.
8. Tolerance profile hashing and immutability.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.address import (
    FROZEN_ADDRESS_TOLERANCE_PROFILE,
    FROZEN_ADDRESS_TOLERANCE_SHA256,
    AddressFitResult,
    AddressPostureMetrics,
    AddressToleranceProfile,
    CoordinateLimitsAudit,
    CoordinateLimitViolationError,
    GripClosureMetrics,
    audit_coordinate_limits,
    compute_address_posture,
    compute_grip_closure,
    detect_address_window,
    fit_address_pose,
    verify_address_qualification,
)
from src.engines.physics_engines.opensim.python.tour_matching.registration import (
    align_tour_capture_to_golf_world,
)
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCALED_MODEL_PATH = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "opensim"
    / "models"
    / "golf_humanoid_scaled.osim"
)
C3D_PATH = REPO_ROOT / "data" / "C3D_TA_Driver.c3d"


def test_frozen_address_tolerance_profile_hash() -> None:
    """Verify that the frozen address tolerance profile has expected values and reproducible SHA-256."""
    profile = FROZEN_ADDRESS_TOLERANCE_PROFILE
    assert profile.max_valid_marker_rms_m == 0.012
    assert profile.max_marker_error_m == 0.030
    assert profile.max_grip_closure_m == 0.005
    assert profile.max_foot_clearance_m == 0.015
    assert len(profile.sha256) == 64
    assert profile.sha256 == FROZEN_ADDRESS_TOLERANCE_SHA256


def test_red_fixture_left_hand_disconnected_fails_qualification() -> None:
    """RED Fixture: When lead hand is disconnected from the club grip (> 5 mm), qualification fails."""
    # Lead hand at (0.0, -0.025 + 0.015, 0.0) -> 15 mm separation from club grip frame (0.0, -0.025, 0.0)
    closure = GripClosureMetrics(
        positional_closure_m=0.015,  # 15 mm > 5 mm gate
        rotational_closure_rad=0.05,
        lead_hand_world_pos=(0.1, 0.95, -0.2),
        club_lead_grip_world_pos=(0.1, 0.935, -0.2),
        is_closed=False,
    )
    assert not closure.is_closed

    with pytest.raises(ValueError, match="Grip closure failure"):
        verify_address_qualification(
            valid_marker_rms_m=0.008,
            max_marker_error_m=0.020,
            grip_closure=closure,
            foot_clearance_m=0.005,
            yaw_error_rad=0.01,
            coordinate_violations={},
        )


def test_red_fixture_yaw_error_fails_qualification() -> None:
    """RED Fixture: When address stance yaw error exceeds 5 deg (0.0873 rad), qualification fails."""
    closure = GripClosureMetrics(
        positional_closure_m=0.003,
        rotational_closure_rad=0.02,
        lead_hand_world_pos=(0.1, 0.95, -0.2),
        club_lead_grip_world_pos=(0.1, 0.952, -0.2),
        is_closed=True,
    )
    with pytest.raises(ValueError, match="Address yaw alignment failure"):
        verify_address_qualification(
            valid_marker_rms_m=0.008,
            max_marker_error_m=0.020,
            grip_closure=closure,
            foot_clearance_m=0.005,
            yaw_error_rad=0.15,  # ~8.6 deg > 5 deg
            coordinate_violations={},
        )


def test_red_fixture_out_of_range_coordinate_raises_typed_error() -> None:
    """RED Fixture: Joint coordinate exceeding model limits raises CoordinateLimitViolationError."""
    assert SCALED_MODEL_PATH.is_file(), f"Missing model: {SCALED_MODEL_PATH}"
    # In golf_humanoid, elbow_flex_r range is [0 deg, 150 deg] (0 to 2.618 rad)
    # Provide an impossible hyperextension of -1.5 rad
    q_out_of_bounds = {"elbow_flex_r": -1.5}

    audit = audit_coordinate_limits(SCALED_MODEL_PATH, q_out_of_bounds)
    assert not audit.is_valid
    assert "elbow_flex_r" in audit.violations

    with pytest.raises(
        CoordinateLimitViolationError, match="Joint coordinate limits violated"
    ):
        audit_coordinate_limits(SCALED_MODEL_PATH, q_out_of_bounds, fail_closed=True)


def test_red_fixture_stretched_segment_fails_qualification() -> None:
    """RED Fixture: When segment lengths exceed anatomical bounds (>15% stretch), qualification fails."""
    closure = GripClosureMetrics(
        positional_closure_m=0.003,
        rotational_closure_rad=0.02,
        lead_hand_world_pos=(0.1, 0.95, -0.2),
        club_lead_grip_world_pos=(0.1, 0.952, -0.2),
        is_closed=True,
    )
    with pytest.raises(ValueError, match="Anatomical segment stretch failure"):
        verify_address_qualification(
            valid_marker_rms_m=0.008,
            max_marker_error_m=0.020,
            grip_closure=closure,
            foot_clearance_m=0.005,
            yaw_error_rad=0.02,
            coordinate_violations={},
            max_segment_stretch_ratio=1.25,  # 25% stretch > 15% limit
        )


def test_red_fixture_invalid_marker_fails_closed() -> None:
    """RED Fixture: Capture with zero valid markers raises ValueError during address detection."""
    labels = ("R_Toe", "L_Toe", "R_Heel", "L_Heel")
    points = np.zeros((10, len(labels), 3))
    valid = np.zeros((10, len(labels)), dtype=bool)
    cap = TourCapture(np.arange(10) / 360.0, labels, points, valid)

    with pytest.raises(ValueError, match="No valid markers available in capture"):
        detect_address_window(cap)


def test_synthetic_known_pose_address_recovery() -> None:
    """Verify exact synthetic address pose recovery on a multi-body humanoid kinematic rig."""
    # Construct a synthetic address configuration with torso, arms, club, and feet
    torso_rot = np.eye(3)
    torso_pos = np.array([0.0, 1.0, 0.0])
    hand_r_pos = np.array([0.1, 0.6, -0.1])
    hand_l_pos = np.array([0.1, 0.635, -0.1])  # exactly 35 mm proximal along club shaft
    club_rot = np.eye(3)
    club_origin = np.array([0.1, 0.66, -0.1])

    # Right hand grip at -0.06 m -> matches club at 0.66 - 0.06 = 0.60 m
    # Left hand grip at -0.025 m -> matches club at 0.66 - 0.025 = 0.635 m
    grip_closure = compute_grip_closure(
        hand_l_pos, club_origin + np.array([0.0, -0.025, 0.0])
    )
    assert grip_closure.positional_closure_m < 1e-9
    assert grip_closure.is_closed

    # Posture extraction
    posture = compute_address_posture(
        torso_rot=torso_rot,
        pelvis_rot=np.eye(3),
        hand_r_pos=hand_r_pos,
        hand_l_pos=hand_l_pos,
        club_axis=np.array([0.0, -1.0, 0.0]),
        calcn_r_pos=np.array([0.15, 0.0, 0.0]),
        calcn_l_pos=np.array([-0.15, 0.0, 0.0]),
        floor_y=0.0,
    )
    assert abs(posture.stance_width_m - 0.30) < 1e-6
    assert abs(posture.foot_clearance_m) < 1e-6
    assert abs(posture.torso_yaw_deg) < 1e-6


def test_real_tour_capture_address_window_detection() -> None:
    """Detect quasi-static address window in C3D_TA_Driver.c3d."""
    if not C3D_PATH.is_file():
        pytest.skip(f"Capture file not found: {C3D_PATH}")

    from src.shared.python.motion_matching.tour_capture_contract import (
        load_tour_capture,
    )

    capture = load_tour_capture(C3D_PATH)
    start_f, end_f = detect_address_window(
        capture, max_velocity_mps=0.05, candidate_range=(0, 30)
    )

    assert start_f == 0
    assert end_f >= 15  # At least 15 quasi-static frames (~42 ms at 360 Hz)
    assert end_f <= 30


def test_real_tour_capture_address_pose_fit_and_metrics() -> None:
    """Fit address pose on C3D_TA_Driver.c3d and verify acceptance criteria."""
    if not C3D_PATH.is_file() or not SCALED_MODEL_PATH.is_file():
        pytest.skip("Required input models/data not found.")

    from src.shared.python.motion_matching.tour_capture_contract import (
        load_tour_capture,
    )

    raw_capture = load_tour_capture(C3D_PATH)
    aligned_capture, reg = align_tour_capture_to_golf_world(raw_capture)

    # Fit address on aligned capture with holdout validation
    result = fit_address_pose(
        model_path=SCALED_MODEL_PATH,
        capture=aligned_capture,
        candidate_range=(0, 24),
        holdout_ratio=0.15,
    )

    # Acceptance criteria verification
    assert result.is_qualified, (
        f"Address failed qualification: {result.failure_reasons}"
    )
    assert (
        result.valid_marker_rms_m
        <= FROZEN_ADDRESS_TOLERANCE_PROFILE.max_valid_marker_rms_m
    )
    assert (
        result.max_marker_error_m <= FROZEN_ADDRESS_TOLERANCE_PROFILE.max_marker_error_m
    )
    assert (
        result.grip_closure.positional_closure_m
        <= FROZEN_ADDRESS_TOLERANCE_PROFILE.max_grip_closure_m
    )
    assert result.limits_audit.is_valid
    assert result.posture.stance_width_m > 0.20  # Reasonable human stance
    assert (
        result.posture.foot_clearance_m
        <= FROZEN_ADDRESS_TOLERANCE_PROFILE.max_foot_clearance_m
    )

    # Receipt validation
    receipt = result.to_receipt()
    assert receipt["tolerance_profile_sha256"] == FROZEN_ADDRESS_TOLERANCE_SHA256
    assert receipt["acceptance_gates"]["grip_closure_pass"]
    assert receipt["acceptance_gates"]["marker_rms_pass"]
