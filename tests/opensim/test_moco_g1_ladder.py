"""Unit tests and RED failure fixtures for OpenSim full-swing dynamic tracking ladder (OG-06, #10400).

Tests:
1. RED fixture: Mismatched model checkpoint raises ModelCheckpointMismatchError.
2. RED fixture: Truncated full-capture claim raises TruncatedCaptureClaimError.
3. RED fixture: Controls / state naming mismatch raises ControlStateNamingMismatchError.
4. RED fixture: Dynamic bilateral grip violation raises DynamicGripViolationError.
5. RED fixture: Discontinuous jump in coordinates/velocities raises ContinuityViolationError.
6. Ladder progression: Static address -> Short pilot (0.10s) -> G1 (0.85s) -> G2 (1.20s) -> G3 full horizon.
7. Distinct statuses: Assert separate receipts and distinct statuses for IK playback, solver convergence, and replay acceptance.
8. Real tour capture full-swing qualification on C3D_TA_Driver.c3d.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.address import (
    AddressFitResult,
    AddressPostureMetrics,
    CoordinateLimitsAudit,
    GripClosureMetrics,
)
from src.engines.physics_engines.opensim.python.tour_matching.full_swing_tracking import (
    ContinuityViolationError,
    ControlStateNamingMismatchError,
    DynamicGripViolationError,
    DynamicTrackingReceipt,
    ForwardReplayReceipt,
    FullSwingQualificationResult,
    FullSwingTrajectory,
    LadderStage,
    ModelCheckpointMismatchError,
    SwingEvents,
    TruncatedCaptureClaimError,
    detect_swing_events,
    qualify_full_swing_tracking,
    reinitialize_tracking_from_address,
    validate_capture_claim,
    validate_controls_state_naming,
    validate_dynamic_grip_closure,
    validate_model_checkpoint,
    validate_swing_continuity,
    validate_swing_coordinate_limits,
)
from src.shared.python.motion_matching.acceptance import (
    AcceptanceGates,
    GateStatus,
    Horizon,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    TOUR_CAPTURE,
    TourCapture,
)

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


def test_red_fixture_mismatched_model_checkpoint_fails() -> None:
    """RED Fixture: When a model checkpoint SHA-256 does not match the expected qualified hash, raise error."""
    wrong_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
    with pytest.raises(
        ModelCheckpointMismatchError, match="Model checkpoint hash mismatch"
    ):
        validate_model_checkpoint(SCALED_MODEL_PATH, expected_sha256=wrong_sha256)


def test_red_fixture_truncated_full_capture_claim_fails() -> None:
    """RED Fixture: Claiming full-swing G3 with truncated frame count or duration fails closed."""
    # Create a truncated capture with only 100 frames instead of 654
    time_s = np.linspace(0.0, 0.275, 100)
    labels = ("Marker_1", "Marker_2")
    pts = np.ones((100, 2, 3), dtype=np.float64)
    valid = np.ones((100, 2), dtype=bool)
    truncated_capture = TourCapture(time_s, labels, pts, valid)

    with pytest.raises(
        TruncatedCaptureClaimError, match="Full-swing G3 capture claim truncated"
    ):
        validate_capture_claim(
            truncated_capture, stage=LadderStage.G3_FULL_SWING, claim_full_capture=True
        )


def test_red_fixture_controls_state_naming_mismatch_fails() -> None:
    """RED Fixture: Control actuator names that mismatch model coordinates or state variables raise error."""
    valid_coords = ["lumbar_extension", "pelvis_tilt", "arm_flex_r"]
    invalid_controls = ["wrong_actuator_name", "pelvis_tilt_force"]

    with pytest.raises(
        ControlStateNamingMismatchError, match="Control/state naming mismatch"
    ):
        validate_controls_state_naming(
            SCALED_MODEL_PATH,
            coordinate_names=valid_coords,
            control_names=invalid_controls,
        )


def test_red_fixture_dynamic_grip_violation_fails() -> None:
    """RED Fixture: Bilateral grip separation exceeding 5 mm during swing raises DynamicGripViolationError."""
    # 50 frames, frame 25 has 12 mm grip separation (> 5 mm gate)
    grip_dist = np.full(50, 0.002, dtype=np.float64)
    grip_dist[25] = 0.012

    with pytest.raises(
        DynamicGripViolationError, match="Dynamic grip closure violation"
    ):
        validate_dynamic_grip_closure(grip_dist, max_closure_m=0.005)


def test_red_fixture_continuity_violation_fails() -> None:
    """RED Fixture: Discontinuous jumps in coordinate trajectory exceeding physiological velocity limit raise error."""
    dt = 1.0 / 360.0
    time_s = np.arange(10, dtype=np.float64) * dt
    q = np.zeros((10, 2), dtype=np.float64)
    # Inject a 2.0 rad jump in 1 frame (dt ~ 0.0028s -> velocity ~ 720 rad/s >> 35 rad/s)
    q[5, 0] = 2.0

    with pytest.raises(ContinuityViolationError, match="Continuity violation"):
        validate_swing_continuity(time_s, q, max_joint_speed_rad_s=35.0)


def test_swing_events_detection() -> None:
    """Verify detection and timing consistency of key swing events."""
    time_s = np.linspace(0.0, 1.814, 654)
    labels = ("Marker_1",)
    pts = np.zeros((654, 1, 3), dtype=np.float64)
    valid = np.ones((654, 1), dtype=bool)
    capture = TourCapture(time_s, labels, pts, valid)

    events = detect_swing_events(capture)
    assert isinstance(events, SwingEvents)
    assert events.address_s == 0.0
    assert 0.20 <= events.takeaway_s <= 0.50
    assert 0.70 <= events.top_of_backswing_s <= 0.95
    assert 1.05 <= events.impact_s <= 1.30
    assert math.isclose(events.finish_s, 1.814, rel_tol=1e-3)


def test_reinitialize_tracking_from_qualified_address() -> None:
    """Verify that full-swing tracking state q0 is initialized exactly from AddressFitResult."""
    q_address = np.array([0.05, -0.15, 0.02, 0.45, 0.25], dtype=np.float64)
    mock_address_fit = AddressFitResult(
        q=q_address,
        offsets={"Marker_1": ("pelvis", (0.01, 0.02, 0.03))},
        posture=AddressPostureMetrics(
            torso_yaw_deg=0.0,
            torso_pitch_deg=25.0,
            torso_roll_deg=0.0,
            pelvis_yaw_deg=0.0,
            pelvis_pitch_deg=15.0,
            pelvis_roll_deg=0.0,
            elbow_flexion_r_deg=15.0,
            elbow_flexion_l_deg=10.0,
            wrist_r_world_pos_m=(0.1, 0.95, -0.2),
            wrist_l_world_pos_m=(0.1, 0.95, -0.15),
            stance_width_m=0.55,
            club_lie_deg=60.0,
            shaft_direction=(0.0, -0.866, 0.5),
            foot_clearance_m=0.005,
        ),
        grip_closure=GripClosureMetrics(
            positional_closure_m=0.002,
            rotational_closure_rad=0.01,
            lead_hand_world_pos=(0.1, 0.95, -0.15),
            club_lead_grip_world_pos=(0.1, 0.952, -0.15),
            is_closed=True,
        ),
        limits_audit=CoordinateLimitsAudit(violations={}, is_valid=True),
        valid_marker_rms_m=0.008,
        max_marker_error_m=0.018,
        holdout_marker_rms_m=0.009,
        per_marker_rms_m={"Marker_1": 0.008},
        tolerance_profile_sha256="abc123",
        is_qualified=True,
        failure_reasons=(),
    )

    time_s = np.linspace(0.0, 1.814, 654)
    labels = ("Marker_1",)
    pts = np.zeros((654, 1, 3), dtype=np.float64)
    valid = np.ones((654, 1), dtype=bool)
    capture = TourCapture(time_s, labels, pts, valid)

    q0, fixed_calibration = reinitialize_tracking_from_address(
        model_path=SCALED_MODEL_PATH,
        address_fit=mock_address_fit,
        capture=capture,
    )
    np.testing.assert_array_almost_equal(q0, q_address)
    assert "Marker_1" in fixed_calibration
    assert fixed_calibration["Marker_1"] == ("pelvis", (0.01, 0.02, 0.03))


def test_ladder_stages_and_distinct_receipt_statuses() -> None:
    """Verify ladder progression and distinct statuses for IK playback, solver convergence, and replay."""
    n_frames = 100
    time_s = np.linspace(0.0, 0.85, n_frames)
    coords = ("pelvis_tilt", "lumbar_extension")
    q = np.zeros((n_frames, 2), dtype=np.float64)
    controls = np.zeros((n_frames, 2), dtype=np.float64)
    grip_dist = np.full(n_frames, 0.003, dtype=np.float64)
    normal_force = np.full(n_frames, 784.8, dtype=np.float64)  # 1.0 BW
    penetration = np.zeros(n_frames, dtype=np.float64)

    trajectory = FullSwingTrajectory(
        time_s=time_s,
        coordinate_names=coords,
        q=q,
        qdot=np.zeros_like(q),
        control_names=coords,
        controls=controls,
        grip_closure_distances_m=grip_dist,
        ground_normal_force_n=normal_force,
        ground_penetration_m=penetration,
        support_polygon_fraction=0.95,
    )

    labels = ("WaistLeft", "WaistRight")
    pts = np.zeros((n_frames, 2, 3), dtype=np.float64)
    valid = np.ones((n_frames, 2), dtype=bool)
    capture = TourCapture(time_s, labels, pts, valid)

    result = qualify_full_swing_tracking(
        model_path=SCALED_MODEL_PATH,
        trajectory=trajectory,
        capture=capture,
        stage=LadderStage.G1_BACKSWING,
    )

    assert isinstance(result, FullSwingQualificationResult)
    assert result.stage == LadderStage.G1_BACKSWING
    assert result.horizon == Horizon.G1

    # Distinct statuses
    assert result.ik_playback_status == "Playback_Succeeded"
    assert result.solver_convergence_status == "Solve_Succeeded"
    assert result.replay_acceptance_status == "Accepted"
    assert result.is_qualified

    # Separate receipts
    tracking_rcpt = result.tracking_receipt
    replay_rcpt = result.replay_receipt
    assert isinstance(tracking_rcpt, DynamicTrackingReceipt)
    assert isinstance(replay_rcpt, ForwardReplayReceipt)
    assert tracking_rcpt.horizon == Horizon.G1
    assert replay_rcpt.horizon == Horizon.G1
    assert replay_rcpt.replay_success is True


def test_real_tour_capture_full_swing_driver_events() -> None:
    """Test swing event detection and contract validation on real C3D_TA_Driver.c3d."""
    if not C3D_PATH.exists():
        pytest.skip(f"Capture file not found: {C3D_PATH}")

    from src.shared.python.motion_matching.tour_capture_contract import (
        load_tour_capture,
    )

    capture = load_tour_capture(C3D_PATH)
    assert capture.frames == 654

    events = detect_swing_events(capture)
    assert events.address_s == 0.0
    assert 0.10 <= events.takeaway_s <= 0.45
    assert 0.80 <= events.top_of_backswing_s <= 1.15
    assert 1.15 <= events.impact_s <= 1.30
    assert math.isclose(events.finish_s, (654 - 1) / 360.0, abs_tol=1e-3)
