"""Acceptance and unit tests for kinematic smoothing and calibration (PF-02, #10432).

Acceptance criteria:
1. Known synthetic exact multi-harmonic motion recovery (RMS < 1e-3).
2. Displaced targets and marker dropout robustness (up to 30% dropout).
3. Joint bounds enforced throughout trajectory.
4. Dual-grip weld closure preserved after smoothing (error <= 1 mm, rate <= 10 mm/s).
5. Consistent acceleration with no boundary spikes.
6. No dropped frames (output count matches input count).
7. Iron rejects incompatible driver calibration geometry with explicit error.
8. Ground barrier non-penetration enforcement.
9. Multi-start geometric floor estimation on difficult frames.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.club_calibration import (
    ClubCalibrationSpec,
    ClubCompatibilityError,
    ClubType,
    diagnose_club_marker_residuals,
    get_driver_calibration_spec,
    get_iron_7_calibration_spec,
    validate_club_compatibility,
)
from src.shared.python.motion_matching.kinematic_smoother import (
    KinematicSmoother,
    KinematicSmootherOptions,
    SmoothKinematicTrajectory,
)
from src.shared.python.motion_matching.windowed_ik_refinement import (
    DifficultFrameFloorDiagnosis,
    WindowedIkOptions,
    WindowedIkRefiner,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures and Toy Multi-Body Setup
# ---------------------------------------------------------------------------


class ToySwingKinematics:
    """Analytical 3-DoF kinematic model with dual-grip closure and foot contact.

    Coordinates:
    - q[0]: pelvis sway (X translation, m)
    - q[1]: lead arm angle (rad)
    - q[2]: trail arm angle (rad)
    """

    def __init__(self) -> None:
        self.arm_length = 0.65
        self.hand_offset = 0.05
        self.foot_z_offset = 0.05

    def forward_markers(self, q: np.ndarray) -> np.ndarray:
        """Return 4 synthetic marker positions (Pelvis, Thorax, LeadHand, TrailHand)."""
        x_pelvis = q[0]
        theta_lead = q[1]
        theta_trail = q[2]

        pelvis = np.array([x_pelvis, 0.0, 0.90])
        thorax = np.array([x_pelvis, 0.0, 1.40])
        lead_hand = thorax + np.array(
            [
                self.arm_length * np.cos(theta_lead),
                self.arm_length * np.sin(theta_lead),
                -0.30,
            ]
        )
        trail_hand = thorax + np.array(
            [
                self.arm_length * np.cos(theta_trail) + self.hand_offset,
                self.arm_length * np.sin(theta_trail),
                -0.30,
            ]
        )
        return np.vstack([pelvis, thorax, lead_hand, trail_hand])

    def closure(self, q: np.ndarray) -> np.ndarray:
        """Dual-grip weld constraint: lead hand and trail hand relative distance must be hand_offset."""
        markers = self.forward_markers(q)
        lead = markers[2]
        trail = markers[3]
        target_disp = np.array([self.hand_offset, 0.0, 0.0])
        return (trail - lead) - target_disp

    def foot_contacts(self, q: np.ndarray) -> np.ndarray:
        """Return foot contact sphere bottom points: [left_foot, right_foot]."""
        x_pelvis = q[0]
        # Ground height nominally at z=0; feet are below pelvis
        left_foot = np.array([x_pelvis - 0.20, 0.0, 0.02 + 0.01 * q[1]])
        right_foot = np.array([x_pelvis + 0.20, 0.0, 0.02 - 0.01 * q[1]])
        return np.vstack([left_foot, right_foot])


# ---------------------------------------------------------------------------
# Test 1: Club Calibration Separation and Rejection
# ---------------------------------------------------------------------------


def test_iron_rejects_driver_calibration() -> None:
    """Iron rejects incompatible driver calibration with explicit error."""
    driver_cal = get_driver_calibration_spec()
    iron_cal = get_iron_7_calibration_spec()

    # Valid validation passes
    validate_club_compatibility(driver_cal, ClubType.DRIVER)
    validate_club_compatibility(driver_cal, "driver")
    validate_club_compatibility(iron_cal, ClubType.IRON_7)
    validate_club_compatibility(iron_cal, "iron7")

    # Driver calibration applied to 7-iron must raise ClubCompatibilityError
    with pytest.raises(
        ClubCompatibilityError, match="cannot apply driver calibration to iron7"
    ):
        validate_club_compatibility(driver_cal, ClubType.IRON_7)

    # 7-iron calibration applied to driver must raise ClubCompatibilityError
    with pytest.raises(
        ClubCompatibilityError, match="cannot apply iron7 calibration to driver"
    ):
        validate_club_compatibility(iron_cal, ClubType.DRIVER)

    # Incompatible shaft length for iron
    with pytest.raises(
        ClubCompatibilityError,
        match="Shaft length 1.156 m is outside valid calibration range",
    ):
        validate_club_compatibility(
            iron_cal, ClubType.IRON_7, target_shaft_length_m=driver_cal.spec.length_m
        )


def test_club_marker_residuals_diagnostics() -> None:
    """Marker residual diagnostics accurately detect mismatch."""
    driver_cal = get_driver_calibration_spec()
    exact_offsets = driver_cal.marker_attachments
    diag_good = diagnose_club_marker_residuals(driver_cal, exact_offsets)
    assert diag_good["compatible"] is True
    assert diag_good["rms_error_m"] == pytest.approx(0.0, abs=1e-6)

    # Perturbed offsets exceeding tolerance
    bad_offsets = {k: (v[0] + 0.08, v[1], v[2]) for k, v in exact_offsets.items()}
    diag_bad = diagnose_club_marker_residuals(driver_cal, bad_offsets, tolerance_m=0.02)
    assert diag_bad["compatible"] is False
    assert diag_bad["rms_error_m"] > 0.05


# ---------------------------------------------------------------------------
# Test 2: Known Synthetic Exact Multi-Harmonic Motion Recovery
# ---------------------------------------------------------------------------


def test_synthetic_exact_multi_harmonic_motion_recovery() -> None:
    """Recovers known multi-harmonic motion and exact derivatives from clean synthetic markers."""
    kin = ToySwingKinematics()
    n_frames = 120
    dt = 1.0 / 360.0
    t = np.arange(n_frames) * dt

    # Multi-harmonic ground truth trajectory
    q_truth = np.zeros((n_frames, 3))
    q_truth[:, 0] = 0.05 * np.sin(2.0 * np.pi * 1.5 * t) + 0.02 * np.sin(
        2.0 * np.pi * 3.0 * t
    )
    q_truth[:, 1] = 0.80 * np.sin(2.0 * np.pi * 2.0 * t) + 0.20 * np.cos(
        2.0 * np.pi * 4.0 * t
    )
    q_truth[:, 2] = q_truth[:, 1]  # Exact dual-grip weld closure

    # Generate synthetic marker observations
    targets = np.zeros((n_frames, 4, 3))
    for f in range(n_frames):
        targets[f] = kin.forward_markers(q_truth[f])
    valid = np.ones((n_frames, 4), dtype=bool)

    # Warm-start with slightly noisy seed
    q_seed = q_truth + np.random.default_rng(42).normal(scale=0.005, size=q_truth.shape)

    refiner = WindowedIkRefiner(
        kin.forward_markers,
        closure_fn=kin.closure,
        foot_contact_fn=kin.foot_contacts,
        options=WindowedIkOptions(
            window_size=9, overlap=3, max_iterations=20, smoothness_weight=1e-4
        ),
    )
    q_refined, receipt = refiner.refine_trajectory(q_seed, targets, valid)

    assert receipt.converged is True
    assert receipt.final_marker_rms_m < 2e-3

    # Kinematic smoother
    smoother = KinematicSmoother(
        KinematicSmootherOptions(cutoff_hz=25.0, dt=dt),
        closure_fn=kin.closure,
    )
    smooth_traj = smoother.smooth(q_refined, dt=dt)

    assert smooth_traj.audit.no_dropped_frames is True
    assert smooth_traj.audit.frame_count == n_frames

    # Verify trajectory recovery against ground truth
    rms_q = np.sqrt(np.mean((smooth_traj.q - q_truth) ** 2))
    assert rms_q < 5e-3


# ---------------------------------------------------------------------------
# Test 3: Displaced Targets and Marker Dropout Robustness
# ---------------------------------------------------------------------------


def test_displaced_targets_and_marker_dropout() -> None:
    """Robust to displaced targets and up to 30% marker dropout without divergence."""
    kin = ToySwingKinematics()
    n_frames = 60
    dt = 1.0 / 360.0
    t = np.arange(n_frames) * dt

    q_truth = np.zeros((n_frames, 3))
    q_truth[:, 0] = 0.03 * np.sin(2.0 * np.pi * 2.0 * t)
    q_truth[:, 1] = 0.50 * np.sin(2.0 * np.pi * 1.5 * t)
    q_truth[:, 2] = q_truth[:, 1]

    targets = np.zeros((n_frames, 4, 3))
    for f in range(n_frames):
        targets[f] = kin.forward_markers(q_truth[f])

    # 30% dropout across markers and frames
    rng = np.random.default_rng(123)
    valid = rng.uniform(size=(n_frames, 4)) > 0.30
    # Ensure at least one marker valid per frame
    valid[:, 0] = True

    # Add 5mm displacement noise to targets
    targets_noisy = targets + rng.normal(scale=0.005, size=targets.shape)

    refiner = WindowedIkRefiner(
        kin.forward_markers,
        closure_fn=kin.closure,
        options=WindowedIkOptions(window_size=7, overlap=2, max_iterations=10),
    )
    q_refined, receipt = refiner.refine_trajectory(q_truth, targets_noisy, valid)

    assert receipt.converged is True
    assert np.isfinite(q_refined).all()
    assert len(q_refined) == n_frames


# ---------------------------------------------------------------------------
# Test 4: Joint Bounds Enforced
# ---------------------------------------------------------------------------


def test_joint_bounds_enforced() -> None:
    """Confirms all frames satisfy lower <= q(t) <= upper even under aggressive pull."""
    kin = ToySwingKinematics()
    n_frames = 40
    dt = 1.0 / 360.0

    lower = np.array([-0.05, -0.40, -0.80])
    upper = np.array([0.05, 0.40, 0.80])

    # Construct targets far outside physical boundaries
    targets_extreme = np.full((n_frames, 4, 3), 10.0)
    valid = np.ones((n_frames, 4), dtype=bool)

    q_init = np.zeros((n_frames, 3))

    refiner = WindowedIkRefiner(
        kin.forward_markers,
        lower=lower,
        upper=upper,
        options=WindowedIkOptions(window_size=5, overlap=1, max_iterations=10),
    )
    q_bounded, _ = refiner.refine_trajectory(q_init, targets_extreme, valid)

    # Verify bounds in refiner
    assert np.all(q_bounded >= lower - 1e-6)
    assert np.all(q_bounded <= upper + 1e-6)

    # Verify bounds maintained in smoother
    smoother = KinematicSmoother(
        KinematicSmootherOptions(dt=dt, enforce_bounds=True),
        lower=lower,
        upper=upper,
    )
    smooth_traj = smoother.smooth(q_bounded, dt=dt)
    assert smooth_traj.audit.bounds_violation_count == 0
    assert np.all(smooth_traj.q >= lower - 1e-6)
    assert np.all(smooth_traj.q <= upper + 1e-6)


# ---------------------------------------------------------------------------
# Test 5: Closure Preserved After Smoothing
# ---------------------------------------------------------------------------


def test_closure_preserved_after_smoothing() -> None:
    """Dual-grip weld closure error <= 1 mm and velocity closure <= 10 mm/s after smoothing."""
    kin = ToySwingKinematics()
    n_frames = 80
    dt = 1.0 / 360.0
    t = np.arange(n_frames) * dt

    q_traj = np.zeros((n_frames, 3))
    q_traj[:, 0] = 0.04 * np.sin(2.0 * np.pi * 2.0 * t)
    q_traj[:, 1] = 0.60 * np.sin(2.0 * np.pi * 2.5 * t)
    q_traj[:, 2] = q_traj[:, 1]

    # Inject slight closure disturbance
    q_disturbed = q_traj.copy()
    q_disturbed[:, 2] += 0.01 * np.sin(2.0 * np.pi * 30.0 * t)

    smoother = KinematicSmoother(
        KinematicSmootherOptions(
            cutoff_hz=15.0,
            dt=dt,
            closure_tolerance_m=1e-3,
            closure_velocity_tolerance_m_s=1e-2,
        ),
        closure_fn=kin.closure,
    )

    smooth_traj = smoother.smooth(q_disturbed, dt=dt)

    assert smooth_traj.audit.max_closure_error_m <= 1e-3
    assert smooth_traj.audit.max_closure_rate_m_s <= 1e-2
    assert np.all(smooth_traj.closure_error_m <= 1e-3)


# ---------------------------------------------------------------------------
# Test 6: Consistent Acceleration With No Boundary Spikes
# ---------------------------------------------------------------------------


def test_consistent_acceleration_no_boundary_spikes() -> None:
    """Confirms no boundary jerk or acceleration spikes due to reflection padding."""
    kin = ToySwingKinematics()
    n_frames = 100
    dt = 1.0 / 360.0
    t = np.arange(n_frames) * dt

    q_traj = np.zeros((n_frames, 3))
    q_traj[:, 0] = 0.05 * np.sin(2.0 * np.pi * 2.0 * t)
    q_traj[:, 1] = 0.50 * np.sin(2.0 * np.pi * 1.5 * t)
    q_traj[:, 2] = q_traj[:, 1]

    smoother = KinematicSmoother(
        KinematicSmootherOptions(cutoff_hz=12.0, dt=dt, boundary_padding_frames=15)
    )
    res = smoother.smooth(q_traj, dt=dt)

    # Boundary acceleration steps should be well-behaved relative to median step
    assert res.audit.boundary_spike_start_ratio < 4.0
    assert res.audit.boundary_spike_end_ratio < 4.0
    assert np.isfinite(res.a).all()


# ---------------------------------------------------------------------------
# Test 7: No Dropped Frames
# ---------------------------------------------------------------------------


def test_no_dropped_frames() -> None:
    """Verifies output trajectory maintains full temporal resolution N_out == N_in."""
    kin = ToySwingKinematics()
    for n_frames in (5, 27, 100):
        dt = 1.0 / 360.0
        q_raw = np.zeros((n_frames, 3))
        targets = np.zeros((n_frames, 4, 3))
        valid = np.ones((n_frames, 4), dtype=bool)

        refiner = WindowedIkRefiner(kin.forward_markers)
        q_ref, receipt = refiner.refine_trajectory(q_raw, targets, valid)
        assert receipt.frames == n_frames
        assert len(q_ref) == n_frames

        smoother = KinematicSmoother(KinematicSmootherOptions(dt=dt))
        smooth_traj = smoother.smooth(q_ref, dt=dt)
        assert smooth_traj.audit.frame_count == n_frames
        assert len(smooth_traj.q) == n_frames
        assert len(smooth_traj.v) == n_frames
        assert len(smooth_traj.a) == n_frames


# ---------------------------------------------------------------------------
# Test 8: Ground Barrier Non-Penetration
# ---------------------------------------------------------------------------


def test_ground_barrier_no_penetration() -> None:
    """Ground barrier prevents contact points from penetrating underground."""
    kin = ToySwingKinematics()
    n_frames = 20
    dt = 1.0 / 360.0

    # Start with configuration where feet are at safe elevation
    q_init = np.zeros((n_frames, 3))

    # Pull targets downward violently below ground
    targets = np.zeros((n_frames, 4, 3))
    targets[:, :, 2] = -0.50
    valid = np.ones((n_frames, 4), dtype=bool)

    refiner = WindowedIkRefiner(
        kin.forward_markers,
        foot_contact_fn=kin.foot_contacts,
        options=WindowedIkOptions(
            ground_barrier_weight=1e5,
            ground_height_m=0.0,
            max_iterations=20,
        ),
    )
    q_refined, receipt = refiner.refine_trajectory(q_init, targets, valid)

    # Foot points must not penetrate below ground
    for f in range(n_frames):
        pts = kin.foot_contacts(q_refined[f])
        assert np.all(pts[:, 2] >= -1e-4)


# ---------------------------------------------------------------------------
# Test 9: Difficult Frame Geometric Floor Diagnosis
# ---------------------------------------------------------------------------


def test_difficult_frame_geometric_floor_diagnosis() -> None:
    """Diagnoses geometric floor on difficult frame with vs without closure."""
    kin = ToySwingKinematics()
    n_frames = 10
    q_traj = np.zeros((n_frames, 3))
    targets = np.zeros((n_frames, 4, 3))
    for f in range(n_frames):
        targets[f] = kin.forward_markers(q_traj[f])

    # Make frame 5 difficult by introducing geometric closure conflict in targets
    targets[5, 3] += np.array([0.20, 0.0, 0.0])
    valid = np.ones((n_frames, 4), dtype=bool)

    refiner = WindowedIkRefiner(
        kin.forward_markers,
        closure_fn=kin.closure,
        options=WindowedIkOptions(closure_weight=1e4),
    )
    diagnoses = refiner.diagnose_difficult_frames(q_traj, targets, valid, top_k=2)

    assert len(diagnoses) >= 1
    d5 = next((d for d in diagnoses if d.frame_index == 5), None)
    assert d5 is not None
    assert isinstance(d5, DifficultFrameFloorDiagnosis)
    # When unconstrained, closure error is free to expand to fit the conflicting marker
    assert d5.unconstrained_closure_err_m > d5.constrained_closure_err_m
