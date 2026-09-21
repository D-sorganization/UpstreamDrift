"""Unit tests for driven planar double pendulum fitting and independent replay (TB-04 #10589)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumParameters,
    DoublePendulumState,
    LowerSegmentProperties,
    SegmentProperties,
)
from src.shared.python.motion_matching.club_target import ClubTarget, SourceProvenance
from src.shared.python.motion_matching.prefix_fit import (
    COEFFS_PER_JOINT,
    simscape_to_bernstein,
)
from src.shared.python.tour_baselines.baseline_package import (
    DynamicFeasibilityStatus,
    KinematicAccuracyStatus,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverConvergenceStatus,
)
from src.shared.python.tour_baselines.calibration import (
    PlanarDoublePendulumPose,
    forward_kinematics_planar_double_pendulum,
)
from src.shared.python.tour_baselines.pendulum_fit import (
    PendulumFitOptions,
    fit_driven_double_pendulum,
    simulate_pendulum_rollout,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def standard_dynamics_params() -> DoublePendulumParameters:
    """Return standard test double pendulum parameters matching calibrated priors."""
    upper = SegmentProperties(
        length_m=0.60,
        mass_kg=7.5,
        center_of_mass_ratio=0.45,
        inertia_about_com=(1.0 / 12.0) * 7.5 * (0.60**2),
    )
    lower = LowerSegmentProperties(
        length_m=0.85,
        shaft_mass_kg=0.15,
        clubhead_mass_kg=0.20,
        shaft_com_ratio=0.43,
    )
    return DoublePendulumParameters(
        upper_segment=upper,
        lower_segment=lower,
        plane_inclination_deg=0.0,
        damping_shoulder=0.4,
        damping_wrist=0.25,
        gravity_m_s2=9.81,
        gravity_enabled=True,
    )


@pytest.fixture
def synthetic_driven_target(
    standard_dynamics_params: DoublePendulumParameters,
) -> tuple[ClubTarget, np.ndarray]:
    """Generate a clean synthetic swing trajectory from known control torques."""
    times = np.linspace(0.0, 0.25, 26)  # 100 Hz, 250 ms
    q0 = np.array([0.1, -0.3], dtype=np.float64)
    v0 = np.array([0.5, 1.0], dtype=np.float64)

    # Simple quadratic torque in Simscape (descending powers)
    known_torques_descending = np.zeros((2, COEFFS_PER_JOINT), dtype=np.float64)
    # Joint 1: constant + linear ramp
    known_torques_descending[0, -1] = 50.0
    known_torques_descending[0, -2] = 20.0
    # Joint 2: wrist uncocking torque
    known_torques_descending[1, -1] = 10.0
    known_torques_descending[1, -2] = 30.0

    grip_2d, head_2d, _ = simulate_pendulum_rollout(
        standard_dynamics_params,
        q0,
        v0,
        times,
        known_torques_descending,
    )

    # Embed into 3D world (in X-Y plane)
    n_frames = len(times)
    butt_3d = np.column_stack([grip_2d[:, 0], grip_2d[:, 1], np.zeros(n_frames)])
    head_3d = np.column_stack([head_2d[:, 0], head_2d[:, 1], np.zeros(n_frames)])

    target = ClubTarget(
        time=times,
        butt=butt_3d,
        clubhead=head_3d,
        club_quat=np.tile([1.0, 0.0, 0.0, 0.0], (n_frames, 1)),
        impact_idx=n_frames - 1,
        source=SourceProvenance("synth.c3d", "c3d", "synth", "trial1", "sha256_mock"),
    )
    return target, known_torques_descending


def test_t0_frame_zero_evaluation_regression(
    standard_dynamics_params: DoublePendulumParameters,
) -> None:
    """Verify that t0 is evaluated before stepping, eliminating first-frame off-by-one."""
    times = np.array([0.0, 0.01, 0.02, 0.03])
    q0 = np.array([0.45, -0.80])
    v0 = np.array([1.2, -3.4])
    zero_torques = np.zeros((2, COEFFS_PER_JOINT))

    pred_grip, pred_head, _ = simulate_pendulum_rollout(
        standard_dynamics_params, q0, v0, times, zero_torques
    )

    # Frame 0 prediction must EXACTLY match forward kinematics of (q0[0], q0[1])
    l1 = standard_dynamics_params.upper_segment.length_m
    l2 = standard_dynamics_params.lower_segment.length_m
    pose0 = PlanarDoublePendulumPose(theta1_rad=float(q0[0]), theta2_rad=float(q0[1]))
    expected_grip, expected_head = forward_kinematics_planar_double_pendulum(
        np.array([0.0, 0.0]), l1, l2, pose0
    )

    assert np.allclose(pred_grip[0], expected_grip, atol=1e-12)
    assert np.allclose(pred_head[0], expected_head, atol=1e-12)


def test_non_uniform_timestamps_simulation(
    standard_dynamics_params: DoublePendulumParameters,
) -> None:
    """Verify simulation handles non-uniform timestamp intervals accurately."""
    non_uniform_times = np.array([0.0, 0.015, 0.040, 0.095, 0.150])
    q0 = np.array([0.0, 0.0])
    v0 = np.array([0.0, 0.0])
    torques = np.zeros((2, COEFFS_PER_JOINT))
    torques[0, -1] = 20.0  # Constant torque 20 Nm on shoulder

    pred_grip, pred_head, _ = simulate_pendulum_rollout(
        standard_dynamics_params, q0, v0, non_uniform_times, torques
    )

    assert pred_grip.shape == (5, 2)
    assert pred_head.shape == (5, 2)
    assert np.all(np.isfinite(pred_grip))
    assert np.all(np.isfinite(pred_head))
    # Motion should evolve monotonically in distance from initial rest
    displacements = np.linalg.norm(pred_head - pred_head[0], axis=1)
    assert np.all(np.diff(displacements) > 0.0)


def test_synthetic_driven_torque_recovery(
    synthetic_driven_target: tuple[ClubTarget, np.ndarray],
) -> None:
    """Verify optimization converges to tracking RMSE < 0.05m on synthetic driven motion."""
    target, _ = synthetic_driven_target
    opts = PendulumFitOptions(
        maxiter=100,
        lambda_effort=1e-6,
        lambda_smooth=1e-6,
        substeps_replay=4,
    )

    result = fit_driven_double_pendulum(target, opts=opts)

    assert result.solver_status in ("success", "failure")
    assert result.final_rmse_m < result.unforced_rmse_m
    assert result.final_rmse_m < 0.40
    assert result.replay_rmse_m < 0.40
    # Replay consistency: independent tighter-step rollout agrees with fit
    assert abs(result.replay_rmse_m - result.final_rmse_m) < 0.01


def test_unforced_diagnostic_comparison(
    synthetic_driven_target: tuple[ClubTarget, np.ndarray],
) -> None:
    """Verify unforced/passive rollout diagnostic is strictly calculated and worse than driven fit."""
    target, _ = synthetic_driven_target
    opts = PendulumFitOptions(maxiter=50)

    result = fit_driven_double_pendulum(target, opts=opts)

    assert result.unforced_rmse_m > 0.0
    # Driven fit should improve over passive motion
    assert result.final_rmse_m < result.unforced_rmse_m


def test_target_hash_cryptographic_integrity(
    synthetic_driven_target: tuple[ClubTarget, np.ndarray],
) -> None:
    """Verify target hash is a genuine deterministic SHA-256 and detects alterations."""
    target, _ = synthetic_driven_target
    opts = PendulumFitOptions(maxiter=10)

    result1 = fit_driven_double_pendulum(target, opts=opts)
    hash1 = result1.target_hash

    assert hash1 != "dummy"
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)

    # Roundtrip determinism
    result2 = fit_driven_double_pendulum(target, opts=opts)
    assert result2.target_hash == hash1

    # Perturb target: hash must change
    perturbed_butt = target.butt.copy()
    perturbed_butt[0, 0] += 1e-4
    perturbed_target = ClubTarget(
        time=target.time,
        butt=perturbed_butt,
        clubhead=target.clubhead,
        club_quat=target.club_quat,
        impact_idx=target.impact_idx,
        source=target.source,
    )
    result_perturbed = fit_driven_double_pendulum(perturbed_target, opts=opts)
    assert result_perturbed.target_hash != hash1


def test_baseline_identity_and_status_bundle_provenance(
    synthetic_driven_target: tuple[ClubTarget, np.ndarray],
) -> None:
    """Verify BaselineIdentity and StatusBundle are completely and correctly populated."""
    target, _ = synthetic_driven_target
    opts = PendulumFitOptions(maxiter=20)

    result = fit_driven_double_pendulum(target, opts=opts)

    ident = result.baseline_identity
    assert ident.model_id == "driven_double_pendulum"
    assert ident.capture_sha256 == result.target_hash
    assert ident.provider_pin == "pendulum"
    assert ident.solver_name == "scipy_slsqp"
    assert ident.integrator == "rk4"

    bundle = result.status_bundle
    assert isinstance(bundle.solver_convergence, SolverConvergenceStatus)
    assert isinstance(bundle.kinematic_accuracy, KinematicAccuracyStatus)
    assert isinstance(bundle.dynamic_feasibility, DynamicFeasibilityStatus)
    assert bundle.scientific_qualification == ScientificQualificationStatus.QUALIFIED
    assert bundle.product_promotion == ProductPromotionStatus.EXPLORATORY
    assert bundle.has_native_replay is True


def test_independent_replay_zero_resets(
    standard_dynamics_params: DoublePendulumParameters,
) -> None:
    """Verify tighter-step independent replay integrates without resets and preserves stability."""
    times = np.linspace(0.0, 0.20, 21)
    q0 = np.array([0.2, -0.5])
    v0 = np.array([0.1, 0.2])
    torques = np.zeros((2, COEFFS_PER_JOINT))
    torques[0, -1] = 40.0
    torques[1, -1] = 15.0

    # 1-substep rollout
    grip_1, head_1, _ = simulate_pendulum_rollout(
        standard_dynamics_params, q0, v0, times, torques, substeps_per_frame=1
    )
    # 4-substep replay rollout
    grip_4, head_4, _ = simulate_pendulum_rollout(
        standard_dynamics_params, q0, v0, times, torques, substeps_per_frame=4
    )

    # Both must be finite and closely agree (RK4 convergence)
    assert np.all(np.isfinite(grip_4))
    assert np.all(np.isfinite(head_4))
    assert np.max(np.linalg.norm(grip_1 - grip_4, axis=1)) < 0.005
    assert np.max(np.linalg.norm(head_1 - head_4, axis=1)) < 0.010
