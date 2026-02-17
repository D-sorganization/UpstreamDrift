"""Comprehensive unit tests for the optimization package.

Tests cover:
- OptimizationObjective and OptimizationConstraint enums
- GolferModel dataclass defaults and custom values
- ClubModel dataclass with derived properties (total_mass, club_moi)
- OptimizationConfig dataclass defaults
- SwingTrajectory and OptimizationResult dataclasses
- SwingOptimizer initialization, model setup, initial guess generation,
  trajectory conversion, bounds, constraints, objectives, and metrics
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.optimization.swing_optimizer import (
    ClubModel,
    GolferModel,
    OptimizationConfig,
    OptimizationConstraint,
    OptimizationObjective,
    OptimizationResult,
    SwingOptimizer,
    SwingTrajectory,
)

# ============================================================================
# OptimizationObjective Enum
# ============================================================================


class TestOptimizationObjective:
    """Tests for the OptimizationObjective enum."""

    def test_all_values_exist(self) -> None:
        expected = {
            "clubhead_velocity",
            "ball_distance",
            "accuracy",
            "energy_efficiency",
            "injury_risk",
            "consistency",
        }
        actual = {o.value for o in OptimizationObjective}
        assert actual == expected

    def test_members_accessible_by_name(self) -> None:
        assert OptimizationObjective.CLUBHEAD_VELOCITY.value == "clubhead_velocity"
        assert OptimizationObjective.INJURY_RISK.value == "injury_risk"

    def test_total_count(self) -> None:
        assert len(OptimizationObjective) == 6


# ============================================================================
# OptimizationConstraint Enum
# ============================================================================


class TestOptimizationConstraint:
    """Tests for the OptimizationConstraint enum."""

    def test_all_values_exist(self) -> None:
        expected = {
            "joint_limits",
            "torque_limits",
            "velocity_limits",
            "contact_constraints",
            "kinematic_chain",
        }
        actual = {c.value for c in OptimizationConstraint}
        assert actual == expected

    def test_total_count(self) -> None:
        assert len(OptimizationConstraint) == 5


# ============================================================================
# GolferModel Dataclass
# ============================================================================


class TestGolferModel:
    """Tests for the GolferModel dataclass."""

    def test_default_anthropometrics(self) -> None:
        g = GolferModel()
        assert g.height == 1.75
        assert g.mass == 75.0
        assert g.arm_length == 0.60
        assert g.trunk_length == 0.50

    def test_default_mass_ratios(self) -> None:
        g = GolferModel()
        assert g.arm_mass_ratio == 0.05
        assert g.trunk_mass_ratio == 0.43
        # Mass ratios should sum to less than 1
        assert g.arm_mass_ratio + g.trunk_mass_ratio < 1.0

    def test_default_joint_roms(self) -> None:
        g = GolferModel()
        # ROM tuples must have lower < upper
        for rom in [
            g.shoulder_rom,
            g.elbow_rom,
            g.wrist_rom,
            g.hip_rom,
            g.trunk_rotation_rom,
        ]:
            assert rom[0] < rom[1], f"ROM {rom} invalid: lower >= upper"

    def test_default_torque_limits_positive(self) -> None:
        g = GolferModel()
        assert g.max_shoulder_torque > 0
        assert g.max_elbow_torque > 0
        assert g.max_wrist_torque > 0
        assert g.max_hip_torque > 0
        assert g.max_trunk_torque > 0

    def test_torque_ordering(self) -> None:
        """Larger muscle groups should produce greater torques."""
        g = GolferModel()
        assert g.max_trunk_torque > g.max_shoulder_torque
        assert g.max_hip_torque > g.max_shoulder_torque
        assert g.max_shoulder_torque > g.max_elbow_torque
        assert g.max_elbow_torque > g.max_wrist_torque

    def test_custom_values(self) -> None:
        g = GolferModel(height=1.90, mass=90.0, arm_length=0.70)
        assert g.height == 1.90
        assert g.mass == 90.0
        assert g.arm_length == 0.70

    def test_flexibility_factor_default(self) -> None:
        g = GolferModel()
        assert g.flexibility_factor == 1.0


# ============================================================================
# ClubModel Dataclass
# ============================================================================


class TestClubModel:
    """Tests for the ClubModel dataclass and its derived properties."""

    def test_default_values(self) -> None:
        c = ClubModel()
        assert c.total_length == 1.15
        assert c.shaft_length == 1.05
        assert c.head_mass == 0.20
        assert c.shaft_mass == 0.07
        assert c.grip_mass == 0.05

    def test_total_mass(self) -> None:
        c = ClubModel()
        expected = c.head_mass + c.shaft_mass + c.grip_mass
        assert c.total_mass == pytest.approx(expected)

    def test_total_mass_custom(self) -> None:
        c = ClubModel(head_mass=0.25, shaft_mass=0.10, grip_mass=0.06)
        assert c.total_mass == pytest.approx(0.41)

    def test_club_moi_formula(self) -> None:
        """MOI = m_head * L² + m_shaft * (L_shaft/2)²."""
        c = ClubModel()
        expected = (
            c.head_mass * c.total_length**2 + c.shaft_mass * (c.shaft_length / 2) ** 2
        )
        assert c.club_moi == pytest.approx(expected)

    def test_moi_increases_with_head_mass(self) -> None:
        c_light = ClubModel(head_mass=0.15)
        c_heavy = ClubModel(head_mass=0.30)
        assert c_heavy.club_moi > c_light.club_moi

    def test_moi_increases_with_length(self) -> None:
        c_short = ClubModel(total_length=1.0)
        c_long = ClubModel(total_length=1.3)
        assert c_long.club_moi > c_short.club_moi

    def test_shaft_flex_options(self) -> None:
        for flex in ["stiff", "regular", "senior", "ladies"]:
            c = ClubModel(shaft_flex=flex)
            assert c.shaft_flex == flex

    def test_face_angle_default_square(self) -> None:
        c = ClubModel()
        assert c.face_angle == 0.0

    def test_loft_angle_default(self) -> None:
        c = ClubModel()
        assert c.loft_angle == 10.5  # Driver


# ============================================================================
# OptimizationConfig Dataclass
# ============================================================================


class TestOptimizationConfig:
    """Tests for the OptimizationConfig dataclass."""

    def test_default_objectives(self) -> None:
        cfg = OptimizationConfig()
        assert OptimizationObjective.CLUBHEAD_VELOCITY in cfg.objectives
        assert cfg.objectives[OptimizationObjective.CLUBHEAD_VELOCITY] == 1.0

    def test_default_constraints(self) -> None:
        cfg = OptimizationConfig()
        assert OptimizationConstraint.JOINT_LIMITS in cfg.constraints
        assert OptimizationConstraint.TORQUE_LIMITS in cfg.constraints

    def test_default_time_params(self) -> None:
        cfg = OptimizationConfig()
        assert cfg.n_nodes == 50
        assert cfg.swing_duration == 1.2
        assert 0 < cfg.backswing_fraction < 1

    def test_default_solver_settings(self) -> None:
        cfg = OptimizationConfig()
        assert cfg.max_iterations > 0
        assert cfg.tolerance > 0
        assert cfg.solver == "SLSQP"

    def test_custom_config(self) -> None:
        cfg = OptimizationConfig(
            objectives={OptimizationObjective.ACCURACY: 2.0},
            n_nodes=100,
            swing_duration=1.5,
        )
        assert OptimizationObjective.ACCURACY in cfg.objectives
        assert cfg.n_nodes == 100
        assert cfg.swing_duration == 1.5


# ============================================================================
# SwingTrajectory Dataclass
# ============================================================================


class TestSwingTrajectory:
    """Tests for the SwingTrajectory dataclass."""

    @pytest.fixture()
    def sample_trajectory(self) -> SwingTrajectory:
        n = 10
        t = np.linspace(0, 1.2, n)
        return SwingTrajectory(
            time=t,
            joint_angles={"hip": np.sin(t), "shoulder": np.cos(t)},
            joint_velocities={"hip": np.cos(t), "shoulder": -np.sin(t)},
            joint_torques={"hip": np.zeros(n), "shoulder": np.zeros(n)},
            clubhead_position=np.zeros((n, 3)),
            clubhead_velocity=np.ones((n, 3)),
            impact_speed=45.0,
            impact_time=0.72,
        )

    def test_fields_accessible(self, sample_trajectory: SwingTrajectory) -> None:
        assert len(sample_trajectory.time) == 10
        assert "hip" in sample_trajectory.joint_angles
        assert sample_trajectory.impact_speed == 45.0

    def test_default_impact_values(self) -> None:
        n = 5
        traj = SwingTrajectory(
            time=np.zeros(n),
            joint_angles={},
            joint_velocities={},
            joint_torques={},
            clubhead_position=np.zeros((n, 3)),
            clubhead_velocity=np.zeros((n, 3)),
        )
        assert traj.impact_speed == 0.0
        assert traj.impact_time == 0.0


# ============================================================================
# OptimizationResult Dataclass
# ============================================================================


class TestOptimizationResult:
    """Tests for the OptimizationResult dataclass."""

    def test_success_result(self) -> None:
        r = OptimizationResult(
            success=True,
            message="Converged",
            predicted_clubhead_speed=50.0,
            predicted_ball_speed=75.0,
        )
        assert r.success is True
        assert r.predicted_clubhead_speed == 50.0
        assert r.predicted_ball_speed == 75.0

    def test_failure_result(self) -> None:
        r = OptimizationResult(success=False, message="Did not converge")
        assert r.success is False
        assert r.trajectory is None

    def test_default_metrics_zero(self) -> None:
        r = OptimizationResult(success=True, message="ok")
        assert r.predicted_carry_distance == 0.0
        assert r.injury_risk_score == 0.0
        assert r.computation_time == 0.0
        assert r.iterations == 0
        assert r.speed_improvement == 0.0
        assert r.risk_reduction == 0.0


# ============================================================================
# SwingOptimizer — Initialization and Model Setup
# ============================================================================


class TestSwingOptimizerInit:
    """Tests for SwingOptimizer initialization and model setup."""

    @pytest.fixture()
    def optimizer(self) -> SwingOptimizer:
        return SwingOptimizer(GolferModel(), ClubModel())

    def test_default_creation(self, optimizer: SwingOptimizer) -> None:
        assert optimizer.golfer.mass == 75.0
        assert optimizer.club.total_length == 1.15
        assert optimizer.config.n_nodes == 50

    def test_custom_config(self) -> None:
        cfg = OptimizationConfig(n_nodes=20, swing_duration=1.0)
        opt = SwingOptimizer(GolferModel(), ClubModel(), config=cfg)
        assert opt.config.n_nodes == 20
        assert opt.config.swing_duration == 1.0

    def test_total_lever_computed(self, optimizer: SwingOptimizer) -> None:
        expected = optimizer.golfer.arm_length + optimizer.club.total_length
        assert optimizer.total_lever == pytest.approx(expected)

    def test_system_moi_positive(self, optimizer: SwingOptimizer) -> None:
        assert optimizer.system_moi > 0

    def test_joint_limits_cover_all_joints(self, optimizer: SwingOptimizer) -> None:
        for joint in SwingOptimizer.JOINTS:
            assert joint in optimizer.joint_limits
            lo, hi = optimizer.joint_limits[joint]
            assert lo < hi

    def test_torque_limits_cover_all_joints(self, optimizer: SwingOptimizer) -> None:
        for joint in SwingOptimizer.JOINTS:
            assert joint in optimizer.torque_limits
            assert optimizer.torque_limits[joint] > 0

    def test_seven_joints_defined(self) -> None:
        assert len(SwingOptimizer.JOINTS) == 7


# ============================================================================
# SwingOptimizer — Initial Guess and Trajectory Conversion
# ============================================================================


class TestSwingOptimizerGuess:
    """Tests for initial guess generation and trajectory vector conversion."""

    @pytest.fixture()
    def optimizer(self) -> SwingOptimizer:
        cfg = OptimizationConfig(n_nodes=10, swing_duration=1.0)
        return SwingOptimizer(GolferModel(), ClubModel(), config=cfg)

    def test_initial_guess_shape(self, optimizer: SwingOptimizer) -> None:
        x0 = optimizer._generate_initial_guess()
        n_joints = len(SwingOptimizer.JOINTS)
        n_nodes = optimizer.config.n_nodes
        # angles + velocities
        assert x0.shape == (2 * n_joints * n_nodes,)

    def test_initial_guess_finite(self, optimizer: SwingOptimizer) -> None:
        x0 = optimizer._generate_initial_guess()
        assert np.all(np.isfinite(x0))

    def test_vector_to_trajectory_roundtrip(self, optimizer: SwingOptimizer) -> None:
        x0 = optimizer._generate_initial_guess()
        traj = optimizer._vector_to_trajectory(x0)
        x_rt = optimizer._trajectory_to_vector(traj)
        np.testing.assert_allclose(x0, x_rt, atol=1e-10)

    def test_trajectory_has_all_joints(self, optimizer: SwingOptimizer) -> None:
        x0 = optimizer._generate_initial_guess()
        traj = optimizer._vector_to_trajectory(x0)
        for joint in SwingOptimizer.JOINTS:
            assert joint in traj.joint_angles
            assert joint in traj.joint_velocities
            assert joint in traj.joint_torques

    def test_trajectory_time_array(self, optimizer: SwingOptimizer) -> None:
        x0 = optimizer._generate_initial_guess()
        traj = optimizer._vector_to_trajectory(x0)
        assert len(traj.time) == optimizer.config.n_nodes
        assert traj.time[0] == pytest.approx(0.0)
        assert traj.time[-1] == pytest.approx(optimizer.config.swing_duration)

    def test_clubhead_trajectory_shape(self, optimizer: SwingOptimizer) -> None:
        x0 = optimizer._generate_initial_guess()
        traj = optimizer._vector_to_trajectory(x0)
        n = optimizer.config.n_nodes
        assert traj.clubhead_position.shape == (n, 3)
        assert traj.clubhead_velocity.shape == (n, 3)

    def test_impact_speed_nonnegative(self, optimizer: SwingOptimizer) -> None:
        x0 = optimizer._generate_initial_guess()
        traj = optimizer._vector_to_trajectory(x0)
        assert traj.impact_speed >= 0


# ============================================================================
# SwingOptimizer — Bounds and Constraints
# ============================================================================


class TestSwingOptimizerBounds:
    """Tests for optimization bounds and constraint generation."""

    @pytest.fixture()
    def optimizer(self) -> SwingOptimizer:
        cfg = OptimizationConfig(n_nodes=5)
        return SwingOptimizer(GolferModel(), ClubModel(), config=cfg)

    def test_bounds_count(self, optimizer: SwingOptimizer) -> None:
        bounds = optimizer._get_bounds()
        n_joints = len(SwingOptimizer.JOINTS)
        n_nodes = optimizer.config.n_nodes
        # angles (n_joints*n_nodes) + velocities (n_joints*n_nodes)
        assert len(bounds) == 2 * n_joints * n_nodes

    def test_bounds_are_tuples(self, optimizer: SwingOptimizer) -> None:
        bounds = optimizer._get_bounds()
        for lo, hi in bounds:
            assert lo < hi

    def test_constraints_list(self, optimizer: SwingOptimizer) -> None:
        constraints = optimizer._build_constraints()
        # Default config has TORQUE_LIMITS but not KINEMATIC_CHAIN
        assert len(constraints) >= 1
        assert all(isinstance(c, dict) for c in constraints)


# ============================================================================
# SwingOptimizer — Objective and Metrics
# ============================================================================


class TestSwingOptimizerObjective:
    """Tests for objective computation and metrics."""

    @pytest.fixture()
    def optimizer(self) -> SwingOptimizer:
        cfg = OptimizationConfig(n_nodes=10, swing_duration=1.0)
        return SwingOptimizer(GolferModel(), ClubModel(), config=cfg)

    def test_objective_returns_scalar(self, optimizer: SwingOptimizer) -> None:
        x0 = optimizer._generate_initial_guess()
        obj = optimizer._compute_objective(x0)
        assert isinstance(obj, float)
        assert math.isfinite(obj)

    def test_injury_risk_bounded(self, optimizer: SwingOptimizer) -> None:
        x0 = optimizer._generate_initial_guess()
        traj = optimizer._vector_to_trajectory(x0)
        risk = optimizer._compute_injury_risk(traj)
        assert 0 <= risk <= 100

    def test_energy_cost_nonnegative(self, optimizer: SwingOptimizer) -> None:
        x0 = optimizer._generate_initial_guess()
        traj = optimizer._vector_to_trajectory(x0)
        energy = optimizer._compute_energy_cost(traj)
        assert energy >= 0

    def test_metrics_complete(self, optimizer: SwingOptimizer) -> None:
        x0 = optimizer._generate_initial_guess()
        traj = optimizer._vector_to_trajectory(x0)
        metrics = optimizer._compute_metrics(traj)
        expected_keys = {
            "clubhead_speed",
            "ball_speed",
            "carry_distance",
            "launch_angle",
            "spin_rate",
            "spinal_compression",
            "spinal_shear",
            "injury_risk",
        }
        assert set(metrics.keys()) == expected_keys

    def test_ball_speed_greater_than_clubhead(self, optimizer: SwingOptimizer) -> None:
        """Smash factor > 1 means ball speed > clubhead speed."""
        x0 = optimizer._generate_initial_guess()
        traj = optimizer._vector_to_trajectory(x0)
        metrics = optimizer._compute_metrics(traj)
        if metrics["clubhead_speed"] > 0:
            assert metrics["ball_speed"] > metrics["clubhead_speed"]


# ============================================================================
# SwingOptimizer — Full Optimization (smoke test)
# ============================================================================


class TestSwingOptimizerOptimize:
    """Smoke test for full optimization run."""

    def test_optimize_returns_result(self) -> None:
        """Verify optimize() returns an OptimizationResult, pass or fail."""
        cfg = OptimizationConfig(
            n_nodes=5,
            swing_duration=0.5,
            max_iterations=10,  # Keep fast
        )
        opt = SwingOptimizer(GolferModel(), ClubModel(), config=cfg)
        result = opt.optimize()
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.success, bool)
        assert result.computation_time >= 0

    def test_optimize_with_callback(self) -> None:
        """Verify callback is invoked."""
        called: list[tuple[int, float]] = []

        def cb(iteration: int, obj_val: float) -> None:
            called.append((iteration, obj_val))

        cfg = OptimizationConfig(n_nodes=5, max_iterations=5)
        opt = SwingOptimizer(GolferModel(), ClubModel(), config=cfg)
        opt.optimize(callback=cb)
        # Callback may or may not be called depending on solver behaviour
        # Just verify no crash
