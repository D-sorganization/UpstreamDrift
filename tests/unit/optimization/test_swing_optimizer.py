"""Unit tests for the SwingOptimizer module.

Tests cover:
- Model initialization
- Data classes
- Basic optimization
- Constraint validation
- Edge cases
"""

from __future__ import annotations

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

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_golfer() -> GolferModel:
    """Default golfer model with average parameters."""
    return GolferModel()


@pytest.fixture
def custom_golfer() -> GolferModel:
    """Custom golfer with specific parameters."""
    return GolferModel(
        height=1.85,
        mass=85.0,
        arm_length=0.65,
        max_shoulder_torque=120.0,
        flexibility_factor=1.1,
    )


@pytest.fixture
def default_club() -> ClubModel:
    """Default driver club model."""
    return ClubModel()


@pytest.fixture
def iron_club() -> ClubModel:
    """7-iron club model."""
    return ClubModel(
        total_length=0.95,
        shaft_length=0.85,
        head_mass=0.25,
        shaft_flex="stiff",
    )


@pytest.fixture
def basic_config() -> OptimizationConfig:
    """Basic optimization config for fast testing."""
    return OptimizationConfig(
        objectives={OptimizationObjective.CLUBHEAD_VELOCITY: 1.0},
        constraints=[OptimizationConstraint.JOINT_LIMITS],
        n_nodes=10,  # Reduced for faster tests
        max_iterations=5,  # Minimal iterations for unit tests
    )


# =============================================================================
# GolferModel Tests
# =============================================================================


def test_golfer_model_defaults() -> None:
    """Test GolferModel has reasonable defaults."""
    golfer = GolferModel()
    assert golfer.height == 1.75
    assert golfer.mass == 75.0
    assert golfer.arm_length == 0.60
    assert golfer.flexibility_factor == 1.0


def test_golfer_model_custom() -> None:
    """Test GolferModel accepts custom parameters."""
    golfer = GolferModel(height=1.90, mass=90.0)
    assert golfer.height == 1.90
    assert golfer.mass == 90.0


def test_golfer_model_joint_limits() -> None:
    """Test GolferModel joint limits are tuples."""
    golfer = GolferModel()
    assert isinstance(golfer.shoulder_rom, tuple)
    assert len(golfer.shoulder_rom) == 2
    assert golfer.shoulder_rom[0] < golfer.shoulder_rom[1]


def test_golfer_model_torque_limits_positive() -> None:
    """Test GolferModel torque limits are positive."""
    golfer = GolferModel()
    assert golfer.max_shoulder_torque > 0
    assert golfer.max_elbow_torque > 0
    assert golfer.max_wrist_torque > 0
    assert golfer.max_hip_torque > 0
    assert golfer.max_trunk_torque > 0


# =============================================================================
# ClubModel Tests
# =============================================================================


def test_club_model_defaults() -> None:
    """Test ClubModel has reasonable defaults for a driver."""
    club = ClubModel()
    assert club.total_length == 1.15
    assert club.head_mass == 0.20
    assert club.shaft_flex == "regular"


def test_club_model_total_mass() -> None:
    """Test ClubModel calculates total mass correctly."""
    club = ClubModel()
    expected_mass = club.head_mass + club.shaft_mass + club.grip_mass
    assert club.total_mass == expected_mass


def test_club_model_moment_of_inertia() -> None:
    """Test ClubModel MOI is positive and reasonable."""
    club = ClubModel()
    assert club.club_moi > 0
    # MOI should be on the order of 0.1 kg*m^2 for a golf club
    assert 0.01 < club.club_moi < 1.0


# =============================================================================
# OptimizationConfig Tests
# =============================================================================


def test_optimization_config_defaults() -> None:
    """Test OptimizationConfig has valid defaults."""
    config = OptimizationConfig()
    assert config.n_nodes >= 10
    assert config.swing_duration > 0
    assert len(config.objectives) > 0


def test_optimization_config_custom_objectives() -> None:
    """Test OptimizationConfig accepts custom objectives."""
    config = OptimizationConfig(
        objectives={
            OptimizationObjective.CLUBHEAD_VELOCITY: 0.7,
            OptimizationObjective.INJURY_RISK: 0.3,
        }
    )
    assert OptimizationObjective.CLUBHEAD_VELOCITY in config.objectives
    assert OptimizationObjective.INJURY_RISK in config.objectives


def test_optimization_config_constraints() -> None:
    """Test OptimizationConfig accepts constraints."""
    config = OptimizationConfig(
        constraints=[
            OptimizationConstraint.JOINT_LIMITS,
            OptimizationConstraint.TORQUE_LIMITS,
        ]
    )
    assert OptimizationConstraint.JOINT_LIMITS in config.constraints
    assert OptimizationConstraint.TORQUE_LIMITS in config.constraints


# =============================================================================
# SwingOptimizer Initialization Tests
# =============================================================================


def test_optimizer_init(default_golfer, default_club, basic_config) -> None:
    """Test SwingOptimizer initializes correctly."""
    optimizer = SwingOptimizer(default_golfer, default_club, basic_config)
    assert optimizer.golfer == default_golfer
    assert optimizer.club == default_club
    assert optimizer.config == basic_config


def test_optimizer_init_default_config(default_golfer, default_club) -> None:
    """Test SwingOptimizer uses default config when not provided."""
    optimizer = SwingOptimizer(default_golfer, default_club)
    assert optimizer.config is not None
    assert isinstance(optimizer.config, OptimizationConfig)


def test_optimizer_joint_limits_setup(
    default_golfer, default_club, basic_config
) -> None:
    """Test SwingOptimizer sets up joint limits correctly."""
    optimizer = SwingOptimizer(default_golfer, default_club, basic_config)
    assert "hip_rotation" in optimizer.joint_limits
    assert "shoulder_horizontal" in optimizer.joint_limits
    assert len(optimizer.joint_limits) == len(SwingOptimizer.JOINTS)


def test_optimizer_torque_limits_setup(
    default_golfer, default_club, basic_config
) -> None:
    """Test SwingOptimizer sets up torque limits correctly."""
    optimizer = SwingOptimizer(default_golfer, default_club, basic_config)
    assert "hip_rotation" in optimizer.torque_limits
    assert all(t > 0 for t in optimizer.torque_limits.values())


def test_optimizer_lever_length(default_golfer, default_club, basic_config) -> None:
    """Test total lever (arm + club) is calculated correctly."""
    optimizer = SwingOptimizer(default_golfer, default_club, basic_config)
    expected = default_golfer.arm_length + default_club.total_length
    assert optimizer.total_lever == expected


# =============================================================================
# SwingTrajectory Tests
# =============================================================================


def test_swing_trajectory_creation() -> None:
    """Test SwingTrajectory can be created with valid data."""
    n_nodes = 20
    trajectory = SwingTrajectory(
        time=np.linspace(0, 0.3, n_nodes),
        joint_angles={"shoulder": np.zeros(n_nodes), "elbow": np.zeros(n_nodes)},
        joint_velocities={"shoulder": np.zeros(n_nodes), "elbow": np.zeros(n_nodes)},
        joint_torques={"shoulder": np.zeros(n_nodes), "elbow": np.zeros(n_nodes)},
        clubhead_position=np.zeros((n_nodes, 3)),
        clubhead_velocity=np.zeros((n_nodes, 3)),
    )
    assert trajectory.time.shape == (n_nodes,)
    assert "shoulder" in trajectory.joint_angles


# =============================================================================
# Optimization Tests (Smoke Tests)
# =============================================================================


@pytest.mark.slow
def test_optimizer_optimize_runs(default_golfer, default_club, basic_config) -> None:
    """Test optimize() runs without error (smoke test)."""
    optimizer = SwingOptimizer(default_golfer, default_club, basic_config)
    result = optimizer.optimize()
    assert isinstance(result, OptimizationResult)


@pytest.mark.slow
def test_optimizer_result_has_trajectory(
    default_golfer, default_club, basic_config
) -> None:
    """Test optimization result contains a trajectory."""
    optimizer = SwingOptimizer(default_golfer, default_club, basic_config)
    result = optimizer.optimize()
    assert result.trajectory is not None
    assert isinstance(result.trajectory, SwingTrajectory)


@pytest.mark.slow
def test_optimizer_result_has_metrics(
    default_golfer, default_club, basic_config
) -> None:
    """Test optimization result contains performance metrics."""
    optimizer = SwingOptimizer(default_golfer, default_club, basic_config)
    result = optimizer.optimize()
    assert result.predicted_clubhead_speed >= 0


@pytest.mark.slow
def test_optimizer_respects_joint_limits(
    default_golfer, default_club, basic_config
) -> None:
    """Test optimized trajectory respects joint limits."""
    optimizer = SwingOptimizer(default_golfer, default_club, basic_config)
    result = optimizer.optimize()

    if result.trajectory is not None:
        angles = result.trajectory.joint_angles
        # Check angles are within reasonable bounds (allowing some tolerance)
        for joint_name, joint_angles in angles.items():
            assert np.all(np.abs(joint_angles) < 5.0), (
                f"Joint {joint_name} exceeds limits"
            )


@pytest.mark.slow
def test_optimizer_callback(default_golfer, default_club, basic_config) -> None:
    """Test optimizer calls callback function."""
    optimizer = SwingOptimizer(default_golfer, default_club, basic_config)
    callback_calls = []

    def my_callback(iteration: int, value: float) -> None:
        """Record each optimizer callback invocation."""
        callback_calls.append((iteration, value))

    optimizer.optimize(callback=my_callback)
    assert len(callback_calls) > 0


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_optimizer_with_zero_flexibility(default_club, basic_config) -> None:
    """Test optimizer handles golfer with minimal flexibility."""
    stiff_golfer = GolferModel(flexibility_factor=0.5)
    optimizer = SwingOptimizer(stiff_golfer, default_club, basic_config)
    # Should not crash during setup
    assert optimizer.golfer.flexibility_factor == 0.5


def test_optimizer_with_heavy_club(default_golfer, basic_config) -> None:
    """Test optimizer handles heavy club."""
    heavy_club = ClubModel(head_mass=0.35, shaft_mass=0.12)
    optimizer = SwingOptimizer(default_golfer, heavy_club, basic_config)
    # Should not crash during setup
    assert optimizer.club.head_mass == 0.35


def test_optimizer_objectives_enum() -> None:
    """Test all optimization objectives are defined."""
    assert len(OptimizationObjective) >= 5
    assert OptimizationObjective.CLUBHEAD_VELOCITY.value == "clubhead_velocity"


def test_optimizer_constraints_enum() -> None:
    """Test all optimization constraints are defined."""
    assert len(OptimizationConstraint) >= 4
    assert OptimizationConstraint.JOINT_LIMITS.value == "joint_limits"
