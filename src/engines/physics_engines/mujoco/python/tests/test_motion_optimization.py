"""Comprehensive tests for motion optimization module."""

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.models import DOUBLE_PENDULUM_XML
from mujoco_humanoid_golf.motion_optimization import (
    OptimizationConstraints,
    OptimizationObjectives,
    OptimizationResult,
    SwingOptimizer,
)


class TestOptimizationObjectives:
    """Tests for OptimizationObjectives dataclass."""

    def test_initialization(self) -> None:
        """Test objectives initialization."""
        objectives = OptimizationObjectives()

        assert objectives.maximize_club_speed is True
        assert objectives.minimize_energy is True
        assert objectives.weight_speed == 10.0

    def test_initialization_with_target(self) -> None:
        """Test objectives with target position."""
        target = np.array([1.0, 0.0, 0.0])
        objectives = OptimizationObjectives(target_ball_position=target)

        assert objectives.target_ball_position is not None
        np.testing.assert_array_equal(objectives.target_ball_position, target)


class TestOptimizationConstraints:
    """Tests for OptimizationConstraints dataclass."""

    def test_initialization(self) -> None:
        """Test constraints initialization."""
        constraints = OptimizationConstraints()

        assert constraints.joint_position_limits is True
        assert constraints.joint_velocity_limits is True
        assert constraints.maintain_grip is True


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_initialization(self) -> None:
        """Test result initialization."""
        trajectory = np.zeros((10, 2))
        velocities = np.zeros((10, 2))
        controls = np.zeros((10, 1))

        result = OptimizationResult(
            success=True,
            optimal_trajectory=trajectory,
            optimal_velocities=velocities,
            optimal_controls=controls,
            objective_value=1.5,
            num_iterations=100,
            computation_time=2.0,
            peak_club_speed=10.0,
            final_club_position=np.array([1.0, 0.0, 0.0]),
        )

        assert result.success is True
        assert result.objective_value == 1.5
        assert result.num_iterations == 100


class TestSwingOptimizer:
    """Tests for SwingOptimizer class."""

    @pytest.fixture()
    def model_and_data(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Create model and data for testing."""
        model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def test_initialization(self, model_and_data) -> None:
        """Test optimizer initialization."""
        model, data = model_and_data
        optimizer = SwingOptimizer(model, data)

        assert optimizer.model == model
        assert optimizer.data == data
        assert optimizer.num_knot_points == 10
        assert optimizer.swing_duration == 1.5

    def test_initialization_with_objectives(self, model_and_data) -> None:
        """Test initialization with custom objectives."""
        model, data = model_and_data
        objectives = OptimizationObjectives(weight_speed=20.0)
        optimizer = SwingOptimizer(model, data, objectives=objectives)

        assert optimizer.objectives.weight_speed == 20.0

    def test_initialization_with_constraints(self, model_and_data) -> None:
        """Test initialization with custom constraints."""
        model, data = model_and_data
        constraints = OptimizationConstraints(joint_velocity_limits=False)
        optimizer = SwingOptimizer(model, data, constraints=constraints)

        assert optimizer.constraints.joint_velocity_limits is False

    def test_find_body_id(self, model_and_data) -> None:
        """Test finding body ID."""
        model, data = model_and_data
        optimizer = SwingOptimizer(model, data)

        body_id = optimizer._find_body_id("shoulder")
        if body_id is not None:
            assert body_id > 0
            assert body_id < model.nbody

        # Should return None for nonexistent body
        body_id = optimizer._find_body_id("nonexistent_body_xyz")
        assert body_id is None

    def test_generate_initial_guess(self, model_and_data) -> None:
        """Test generating initial guess."""
        model, data = model_and_data
        optimizer = SwingOptimizer(model, data)

        initial_guess = optimizer._generate_initial_guess()

        assert initial_guess.shape == (optimizer.num_knot_points, model.nv)
        assert np.all(np.isfinite(initial_guess))

    def test_compute_bounds(self, model_and_data) -> None:
        """Test computing optimization bounds."""
        model, data = model_and_data
        optimizer = SwingOptimizer(model, data)

        bounds = optimizer._compute_bounds()

        # Should have bounds for all decision variables
        expected_size = optimizer.num_knot_points * model.nv
        assert len(bounds) == expected_size
        assert all(isinstance(b, tuple) and len(b) == 2 for b in bounds)

    def test_setup_constraints(self, model_and_data) -> None:
        """Test setting up constraints."""
        model, data = model_and_data
        optimizer = SwingOptimizer(model, data)

        constraints = optimizer._setup_constraints()

        assert isinstance(constraints, list)

    def test_evaluate_objective(self, model_and_data) -> None:
        """Test evaluating objective function."""
        model, data = model_and_data
        optimizer = SwingOptimizer(model, data)

        initial_guess = optimizer._generate_initial_guess()
        x = initial_guess.flatten()

        objective_value = optimizer._evaluate_objective(x)

        assert isinstance(objective_value, float)
        assert np.isfinite(objective_value)

    def test_simulate_trajectory(self, model_and_data) -> None:
        """Test simulating trajectory."""
        model, data = model_and_data
        optimizer = SwingOptimizer(model, data)

        trajectory = optimizer._generate_initial_guess()
        velocities, controls, metrics = optimizer._simulate_trajectory(trajectory)

        # _simulate_trajectory returns velocities for all simulation steps,
        # not just knot points
        # The number of steps depends on swing_duration and model timestep
        assert velocities.shape[1] == model.nv
        assert controls.shape[1] == model.nu
        assert "peak_club_speed" in metrics
        assert "final_club_position" in metrics

    def test_optimize_trajectory_slsqp(self, model_and_data) -> None:
        """Test optimizing trajectory with SLSQP."""
        model, data = model_and_data
        optimizer = SwingOptimizer(model, data)

        # Use fewer knot points for faster testing
        optimizer.num_knot_points = 5

        result = optimizer.optimize_trajectory(method="SLSQP")

        assert isinstance(result, OptimizationResult)
        assert result.optimal_trajectory.shape[0] == optimizer.num_knot_points
        # optimal_velocities comes from _simulate_trajectory which returns
        # all simulation steps
        assert (
            result.optimal_velocities.shape[0] > 0
        ), "Should have at least one timestep"
        assert result.optimal_velocities.shape[1] == model.nv
        assert np.all(np.isfinite(result.optimal_trajectory))

    @pytest.mark.slow()
    def test_optimize_trajectory_differential_evolution(self, model_and_data) -> None:
        """Test optimizing trajectory with differential evolution."""
        model, data = model_and_data
        optimizer = SwingOptimizer(model, data)

        # Use fewer knot points for faster testing
        optimizer.num_knot_points = 5

        result = optimizer.optimize_trajectory(method="differential_evolution")

        assert isinstance(result, OptimizationResult)
        assert result.optimal_trajectory.shape[0] == optimizer.num_knot_points

    def test_optimize_trajectory_with_initial_guess(self, model_and_data) -> None:
        """Test optimizing with provided initial guess."""
        model, data = model_and_data
        optimizer = SwingOptimizer(model, data)
        optimizer.num_knot_points = 5

        initial_guess = optimizer._generate_initial_guess()
        result = optimizer.optimize_trajectory(
            initial_guess=initial_guess, method="SLSQP"
        )

        assert isinstance(result, OptimizationResult)
        assert result.optimal_trajectory.shape == initial_guess.shape
