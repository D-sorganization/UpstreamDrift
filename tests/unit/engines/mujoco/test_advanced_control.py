"""Comprehensive tests for advanced control module."""

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.advanced_control import (
    AdvancedController,
    ControlMode,
    HybridControlMask,
    ImpedanceParameters,
    TrajectoryGenerator,
)
from mujoco_humanoid_golf.models import DOUBLE_PENDULUM_XML


class TestControlMode:
    """Tests for ControlMode enum."""

    def test_control_mode_values(self) -> None:
        """Test all control mode values."""
        assert ControlMode.TORQUE.value == "torque"
        assert ControlMode.IMPEDANCE.value == "impedance"
        assert ControlMode.ADMITTANCE.value == "admittance"
        assert ControlMode.HYBRID.value == "hybrid"
        assert ControlMode.COMPUTED_TORQUE.value == "computed_torque"
        assert ControlMode.TASK_SPACE.value == "task_space"


class TestImpedanceParameters:
    """Tests for ImpedanceParameters dataclass."""

    def test_advanced_control_initialization(self) -> None:
        """Test initialization with vector parameters."""
        stiffness = np.array([100.0, 50.0])
        damping = np.array([20.0, 10.0])
        params = ImpedanceParameters(stiffness=stiffness, damping=damping)

        np.testing.assert_array_equal(params.stiffness, stiffness)
        np.testing.assert_array_equal(params.damping, damping)
        assert params.inertia is None

    def test_initialization_with_inertia(self) -> None:
        """Test initialization with inertia matrix."""
        stiffness = np.array([100.0])
        damping = np.array([20.0])
        inertia = np.array([[1.0, 0.0], [0.0, 1.0]])
        params = ImpedanceParameters(
            stiffness=stiffness,
            damping=damping,
            inertia=inertia,
        )

        assert params.inertia is not None
        np.testing.assert_array_equal(params.inertia, inertia)

    def test_as_matrices_vector(self) -> None:
        """Test converting vector parameters to matrices."""
        stiffness = np.array([100.0, 50.0])
        damping = np.array([20.0, 10.0])
        params = ImpedanceParameters(stiffness=stiffness, damping=damping)

        k_matrix, d_matrix, m_matrix = params.as_matrices(2)

        np.testing.assert_array_equal(k_matrix, np.diag(stiffness))
        np.testing.assert_array_equal(d_matrix, np.diag(damping))
        np.testing.assert_array_equal(m_matrix, np.eye(2))

    def test_as_matrices_matrix(self) -> None:
        """Test with matrix parameters."""
        stiffness = np.array([[100.0, 10.0], [10.0, 50.0]])
        damping = np.array([[20.0, 5.0], [5.0, 10.0]])
        params = ImpedanceParameters(stiffness=stiffness, damping=damping)

        k_matrix, d_matrix, m_matrix = params.as_matrices(2)

        np.testing.assert_array_equal(k_matrix, stiffness)
        np.testing.assert_array_equal(d_matrix, damping)
        np.testing.assert_array_equal(m_matrix, np.eye(2))

    def test_as_matrices_with_inertia(self) -> None:
        """Test with inertia matrix."""
        stiffness = np.array([100.0])
        damping = np.array([20.0])
        inertia = np.array([1.0, 2.0])  # Vector
        params = ImpedanceParameters(
            stiffness=stiffness,
            damping=damping,
            inertia=inertia,
        )

        k_matrix, d_matrix, m_matrix = params.as_matrices(2)

        np.testing.assert_array_equal(m_matrix, np.diag(inertia))


class TestHybridControlMask:
    """Tests for HybridControlMask dataclass."""

    def test_advanced_control_initialization(self) -> None:
        """Test mask initialization."""
        force_mask = np.array([True, False, True])
        mask = HybridControlMask(force_mask=force_mask)

        np.testing.assert_array_equal(mask.force_mask, force_mask)

    def test_get_position_mask(self) -> None:
        """Test getting position mask."""
        force_mask = np.array([True, False, True])
        mask = HybridControlMask(force_mask=force_mask)

        position_mask = mask.get_position_mask()

        np.testing.assert_array_equal(position_mask, ~force_mask)

    def test_get_force_selection_matrix(self) -> None:
        """Test getting force selection matrix."""
        force_mask = np.array([True, False, True])
        mask = HybridControlMask(force_mask=force_mask)

        s_f = mask.get_force_selection_matrix()

        expected = np.diag([1.0, 0.0, 1.0])
        np.testing.assert_array_equal(s_f, expected)

    def test_get_position_selection_matrix(self) -> None:
        """Test getting position selection matrix."""
        force_mask = np.array([True, False, True])
        mask = HybridControlMask(force_mask=force_mask)

        s_p = mask.get_position_selection_matrix()

        expected = np.diag([0.0, 1.0, 0.0])
        np.testing.assert_array_equal(s_p, expected)


class TestTrajectoryGenerator:
    """Tests for TrajectoryGenerator class."""

    def test_minimum_jerk_trajectory(self) -> None:
        """Test minimum jerk trajectory generation."""
        start = np.array([0.0, 0.0])
        goal = np.array([1.0, 2.0])
        duration = 1.0
        dt = 0.01

        positions, velocities, accelerations = (
            TrajectoryGenerator.minimum_jerk_trajectory(
                start,
                goal,
                duration,
                dt,
            )
        )

        assert positions.shape[1] == 2
        assert velocities.shape[1] == 2
        assert accelerations.shape[1] == 2
        assert len(positions) == len(velocities) == len(accelerations)

        # Check boundary conditions
        np.testing.assert_allclose(positions[0], start, atol=1e-6)
        np.testing.assert_allclose(positions[-1], goal, atol=1e-6)
        np.testing.assert_allclose(velocities[0], [0, 0], atol=1e-3)
        np.testing.assert_allclose(velocities[-1], [0, 0], atol=1e-3)

    def test_minimum_jerk_trajectory_1d(self) -> None:
        """Test minimum jerk trajectory for 1D case."""
        start = np.array([0.0])
        goal = np.array([1.0])
        duration = 0.5
        dt = 0.01

        positions, velocities, accelerations = (
            TrajectoryGenerator.minimum_jerk_trajectory(
                start,
                goal,
                duration,
                dt,
            )
        )

        assert positions.shape[1] == 1
        assert positions[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert positions[-1, 0] == pytest.approx(1.0, abs=1e-6)

    def test_quintic_spline(self) -> None:
        """Test quintic spline generation."""
        waypoints = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]])
        duration = 2.0
        dt = 0.01

        positions, velocities, accelerations = TrajectoryGenerator.quintic_spline(
            waypoints,
            duration,
            dt,
        )

        assert positions.shape[1] == 2
        assert velocities.shape[1] == 2
        assert accelerations.shape[1] == 2

        # Check that trajectory passes through waypoints
        # First waypoint
        np.testing.assert_allclose(positions[0], waypoints[0], atol=1e-3)
        # Last waypoint
        np.testing.assert_allclose(positions[-1], waypoints[-1], atol=1e-3)

    def test_quintic_spline_two_waypoints(self) -> None:
        """Test quintic spline with two waypoints."""
        waypoints = np.array([[0.0], [1.0]])
        duration = 1.0
        dt = 0.01

        positions, velocities, accelerations = TrajectoryGenerator.quintic_spline(
            waypoints,
            duration,
            dt,
        )

        assert positions.shape[1] == 1
        assert len(positions) > 0

    def test_trajectory_smoothness(self) -> None:
        """Test that trajectories are smooth (no discontinuities)."""
        start = np.array([0.0])
        goal = np.array([1.0])
        duration = 1.0
        dt = 0.01

        positions, velocities, accelerations = (
            TrajectoryGenerator.minimum_jerk_trajectory(
                start,
                goal,
                duration,
                dt,
            )
        )

        # Check that velocities and accelerations are finite
        assert np.all(np.isfinite(velocities))
        assert np.all(np.isfinite(accelerations))

        # Check that there are no large jumps
        pos_diffs = np.diff(positions, axis=0)
        assert np.all(np.abs(pos_diffs) < 0.1)  # No large jumps
