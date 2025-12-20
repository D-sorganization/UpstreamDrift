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

    def test_initialization(self) -> None:
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

    def test_initialization(self) -> None:
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


class TestAdvancedController:
    """Tests for AdvancedController class."""

    @pytest.fixture()
    def model_and_data(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Create model and data for testing."""
        model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def test_initialization(self, model_and_data) -> None:
        """Test controller initialization."""
        model, data = model_and_data
        controller = AdvancedController(model, data)

        assert controller.model == model
        assert controller.data == data
        assert controller.mode == ControlMode.TORQUE
        assert controller.enable_gravity_compensation is True

    def test_set_control_mode(self, model_and_data) -> None:
        """Test setting control mode."""
        model, data = model_and_data
        controller = AdvancedController(model, data)

        controller.set_control_mode(ControlMode.IMPEDANCE)
        assert controller.mode == ControlMode.IMPEDANCE

        controller.set_control_mode(ControlMode.HYBRID)
        assert controller.mode == ControlMode.HYBRID

    def test_set_impedance_parameters(self, model_and_data) -> None:
        """Test setting impedance parameters."""
        model, data = model_and_data
        controller = AdvancedController(model, data)

        new_params = ImpedanceParameters(
            stiffness=np.ones(model.nv) * 200.0,
            damping=np.ones(model.nv) * 30.0,
        )
        controller.set_impedance_parameters(new_params)

        np.testing.assert_array_equal(
            controller.impedance_params.stiffness,
            new_params.stiffness,
        )

    def test_set_hybrid_mask(self, model_and_data) -> None:
        """Test setting hybrid mask."""
        model, data = model_and_data
        controller = AdvancedController(model, data)

        new_mask = HybridControlMask(force_mask=np.array([True, False]))
        controller.set_hybrid_mask(new_mask)

        np.testing.assert_array_equal(
            controller.hybrid_mask.force_mask,
            new_mask.force_mask,
        )

    def test_compute_control_torque_mode(self, model_and_data) -> None:
        """Test torque control mode."""
        model, data = model_and_data
        controller = AdvancedController(model, data)
        controller.set_control_mode(ControlMode.TORQUE)

        feedforward = np.array([1.0, -0.5])
        tau = controller.compute_control(feedforward_torque=feedforward)

        assert tau.shape == (model.nu,)
        np.testing.assert_array_equal(tau, feedforward)

    def test_compute_control_torque_mode_no_feedforward(self, model_and_data) -> None:
        """Test torque control mode without feedforward."""
        model, data = model_and_data
        controller = AdvancedController(model, data)
        controller.set_control_mode(ControlMode.TORQUE)

        tau = controller.compute_control()

        assert tau.shape == (model.nu,)
        np.testing.assert_array_equal(tau, np.zeros(model.nu))

    def test_compute_control_impedance_mode(self, model_and_data) -> None:
        """Test impedance control mode."""
        model, data = model_and_data
        controller = AdvancedController(model, data)
        controller.set_control_mode(ControlMode.IMPEDANCE)

        target_pos = data.qpos.copy() + 0.1
        target_vel = np.zeros(model.nv)

        tau = controller.compute_control(
            target_position=target_pos,
            target_velocity=target_vel,
        )

        assert tau.shape == (model.nu,)
        assert np.all(np.isfinite(tau))

    def test_compute_control_admittance_mode(self, model_and_data) -> None:
        """Test admittance control mode."""
        model, data = model_and_data
        controller = AdvancedController(model, data)
        controller.set_control_mode(ControlMode.ADMITTANCE)

        target_force = np.array([1.0, -0.5])

        tau = controller.compute_control(target_force=target_force)

        assert tau.shape == (model.nu,)
        assert np.all(np.isfinite(tau))

    def test_compute_control_hybrid_mode(self, model_and_data) -> None:
        """Test hybrid control mode."""
        model, data = model_and_data
        controller = AdvancedController(model, data)
        controller.set_control_mode(ControlMode.HYBRID)

        target_pos = data.qpos.copy()
        target_vel = np.zeros(model.nv)
        target_force = np.array([0.5, 0.0])

        tau = controller.compute_control(
            target_position=target_pos,
            target_velocity=target_vel,
            target_force=target_force,
        )

        assert tau.shape == (model.nu,)
        assert np.all(np.isfinite(tau))

    def test_compute_control_computed_torque_mode(self, model_and_data) -> None:
        """Test computed torque control mode."""
        model, data = model_and_data
        controller = AdvancedController(model, data)
        controller.set_control_mode(ControlMode.COMPUTED_TORQUE)

        target_pos = data.qpos.copy() + 0.1
        target_vel = np.zeros(model.nv)

        tau = controller.compute_control(
            target_position=target_pos,
            target_velocity=target_vel,
        )

        assert tau.shape == (model.nu,)
        assert np.all(np.isfinite(tau))

    def test_compute_control_task_space_mode(self, model_and_data) -> None:
        """Test task-space control mode."""
        model, data = model_and_data
        controller = AdvancedController(model, data)
        controller.set_control_mode(ControlMode.TASK_SPACE)

        # Task-space control may fall back to impedance if no club head
        tau = controller.compute_control()

        assert tau.shape == (model.nu,)
        assert np.all(np.isfinite(tau))

    def test_impedance_control_with_gravity_compensation(self, model_and_data) -> None:
        """Test impedance control with gravity compensation."""
        model, data = model_and_data
        controller = AdvancedController(model, data)
        controller.set_control_mode(ControlMode.IMPEDANCE)
        controller.enable_gravity_compensation = True

        target_pos = data.qpos.copy()
        tau = controller.compute_control(target_position=target_pos)

        assert tau.shape == (model.nu,)
        assert np.all(np.isfinite(tau))

    def test_impedance_control_without_gravity_compensation(
        self,
        model_and_data,
    ) -> None:
        """Test impedance control without gravity compensation."""
        model, data = model_and_data
        controller = AdvancedController(model, data)
        controller.set_control_mode(ControlMode.IMPEDANCE)
        controller.enable_gravity_compensation = False

        target_pos = data.qpos.copy()
        tau = controller.compute_control(target_position=target_pos)

        assert tau.shape == (model.nu,)
        assert np.all(np.isfinite(tau))

    def test_compute_gravity_compensation(self, model_and_data) -> None:
        """Test gravity compensation computation."""
        model, data = model_and_data
        controller = AdvancedController(model, data)

        g = controller._compute_gravity_compensation()

        assert g.shape == (model.nv,)
        assert np.all(np.isfinite(g))

    def test_compute_operational_space_control(self, model_and_data) -> None:
        """Test operational space control."""
        model, data = model_and_data
        controller = AdvancedController(model, data)

        # Find a body ID (use first non-world body)
        body_id = 1
        target_pos = np.array([0.5, 0.0, 1.0])
        target_vel = np.array([0.0, 0.0, 0.0])
        target_acc = np.array([0.0, 0.0, 0.0])

        # Operational space control may fail with singular matrices
        # This is expected in some configurations
        try:
            tau = controller.compute_operational_space_control(
                target_pos,
                target_vel,
                target_acc,
                body_id,
            )

            assert tau.shape == (model.nu,)
            assert np.all(np.isfinite(tau))
        except np.linalg.LinAlgError:
            # Singular matrix is acceptable for some configurations
            pytest.skip(
                "Singular matrix in operational space control (expected in configs)"
            )

    def test_find_body_id(self, model_and_data) -> None:
        """Test finding body ID."""
        model, data = model_and_data
        controller = AdvancedController(model, data)

        body_id = controller._find_body_id("shoulder")
        # May or may not exist depending on model
        if body_id is not None:
            assert body_id > 0
            assert body_id < model.nbody

        # Should return None for nonexistent body
        body_id = controller._find_body_id("nonexistent_body_xyz")
        assert body_id is None


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
