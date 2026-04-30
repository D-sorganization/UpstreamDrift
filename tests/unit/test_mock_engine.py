"""Tests for MockPhysicsEngine.

Issue #1744: Raise test coverage.
Issue #1741: Populate test directories.
"""

import numpy as np
import pytest

from src.shared.python.engine_core.mock_engine import MockPhysicsEngine, get_mock_engine


class TestMockPhysicsEngineInit:
    """Tests for MockPhysicsEngine initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        engine = MockPhysicsEngine()
        assert engine.num_joints == 7
        assert engine.timestep == 0.001
        assert engine.model_name == "mock_golfer"
        assert engine._time == 0.0
        assert not engine._is_loaded

    def test_custom_joints(self) -> None:
        """Test initialization with custom joint count."""
        engine = MockPhysicsEngine(num_joints=3)
        assert engine.num_joints == 3
        assert len(engine._positions) == 3
        assert len(engine._velocities) == 3

    def test_custom_timestep(self) -> None:
        """Test initialization with custom timestep."""
        engine = MockPhysicsEngine(timestep=0.01)
        assert engine.get_timestep() == 0.01

    def test_factory_function(self) -> None:
        """Test get_mock_engine factory."""
        engine = get_mock_engine()
        assert isinstance(engine, MockPhysicsEngine)
        assert engine.num_joints == 7


class TestMockPhysicsEngineModel:
    """Tests for model loading."""

    def test_load_model(self) -> None:
        """Test loading a model sets state."""
        engine = MockPhysicsEngine()
        engine.load_model("/path/to/model.urdf")
        assert engine._is_loaded
        assert engine.model_name == "/path/to/model.urdf"

    def test_load_from_path(self) -> None:
        """Test backward-compatible load_from_path."""
        engine = MockPhysicsEngine()
        engine.load_from_path("test.xml")
        assert engine._is_loaded

    def test_load_from_string(self) -> None:
        """Test loading from string content."""
        engine = MockPhysicsEngine()
        engine.load_from_string("<robot/>", extension=".urdf")
        assert engine._is_loaded
        assert engine.model_name == "mock_model"


class TestMockPhysicsEngineSimulation:
    """Tests for simulation stepping."""

    def test_step_advances_time(self) -> None:
        """Test that step() advances simulation time."""
        engine = MockPhysicsEngine()
        engine.step()
        assert engine.get_simulation_time() == pytest.approx(0.001)

    def test_step_custom_dt(self) -> None:
        """Test step with custom timestep."""
        engine = MockPhysicsEngine()
        engine.step(dt=0.1)
        assert engine.get_simulation_time() == pytest.approx(0.1)

    def test_step_integrates_torques(self) -> None:
        """Test that torques produce motion via Euler integration."""
        engine = MockPhysicsEngine(num_joints=1)
        engine._torques = np.array([10.0])
        engine.step(dt=0.01)

        # With damping=0.1, mass=1: acc = (10 - 0.1*0) / 1 = 10
        # vel = 0 + 10*0.01 = 0.1
        # pos = 0 + 0.1*0.01 = 0.001
        assert engine._velocities[0] == pytest.approx(0.1)
        assert engine._positions[0] == pytest.approx(0.001)

    def test_reset(self) -> None:
        """Test resetting to initial state."""
        engine = MockPhysicsEngine()
        engine._torques = np.ones(7)
        engine.step(dt=0.01)
        engine.step(dt=0.01)

        engine.reset()
        assert engine._time == 0.0
        assert np.all(engine._positions == 0)
        assert np.all(engine._velocities == 0)
        assert np.all(engine._torques == 0)


class TestMockPhysicsEngineState:
    """Tests for state manipulation."""

    def test_get_state_returns_copies(self) -> None:
        """Test that get_state returns copies, not references."""
        engine = MockPhysicsEngine()
        pos, vel = engine.get_state()
        pos[0] = 999.0
        assert engine._positions[0] == 0.0

    def test_get_state_dict(self) -> None:
        """Test get_state_dict returns expected keys."""
        engine = MockPhysicsEngine()
        state = engine.get_state_dict()
        assert "time" in state
        assert "positions" in state
        assert "velocities" in state
        assert "accelerations" in state
        assert "torques" in state
        assert "is_loaded" in state

    def test_set_state(self) -> None:
        """Test setting full state."""
        engine = MockPhysicsEngine(num_joints=3)
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.1, 0.2, 0.3])
        engine.set_state(pos, vel)
        np.testing.assert_array_equal(engine._positions, pos)
        np.testing.assert_array_equal(engine._velocities, vel)

    def test_set_joint_positions(self) -> None:
        """Test setting joint positions only."""
        engine = MockPhysicsEngine(num_joints=2)
        engine.set_joint_positions(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(engine._positions, [1.0, 2.0])

    def test_set_joint_velocities(self) -> None:
        """Test setting joint velocities only."""
        engine = MockPhysicsEngine(num_joints=2)
        engine.set_joint_velocities(np.array([0.5, 1.5]))
        np.testing.assert_array_equal(engine._velocities, [0.5, 1.5])


class TestMockPhysicsEngineControl:
    """Tests for control methods."""

    def test_apply_torque_named_joint(self) -> None:
        """Test applying torque to a named joint."""
        engine = MockPhysicsEngine(num_joints=5)
        engine.apply_torque("joint_2", 5.0)
        assert engine._torques[2] == 5.0

    def test_apply_torque_invalid_joint(self) -> None:
        """Test that invalid joint name doesn't crash."""
        engine = MockPhysicsEngine(num_joints=3)
        engine.apply_torque("unknown_joint", 1.0)
        # Should not raise, just log warning

    def test_set_control(self) -> None:
        """Test setting control torques for all joints."""
        engine = MockPhysicsEngine(num_joints=3)
        engine.set_control([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(engine._torques, [1.0, 2.0, 3.0])

    def test_set_control_pads_short_input(self) -> None:
        """Test that short control vectors are zero-padded."""
        engine = MockPhysicsEngine(num_joints=5)
        engine.set_control([1.0, 2.0])
        assert len(engine._torques) == 5
        assert engine._torques[0] == 1.0
        assert engine._torques[4] == 0.0


class TestMockPhysicsEngineBiomechanics:
    """Tests for biomechanics methods."""

    def test_compute_mass_matrix(self) -> None:
        """Test mass matrix is identity."""
        engine = MockPhysicsEngine(num_joints=3)
        M = engine.compute_mass_matrix()
        np.testing.assert_array_equal(M, np.eye(3))

    def test_compute_bias_forces(self) -> None:
        """Test bias forces are zero."""
        engine = MockPhysicsEngine(num_joints=3)
        bias = engine.compute_bias_forces()
        np.testing.assert_array_equal(bias, np.zeros(3))

    def test_compute_gravity_forces(self) -> None:
        """Test gravity forces affect first joint."""
        engine = MockPhysicsEngine(num_joints=3)
        g = engine.compute_gravity_forces()
        assert g[0] < 0  # First joint feels gravity
        assert g[1] == 0.0
        assert g[2] == 0.0

    def test_compute_inverse_dynamics(self) -> None:
        """Test inverse dynamics: tau = M*qacc + bias."""
        engine = MockPhysicsEngine(num_joints=3)
        qacc = np.array([1.0, 2.0, 3.0])
        tau = engine.compute_inverse_dynamics(qacc)
        # M=I, bias=0, so tau = qacc
        np.testing.assert_array_equal(tau, qacc)

    def test_compute_contact_forces(self) -> None:
        """Test contact forces are zero for mock."""
        engine = MockPhysicsEngine()
        forces = engine.compute_contact_forces()
        assert len(forces) == 3
        np.testing.assert_array_equal(forces, np.zeros(3))

    def test_compute_jacobian(self) -> None:
        """Test Jacobian returns correct shapes."""
        engine = MockPhysicsEngine(num_joints=5)
        jac = engine.compute_jacobian("test_body")
        assert jac is not None
        assert jac["linear"].shape == (3, 5)
        assert jac["angular"].shape == (3, 5)

    def test_get_body_position(self) -> None:
        """Test body position returns 3D vector."""
        engine = MockPhysicsEngine()
        pos = engine.get_body_position("any_body")
        assert len(pos) == 3

    def test_get_body_velocity(self) -> None:
        """Test body velocity returns 6D vector."""
        engine = MockPhysicsEngine()
        vel = engine.get_body_velocity("any_body")
        assert len(vel) == 6

    def test_get_full_state(self) -> None:
        """Test get_full_state returns expected keys."""
        engine = MockPhysicsEngine()
        state = engine.get_full_state()
        assert "q" in state
        assert "v" in state
        assert "t" in state
        assert "M" in state
        assert state["M"].shape == (7, 7)

    def test_forward_computes_accelerations(self) -> None:
        """Test forward() updates accelerations without advancing time."""
        engine = MockPhysicsEngine(num_joints=1)
        engine._torques = np.array([5.0])
        t_before = engine.get_time()

        engine.forward()

        assert engine._accelerations[0] != 0
        assert engine.get_time() == t_before  # Time should NOT advance

    def test_get_joint_names(self) -> None:
        """Test joint name generation."""
        engine = MockPhysicsEngine(num_joints=3)
        names = engine.get_joint_names()
        assert names == ["joint_0", "joint_1", "joint_2"]
