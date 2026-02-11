"""Tests for the Control Interface module.

Tests control strategy management, torque application, gain configuration,
and feature registry.

Follows TDD and Design by Contract principles.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.control_interface import (
    ControlInterface,
    ControlStrategy,
)
from src.shared.python.engine_core.mock_engine import MockPhysicsEngine

# ---- Fixtures ----


@pytest.fixture
def mock_engine() -> MockPhysicsEngine:
    """Create and initialize a mock physics engine."""
    engine = MockPhysicsEngine(num_joints=4)
    engine.load_from_string("<mock/>")
    return engine


@pytest.fixture
def ctrl(mock_engine: MockPhysicsEngine) -> ControlInterface:
    """Create a control interface with mock engine."""
    return ControlInterface(mock_engine)


# ---- Basic Tests ----


class TestControlInterfaceBasic:
    """Tests for basic ControlInterface functionality."""

    def test_initialization(self, ctrl: ControlInterface) -> None:
        """Test interface initializes correctly."""
        assert ctrl.n_joints == 4
        assert len(ctrl.joint_names) == 4
        assert ctrl.strategy == ControlStrategy.ZERO

    def test_joint_names(self, ctrl: ControlInterface) -> None:
        """Test joint names are available."""
        names = ctrl.joint_names
        assert all(isinstance(n, str) for n in names)
        assert len(names) == 4

    def test_current_torques(self, ctrl: ControlInterface) -> None:
        """Postcondition: initial torques are zero."""
        torques = ctrl.current_torques
        np.testing.assert_array_equal(torques, np.zeros(4))

    def test_get_state(self, ctrl: ControlInterface) -> None:
        """Test get_state returns complete state dict."""
        state = ctrl.get_state()

        assert "strategy" in state
        assert "n_joints" in state
        assert "joint_names" in state
        assert "torques" in state
        assert "joints" in state
        assert "kp" in state
        assert "kd" in state

        assert state["strategy"] == "zero"
        assert state["n_joints"] == 4
        assert len(state["joints"]) == 4


# ---- Strategy Tests ----


class TestControlStrategies:
    """Tests for control strategy management."""

    def test_set_strategy_string(self, ctrl: ControlInterface) -> None:
        """Test setting strategy by string name."""
        ctrl.set_strategy("pd")
        assert ctrl.strategy == ControlStrategy.PD

    def test_set_strategy_enum(self, ctrl: ControlInterface) -> None:
        """Test setting strategy by enum value."""
        ctrl.set_strategy(ControlStrategy.GRAVITY_COMPENSATION)
        assert ctrl.strategy == ControlStrategy.GRAVITY_COMPENSATION

    def test_set_invalid_strategy(self, ctrl: ControlInterface) -> None:
        """Precondition: strategy must be recognized."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            ctrl.set_strategy("nonexistent_strategy")

    def test_get_available_strategies(self, ctrl: ControlInterface) -> None:
        """Test listing available strategies."""
        strategies = ctrl.get_available_strategies()
        assert len(strategies) >= 6
        names = [s["name"] for s in strategies]
        assert "zero" in names
        assert "direct_torque" in names
        assert "pd" in names
        assert "pid" in names
        assert "gravity_compensation" in names
        assert "computed_torque" in names

    def test_all_strategies_have_descriptions(self, ctrl: ControlInterface) -> None:
        """Test all strategies have descriptions."""
        strategies = ctrl.get_available_strategies()
        for s in strategies:
            assert "name" in s
            assert "description" in s
            assert len(s["description"]) > 0


# ---- Torque Tests ----


class TestTorqueControl:
    """Tests for direct torque control."""

    def test_set_torques(self, ctrl: ControlInterface) -> None:
        """Test setting torques for all joints."""
        ctrl.set_torques([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(ctrl.current_torques, [1.0, 2.0, 3.0, 4.0])

    def test_set_torques_numpy(self, ctrl: ControlInterface) -> None:
        """Test setting torques with numpy array."""
        ctrl.set_torques(np.array([5.0, 6.0, 7.0, 8.0]))
        np.testing.assert_array_equal(ctrl.current_torques, [5.0, 6.0, 7.0, 8.0])

    def test_set_torques_wrong_dimension(self, ctrl: ControlInterface) -> None:
        """Precondition: torque dimension must match joints."""
        with pytest.raises(ValueError, match="Expected 4 torques"):
            ctrl.set_torques([1.0, 2.0])

    def test_set_single_joint_torque_by_index(self, ctrl: ControlInterface) -> None:
        """Test setting torque for a single joint by index."""
        ctrl.set_joint_torque(0, 10.0)
        assert ctrl.current_torques[0] == 10.0
        assert ctrl.current_torques[1] == 0.0

    def test_set_single_joint_torque_by_name(self, ctrl: ControlInterface) -> None:
        """Test setting torque for a single joint by name."""
        ctrl.set_joint_torque("joint_2", 15.0)
        assert ctrl.current_torques[2] == 15.0

    def test_set_invalid_joint_index(self, ctrl: ControlInterface) -> None:
        """Precondition: joint index must be valid."""
        with pytest.raises(ValueError, match="out of range"):
            ctrl.set_joint_torque(10, 5.0)

    def test_set_invalid_joint_name(self, ctrl: ControlInterface) -> None:
        """Precondition: joint name must exist."""
        with pytest.raises(ValueError, match="not found"):
            ctrl.set_joint_torque("nonexistent_joint", 5.0)

    def test_torque_clipping(self, ctrl: ControlInterface) -> None:
        """Postcondition: torques are clipped to limits."""
        ctrl.set_torques([500.0, -500.0, 50.0, -50.0])
        torques = ctrl.current_torques
        assert torques[0] == 100.0  # Clipped to default limit
        assert torques[1] == -100.0


# ---- Gain Tests ----


class TestGainConfiguration:
    """Tests for PD/PID gain configuration."""

    def test_set_scalar_gains(self, ctrl: ControlInterface) -> None:
        """Test setting scalar gains (applied to all joints)."""
        ctrl.set_gains(kp=200.0, kd=20.0)
        state = ctrl.get_state()
        assert all(g == 200.0 for g in state["kp"])
        assert all(g == 20.0 for g in state["kd"])

    def test_set_per_joint_gains(self, ctrl: ControlInterface) -> None:
        """Test setting per-joint gains."""
        kp = np.array([100, 200, 300, 400], dtype=float)
        ctrl.set_gains(kp=kp)
        state = ctrl.get_state()
        np.testing.assert_array_equal(state["kp"], kp.tolist())

    def test_set_target_positions(self, ctrl: ControlInterface) -> None:
        """Test setting target positions."""
        targets = [0.1, 0.2, 0.3, 0.4]
        ctrl.set_target_positions(targets)
        state = ctrl.get_state()
        assert state["target_positions"] == targets

    def test_set_target_velocities(self, ctrl: ControlInterface) -> None:
        """Test setting target velocities."""
        targets = [1.0, 2.0, 3.0, 4.0]
        ctrl.set_target_velocities(targets)
        state = ctrl.get_state()
        assert state["target_velocities"] == targets


# ---- Control Computation Tests ----


class TestControlComputation:
    """Tests for control torque computation."""

    def test_zero_strategy(self, ctrl: ControlInterface) -> None:
        """Test zero strategy produces zero torques."""
        ctrl.set_strategy("zero")
        torques = ctrl.compute_control()
        np.testing.assert_array_equal(torques, np.zeros(4))

    def test_direct_torque_strategy(self, ctrl: ControlInterface) -> None:
        """Test direct torque strategy passes through set torques."""
        ctrl.set_strategy("direct_torque")
        ctrl.set_torques([1.0, 2.0, 3.0, 4.0])
        torques = ctrl.compute_control()
        np.testing.assert_array_equal(torques, [1.0, 2.0, 3.0, 4.0])

    def test_pd_strategy(self, ctrl: ControlInterface) -> None:
        """Test PD control computes non-zero torques with error."""
        ctrl.set_strategy("pd")
        ctrl.set_gains(kp=100.0, kd=10.0)
        ctrl.set_target_positions([1.0, 1.0, 1.0, 1.0])
        torques = ctrl.compute_control()

        # With target=[1,1,1,1] and current=[0,0,0,0], should have positive torques
        assert all(t > 0 for t in torques)

    def test_gravity_compensation(self, ctrl: ControlInterface) -> None:
        """Test gravity compensation produces non-zero torques."""
        ctrl.set_strategy("gravity_compensation")
        torques = ctrl.compute_control()
        # Mock engine has gravity on first joint
        assert torques[0] != 0.0

    def test_computed_torque_strategy(self, ctrl: ControlInterface) -> None:
        """Test computed torque control."""
        ctrl.set_strategy("computed_torque")
        ctrl.set_gains(kp=100.0, kd=10.0)
        ctrl.set_target_positions([0.5, 0.5, 0.5, 0.5])
        torques = ctrl.compute_control()
        assert np.all(np.isfinite(torques))

    def test_impedance_strategy(self, ctrl: ControlInterface) -> None:
        """Test impedance control includes gravity compensation."""
        ctrl.set_strategy("impedance")
        ctrl.set_gains(kp=100.0, kd=10.0)
        ctrl.set_target_positions([0.5, 0.5, 0.5, 0.5])
        torques = ctrl.compute_control()
        assert np.all(np.isfinite(torques))

    def test_torque_history(self, ctrl: ControlInterface) -> None:
        """Test that torque history is recorded."""
        ctrl.set_strategy("pd")
        ctrl.set_target_positions([0.5, 0.5, 0.5, 0.5])

        for _ in range(5):
            ctrl.compute_control()

        history = ctrl.get_torque_history()
        assert history.shape == (5, 4)


# ---- Reset Tests ----


class TestControlReset:
    """Tests for control interface reset."""

    def test_reset(self, ctrl: ControlInterface) -> None:
        """Test reset clears all state."""
        ctrl.set_strategy("pd")
        ctrl.set_torques([1.0, 2.0, 3.0, 4.0])
        ctrl.set_target_positions([0.5, 0.5, 0.5, 0.5])
        ctrl.compute_control()

        ctrl.reset()

        np.testing.assert_array_equal(ctrl.current_torques, np.zeros(4))
        state = ctrl.get_state()
        assert state["target_positions"] is None
        assert len(ctrl.get_torque_history()) == 0
