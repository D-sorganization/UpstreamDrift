"""Comprehensive tests for deployment.safety.monitor."""

from __future__ import annotations

import numpy as np
import pytest

from src.deployment.realtime import (
    ControlCommand,
    ControlMode,
    RobotConfig,
    RobotState,
)
from src.deployment.safety.monitor import (
    SafetyLimits,
    SafetyMonitor,
    SafetyStatusLevel,
)


def _cfg(n: int = 3) -> RobotConfig:
    return RobotConfig(
        name="bot",
        n_joints=n,
        joint_limits_lower=np.full(n, -1.0),
        joint_limits_upper=np.full(n, 1.0),
        velocity_limits=np.full(n, 2.0),
        torque_limits=np.full(n, 50.0),
    )


def _state(n: int = 3, **kwargs) -> RobotState:
    defaults = {
        "timestamp": 0.0,
        "joint_positions": np.zeros(n),
        "joint_velocities": np.zeros(n),
        "joint_torques": np.zeros(n),
    }
    defaults.update(kwargs)
    return RobotState(**defaults)


class TestSafetyLimits:
    def test_from_config(self) -> None:
        lim = SafetyLimits.from_config(_cfg())
        assert lim.max_joint_velocity.shape == (3,)
        assert lim.max_joint_torque.shape == (3,)

    def test_from_config_defaults(self) -> None:
        cfg = RobotConfig(name="b", n_joints=3)
        lim = SafetyLimits.from_config(cfg)
        assert np.all(lim.max_joint_velocity == 2.0)
        assert np.all(lim.max_joint_torque == 50.0)


class TestSafetyMonitorCheckState:
    def test_ok(self) -> None:
        m = SafetyMonitor(_cfg())
        st = m.check_state(_state())
        assert st.is_safe
        assert st.level == SafetyStatusLevel.OK

    def test_velocity_violation(self) -> None:
        m = SafetyMonitor(_cfg())
        st = m.check_state(_state(joint_velocities=np.array([10.0, 0, 0])))
        assert not st.is_safe
        assert any("velocity" in v for v in st.violations)

    def test_torque_violation(self) -> None:
        m = SafetyMonitor(_cfg())
        st = m.check_state(_state(joint_torques=np.array([100.0, 0, 0])))
        assert not st.is_safe
        assert any("torque" in v for v in st.violations)

    def test_lower_limit_violation(self) -> None:
        m = SafetyMonitor(_cfg())
        st = m.check_state(_state(joint_positions=np.array([-2.0, 0, 0])))
        assert not st.is_safe
        assert any("Lower" in v for v in st.violations)

    def test_upper_limit_violation(self) -> None:
        m = SafetyMonitor(_cfg())
        st = m.check_state(_state(joint_positions=np.array([2.0, 0, 0])))
        assert not st.is_safe
        assert any("Upper" in v for v in st.violations)

    def test_warning_approaching_limit(self) -> None:
        m = SafetyMonitor(_cfg())
        st = m.check_state(_state(joint_positions=np.array([0.95, 0, 0])))
        assert st.is_safe
        assert st.level == SafetyStatusLevel.WARNING

    def test_emergency_stop_violation(self) -> None:
        m = SafetyMonitor(_cfg())
        m.trigger_emergency_stop()
        st = m.check_state(_state())
        assert not st.is_safe
        assert any("Emergency" in v for v in st.violations)


class TestSafetyMonitorCheckCommand:
    def test_ok(self) -> None:
        m = SafetyMonitor(_cfg())
        cmd = ControlCommand.torque_command(0.0, np.zeros(3))
        st = m.check_command(cmd)
        assert st.is_safe

    def test_torque_command_violation(self) -> None:
        m = SafetyMonitor(_cfg())
        cmd = ControlCommand.torque_command(0.0, np.array([100.0, 0, 0]))
        st = m.check_command(cmd)
        assert not st.is_safe

    def test_position_below_limit(self) -> None:
        m = SafetyMonitor(_cfg())
        cmd = ControlCommand.position_command(0.0, np.array([-2.0, 0, 0]))
        st = m.check_command(cmd)
        assert not st.is_safe
        assert any("below" in v for v in st.violations)

    def test_position_above_limit(self) -> None:
        m = SafetyMonitor(_cfg())
        cmd = ControlCommand.position_command(0.0, np.array([2.0, 0, 0]))
        st = m.check_command(cmd)
        assert not st.is_safe
        assert any("above" in v for v in st.violations)


class TestSafetyMonitorComputeSafe:
    def test_clip_torque(self) -> None:
        m = SafetyMonitor(_cfg())
        cmd = ControlCommand.torque_command(0.0, np.array([100.0, -100.0, 0.0]))
        safe = m.compute_safe_command(cmd, _state())
        assert safe.torque_commands is not None
        assert np.all(np.abs(safe.torque_commands) <= 50.0)

    def test_clip_position(self) -> None:
        m = SafetyMonitor(_cfg())
        cmd = ControlCommand.position_command(0.0, np.array([2.0, -2.0, 0.5]))
        safe = m.compute_safe_command(cmd, _state())
        assert safe.position_targets is not None
        assert safe.position_targets[0] <= 1.0
        assert safe.position_targets[1] >= -1.0

    def test_estop_freezes_position(self) -> None:
        m = SafetyMonitor(_cfg())
        m.trigger_emergency_stop()
        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.IMPEDANCE,
            position_targets=np.array([0.5, 0.5, 0.5]),
            stiffness=np.ones(3),
            damping=np.ones(3),
            feedforward_torque=np.ones(3),
        )
        state = _state(joint_positions=np.array([0.1, 0.1, 0.1]))
        safe = m.compute_safe_command(cmd, state)
        np.testing.assert_array_equal(safe.position_targets, state.joint_positions)
        assert np.all(safe.feedforward_torque == 0)

    def test_speed_override_scales(self) -> None:
        m = SafetyMonitor(_cfg())
        m.set_speed_override(0.5)
        cmd = ControlCommand(
            timestamp=0.0,
            mode=ControlMode.VELOCITY,
            velocity_targets=np.ones(3),
            torque_commands=np.ones(3),
        )
        safe = m.compute_safe_command(cmd, _state())
        np.testing.assert_array_almost_equal(safe.velocity_targets, np.ones(3) * 0.5)
        np.testing.assert_array_almost_equal(safe.torque_commands, np.ones(3) * 0.5)


class TestSafetyMonitorMisc:
    def test_stopping_distance(self) -> None:
        m = SafetyMonitor(_cfg())
        d = m.get_stopping_distance(
            _state(joint_velocities=np.array([2.0, 0, 0])), "ee"
        )
        assert d == pytest.approx(1.0, rel=1e-3)

    def test_set_speed_override_clamps(self) -> None:
        m = SafetyMonitor(_cfg())
        m.set_speed_override(2.0)
        assert m._speed_override == 1.0
        m.set_speed_override(-1.0)
        assert m._speed_override == 0.0

    def test_set_human_nearby(self) -> None:
        m = SafetyMonitor(_cfg())
        m.set_human_nearby(True)
        assert m._human_nearby
        assert m._speed_override <= 0.5

    def test_emergency_stop_cycle(self) -> None:
        m = SafetyMonitor(_cfg())
        assert not m.is_emergency_stopped()
        m.trigger_emergency_stop()
        assert m.is_emergency_stopped()
        assert m._speed_override == 0.0
        m.clear_emergency_stop()
        assert not m.is_emergency_stopped()
        assert m._speed_override == 1.0
