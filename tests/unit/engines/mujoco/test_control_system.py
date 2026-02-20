"""
Unit tests for control system module.

Tests control types, actuator controls, and control system.
"""

import numpy as np
import pytest

# Import directly from module to avoid __init__ imports
from mujoco_humanoid_golf.control_system import (
    POLYNOMIAL_COEFFS_COUNT,
    ActuatorControl,
    ControlSystem,
    ControlType,
)


class TestControlType:
    """Tests for ControlType enum."""

    def test_control_type_values(self) -> None:
        """Test all control types are defined."""
        assert ControlType.CONSTANT.value == "constant"
        assert ControlType.POLYNOMIAL.value == "polynomial"
        assert ControlType.SINE_WAVE.value == "sine_wave"
        assert ControlType.STEP.value == "step"


class TestActuatorControl:
    """Tests for ActuatorControl class."""

    def test_constant_control(self) -> None:
        """Test constant control type."""
        control = ActuatorControl(
            control_type=ControlType.CONSTANT,
            constant_value=5.0,
        )
        assert control.compute_torque(0.0) == 5.0
        assert control.compute_torque(1.0) == 5.0
        assert control.compute_torque(10.0) == 5.0

    def test_constant_control_with_damping(self) -> None:
        """Test constant control with damping."""
        control = ActuatorControl(
            control_type=ControlType.CONSTANT,
            constant_value=10.0,
            damping=0.5,
        )
        # At time 0, velocity 0
        assert control.compute_torque(0.0, 0.0) == 10.0
        # With velocity, damping reduces torque
        torque = control.compute_torque(0.0, 2.0)
        assert torque == 9.0  # 10.0 - 0.5 * 2.0

    def test_polynomial_control(self) -> None:
        """Test polynomial control type."""
        coeffs = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        control = ActuatorControl(
            control_type=ControlType.POLYNOMIAL,
            polynomial_coeffs=coeffs,
        )
        # At t=0: 1.0
        assert control.compute_torque(0.0) == 1.0
        # At t=1: 1 + 2*1 + 3*1^2 = 6.0
        assert control.compute_torque(1.0) == 6.0
        # At t=2: 1 + 2*2 + 3*4 = 17.0
        assert control.compute_torque(2.0) == 17.0

    def test_polynomial_control_invalid_length(self) -> None:
        """Test polynomial control with invalid coefficient length."""
        coeffs = np.array([1.0, 2.0])  # Too short
        with pytest.raises(ValueError):
            ActuatorControl(
                control_type=ControlType.POLYNOMIAL,
                polynomial_coeffs=coeffs,
            )

    def test_sine_wave_control(self) -> None:
        """Test sine wave control type."""
        control = ActuatorControl(
            control_type=ControlType.SINE_WAVE,
            sine_amplitude=5.0,
            sine_frequency=1.0,  # 1 Hz
            sine_phase=0.0,
        )
        # At t=0: 5.0 * sin(0) = 0
        assert abs(control.compute_torque(0.0)) < 1e-10
        # At t=0.25 (quarter period): 5.0 * sin(pi/2) = 5.0
        assert abs(control.compute_torque(0.25) - 5.0) < 1e-10

    def test_sine_wave_control_with_phase(self) -> None:
        """Test sine wave control with phase offset."""
        control = ActuatorControl(
            control_type=ControlType.SINE_WAVE,
            sine_amplitude=5.0,
            sine_frequency=1.0,
            sine_phase=np.pi / 2,  # 90 degree phase
        )
        # At t=0: 5.0 * sin(pi/2) = 5.0
        assert abs(control.compute_torque(0.0) - 5.0) < 1e-10

    def test_step_control(self) -> None:
        """Test step control type."""
        control = ActuatorControl(
            control_type=ControlType.STEP,
            step_time=1.0,
            step_value=10.0,
        )
        # Before step
        assert control.compute_torque(0.5) == 0.0
        # At step time
        assert control.compute_torque(1.0) == 10.0
        # After step
        assert control.compute_torque(2.0) == 10.0

    def test_default_control(self) -> None:
        """Test default control initialization."""
        control = ActuatorControl()
        assert control.control_type == ControlType.CONSTANT
        assert control.constant_value == 0.0
        assert len(control.polynomial_coeffs) == POLYNOMIAL_COEFFS_COUNT
        assert np.all(control.polynomial_coeffs == 0)


class TestControlSystem:
    """Tests for ControlSystem class."""

    def test_control_system_initialization(self) -> None:
        """Test control system initialization."""
        num_actuators = 3
        system = ControlSystem(num_actuators)
        assert len(system.actuator_controls) == num_actuators
        assert all(
            isinstance(ctrl, ActuatorControl) for ctrl in system.actuator_controls
        )

    def test_control_system_compute_control_vector(self) -> None:
        """Test computing control torques for all actuators."""
        num_actuators = 2
        system = ControlSystem(num_actuators)
        # Set different controls
        system.set_constant_value(0, 5.0)
        system.set_constant_value(1, 10.0)

        torques = system.compute_control_vector()
        assert len(torques) == num_actuators
        assert torques[0] == 5.0
        assert torques[1] == 10.0

    def test_control_system_with_velocities(self) -> None:
        """Test control system with joint velocities for damping."""
        num_actuators = 2
        system = ControlSystem(num_actuators)
        system.set_constant_value(0, 10.0)
        system.set_damping(0, 0.5)

        velocities = np.array([2.0, 0.0])
        torques = system.compute_control_vector(velocities)
        # First actuator: 10.0 - 0.5 * 2.0 = 9.0
        assert torques[0] == 9.0
        # Second actuator: default (0.0)
        assert torques[1] == 0.0

    def test_control_system_set_control_type(self) -> None:
        """Test setting control type for specific actuator."""
        num_actuators = 3
        system = ControlSystem(num_actuators)
        system.set_control_type(1, ControlType.POLYNOMIAL)
        system.set_polynomial_coeffs(1, np.ones(POLYNOMIAL_COEFFS_COUNT))
        assert system.actuator_controls[1].control_type == ControlType.POLYNOMIAL

    def test_control_system_time_management(self) -> None:
        """Test time update and advance methods."""
        num_actuators = 1
        system = ControlSystem(num_actuators)
        system.set_control_type(0, ControlType.SINE_WAVE)
        system.set_sine_wave_params(0, 1.0, 1.0, 0.0)

        system.update_time(0.0)
        torque1 = system.compute_control_vector()[0]

        system.advance_time(0.25)  # Quarter period
        torque2 = system.compute_control_vector()[0]

        # Torques should be different
        assert abs(torque1 - torque2) > 0.1

    def test_control_system_reset(self) -> None:
        """Test resetting control system."""
        num_actuators = 1
        system = ControlSystem(num_actuators)
        system.advance_time(1.0)
        assert system.simulation_time == 1.0
        system.reset()
        assert system.simulation_time == 0.0

    def test_control_system_get_actuator_control(self) -> None:
        """Test getting actuator control."""
        num_actuators = 2
        system = ControlSystem(num_actuators)
        control = system.get_actuator_control(0)
        assert isinstance(control, ActuatorControl)

    def test_control_system_get_actuator_control_invalid_index(self) -> None:
        """Test getting actuator control with invalid index."""
        num_actuators = 2
        system = ControlSystem(num_actuators)
        with pytest.raises(IndexError):
            system.get_actuator_control(10)

    def test_control_system_export_import_coefficients(self) -> None:
        """Test exporting and importing polynomial coefficients."""
        num_actuators = 2
        system = ControlSystem(num_actuators)
        coeffs1 = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        system.set_control_type(0, ControlType.POLYNOMIAL)
        system.set_polynomial_coeffs(0, coeffs1)

        exported = system.export_coefficients()
        assert 0 in exported
        np.testing.assert_allclose(exported[0], coeffs1)

        # Import to different actuator
        system.import_coefficients({1: coeffs1})
        assert system.actuator_controls[1].control_type == ControlType.POLYNOMIAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
