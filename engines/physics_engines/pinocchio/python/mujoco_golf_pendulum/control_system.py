"""Advanced control system for golf swing simulation.

Supports multiple control types including constant, polynomial, and time-based controls.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

# Polynomial order constant (6th order = 7 coefficients)
POLYNOMIAL_ORDER = 6
POLYNOMIAL_COEFFS_COUNT = POLYNOMIAL_ORDER + 1  # 7 coefficients


class ControlType(Enum):
    """Types of control inputs."""

    CONSTANT = "constant"
    POLYNOMIAL = "polynomial"
    SINE_WAVE = "sine_wave"
    STEP = "step"


class ActuatorControl:
    """Control configuration for a single actuator.

    Supports multiple control types with parameters for each type.
    """

    def __init__(
        self,
        control_type: ControlType = ControlType.CONSTANT,
        constant_value: float = 0.0,
        polynomial_coeffs: np.ndarray | None = None,
        damping: float = 0.0,
        sine_amplitude: float = 0.0,
        sine_frequency: float = 1.0,
        sine_phase: float = 0.0,
        step_time: float = 0.0,
        step_value: float = 0.0,
    ) -> None:
        """Initialize actuator control.

        Args:
            control_type: Type of control input
            constant_value: Constant torque value (for CONSTANT type)
            polynomial_coeffs: Coefficients for 6th order polynomial\
                [c0, c1, c2, c3, c4, c5, c6]
                              where output = c0 + c1*t + c2*t^2 + ... + c6*t^6
            damping: Damping coefficient (applied as -damping * velocity)
            sine_amplitude: Amplitude for sine wave control
            sine_frequency: Frequency for sine wave control (Hz)
            sine_phase: Phase offset for sine wave control (radians)
            step_time: Time at which step occurs (for STEP type)
            step_value: Value after step (for STEP type)
        """
        self.control_type = control_type
        self.constant_value = constant_value
        self.damping = damping

        # Polynomial coefficients (6th order: 7 coefficients)
        if polynomial_coeffs is None:
            self.polynomial_coeffs = np.zeros(POLYNOMIAL_COEFFS_COUNT, dtype=np.float64)
        else:
            if len(polynomial_coeffs) != POLYNOMIAL_COEFFS_COUNT:
                msg = (
                    f"Polynomial coefficients must have length \
                        {POLYNOMIAL_COEFFS_COUNT} "
                    f"for {POLYNOMIAL_ORDER}th order polynomial"
                )
                raise ValueError(
                    msg,
                )
            self.polynomial_coeffs = np.array(
                polynomial_coeffs,
                dtype=np.float64,
            ).reshape(-1)

        # Sine wave parameters
        self.sine_amplitude = sine_amplitude
        self.sine_frequency = sine_frequency
        self.sine_phase = sine_phase

        # Step function parameters
        self.step_time = step_time
        self.step_value = step_value

    def compute_torque(self, time: float, velocity: float = 0.0) -> float:
        """Compute control torque at given time.

        Args:
            time: Current simulation time (seconds)
            velocity: Current joint velocity (for damping)

        Returns:
            Control torque value
        """
        # Base torque from control type
        if self.control_type == ControlType.CONSTANT:
            base_torque = self.constant_value
        elif self.control_type == ControlType.POLYNOMIAL:
            # Evaluate polynomial: c0 + c1*t + c2*t^2 + ... + c6*t^6
            t_powers = np.array(
                [1.0, time, time**2, time**3, time**4, time**5, time**6],
            )
            base_torque = np.dot(self.polynomial_coeffs, t_powers)
        elif self.control_type == ControlType.SINE_WAVE:
            base_torque = self.sine_amplitude * np.sin(
                2 * np.pi * self.sine_frequency * time + self.sine_phase,
            )
        elif self.control_type == ControlType.STEP:
            base_torque = self.step_value if time >= self.step_time else 0.0
        else:
            base_torque = 0.0

        # Add damping (always applied if damping > 0)
        damping_torque = -self.damping * velocity

        return base_torque + damping_torque

    def get_polynomial_coeffs(self) -> np.ndarray:
        """Get polynomial coefficients."""
        return self.polynomial_coeffs.copy()

    def set_polynomial_coeffs(self, coeffs: np.ndarray) -> None:
        """Set polynomial coefficients."""
        if len(coeffs) != POLYNOMIAL_COEFFS_COUNT:
            msg = f"Polynomial coefficients must have length {POLYNOMIAL_COEFFS_COUNT}"
            raise ValueError(msg)
        self.polynomial_coeffs[:] = coeffs


class ControlSystem:
    """Advanced control system managing all actuators.

    Supports multiple control types per actuator, time-based evaluation,
    and damping controls.
    """

    def __init__(self, num_actuators: int) -> None:
        """Initialize control system.

        Args:
            num_actuators: Number of actuators in the system
        """
        self.num_actuators = num_actuators
        self.actuator_controls: list[ActuatorControl] = [
            ActuatorControl() for _ in range(num_actuators)
        ]
        self.simulation_time = 0.0
        self.time_step = 0.001  # Default timestep

    def set_control_type(self, actuator_index: int, control_type: ControlType) -> None:
        """Set control type for an actuator.

        Args:
            actuator_index: Index of actuator (0-based)
            control_type: Type of control to use
        """
        if 0 <= actuator_index < self.num_actuators:
            self.actuator_controls[actuator_index].control_type = control_type

    def set_constant_value(self, actuator_index: int, value: float) -> None:
        """Set constant value for an actuator.

        Args:
            actuator_index: Index of actuator
            value: Constant torque value
        """
        if 0 <= actuator_index < self.num_actuators:
            self.actuator_controls[actuator_index].constant_value = value

    def set_polynomial_coeffs(self, actuator_index: int, coeffs: np.ndarray) -> None:
        """Set polynomial coefficients for an actuator.

        Args:
            actuator_index: Index of actuator
            coeffs: Array of POLYNOMIAL_COEFFS_COUNT coefficients\
                [c0, c1, c2, c3, c4, c5, c6]
        """
        if 0 <= actuator_index < self.num_actuators:
            self.actuator_controls[actuator_index].set_polynomial_coeffs(coeffs)

    def set_damping(self, actuator_index: int, damping: float) -> None:
        """Set damping coefficient for an actuator.

        Args:
            actuator_index: Index of actuator
            damping: Damping coefficient (applied as -damping * velocity)
        """
        if 0 <= actuator_index < self.num_actuators:
            self.actuator_controls[actuator_index].damping = damping

    def set_sine_wave_params(
        self,
        actuator_index: int,
        amplitude: float,
        frequency: float,
        phase: float = 0.0,
    ) -> None:
        """Set sine wave parameters for an actuator.

        Args:
            actuator_index: Index of actuator
            amplitude: Amplitude of sine wave
            frequency: Frequency in Hz
            phase: Phase offset in radians
        """
        if 0 <= actuator_index < self.num_actuators:
            control = self.actuator_controls[actuator_index]
            control.sine_amplitude = amplitude
            control.sine_frequency = frequency
            control.sine_phase = phase

    def set_step_params(
        self,
        actuator_index: int,
        step_time: float,
        step_value: float,
    ) -> None:
        """Set step function parameters for an actuator.

        Args:
            actuator_index: Index of actuator
            step_time: Time at which step occurs
            step_value: Value after step
        """
        if 0 <= actuator_index < self.num_actuators:
            control = self.actuator_controls[actuator_index]
            control.step_time = step_time
            control.step_value = step_value

    def compute_control_vector(
        self,
        velocities: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute control torques for all actuators.

        Args:
            velocities: Current joint velocities [nv] (optional, for damping)

        Returns:
            Control torque vector [nu]
        """
        control_vector = np.zeros(self.num_actuators, dtype=np.float64)

        for i, control in enumerate(self.actuator_controls):
            # Get velocity for this actuator (if provided)
            vel = 0.0
            if velocities is not None and i < len(velocities):
                # Map actuator index to joint DOF (simplified: assume 1-to-1 for now)
                vel = velocities[i]

            control_vector[i] = control.compute_torque(self.simulation_time, vel)

        return control_vector

    def update_time(self, time: float) -> None:
        """Update simulation time.

        Args:
            time: Current simulation time
        """
        self.simulation_time = time

    def advance_time(self, dt: float) -> None:
        """Advance simulation time by dt.

        Args:
            dt: Time step
        """
        self.simulation_time += dt

    def reset(self) -> None:
        """Reset control system (reset time to 0)."""
        self.simulation_time = 0.0

    def get_actuator_control(self, actuator_index: int) -> ActuatorControl:
        """Get control configuration for an actuator.

        Args:
            actuator_index: Index of actuator

        Returns:
            ActuatorControl object
        """
        if 0 <= actuator_index < self.num_actuators:
            return self.actuator_controls[actuator_index]
        msg = f"Actuator index {actuator_index} out of range [0, {self.num_actuators})"
        raise IndexError(
            msg,
        )

    def export_coefficients(self) -> dict:
        """Export all polynomial coefficients for optimization.

        Returns:
            Dictionary mapping actuator indices to coefficient arrays
        """
        coeffs_dict = {}
        for i, control in enumerate(self.actuator_controls):
            if control.control_type == ControlType.POLYNOMIAL:
                coeffs_dict[i] = control.get_polynomial_coeffs().copy()
        return coeffs_dict

    def import_coefficients(self, coeffs_dict: dict) -> None:
        """Import polynomial coefficients from optimization.

        Args:
            coeffs_dict: Dictionary mapping actuator indices to coefficient arrays
        """
        for actuator_index, coeffs in coeffs_dict.items():
            if 0 <= actuator_index < self.num_actuators:
                self.set_polynomial_coeffs(actuator_index, coeffs)
                # Ensure control type is set to polynomial
                self.actuator_controls[actuator_index].control_type = (
                    ControlType.POLYNOMIAL
                )
