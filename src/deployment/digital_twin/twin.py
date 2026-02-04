"""Digital Twin of a real robot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.deployment.realtime import RealTimeController
    from src.engines.protocols import PhysicsEngineProtocol
    from src.robotics.contact import ContactState


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""

    COLLISION = "collision"
    SLIP = "slip"
    STUCK = "stuck"
    MODEL_MISMATCH = "model_mismatch"
    SENSOR_FAULT = "sensor_fault"
    COMMUNICATION_ERROR = "communication_error"
    JOINT_LIMIT = "joint_limit"
    VELOCITY_LIMIT = "velocity_limit"
    TORQUE_LIMIT = "torque_limit"


@dataclass
class AnomalyReport:
    """Report of detected anomaly.

    Attributes:
        timestamp: Time of anomaly detection.
        anomaly_type: Type of anomaly detected.
        severity: Severity level (0.0 to 1.0).
        affected_joints: List of joint indices affected.
        description: Human-readable description.
        recommended_action: Suggested response action.
        confidence: Confidence in detection (0.0 to 1.0).
        raw_data: Raw sensor/simulation data for debugging.
    """

    timestamp: float
    anomaly_type: AnomalyType
    severity: float
    affected_joints: list[int]
    description: str
    recommended_action: str
    confidence: float = 0.9
    raw_data: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate anomaly report."""
        if not 0 <= self.severity <= 1:
            raise ValueError("severity must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


class DigitalTwin:
    """Digital twin of a real robot.

    Maintains a synchronized simulation that mirrors the real
    robot's state, enabling:
    - State prediction and planning
    - Anomaly detection
    - Contact estimation
    - Model calibration

    Attributes:
        sim: Physics engine for simulation.
        real: Real-time controller interface.
        sync_error: Current synchronization error.
    """

    def __init__(
        self,
        sim_engine: "PhysicsEngineProtocol",
        real_interface: "RealTimeController",
    ) -> None:
        """Initialize digital twin.

        Args:
            sim_engine: Physics engine for simulation.
            real_interface: Real robot interface.
        """
        self.sim = sim_engine
        self.real = real_interface
        self._sync_error = 0.0
        self._anomaly_threshold = 0.1
        self._anomaly_history: list[AnomalyReport] = []

        # State estimation
        from src.deployment.digital_twin.estimator import StateEstimator
        self._state_estimator = StateEstimator()

    @property
    def sync_error(self) -> float:
        """Current synchronization error."""
        return self._sync_error

    def synchronize(self) -> float:
        """Synchronize simulation state with real robot.

        Updates the simulation state to match the real robot's
        current state.

        Returns:
            Synchronization error (norm of state difference).
        """
        real_state = self.real.get_last_state()
        if real_state is None:
            return self._sync_error

        # Get current simulation state
        if hasattr(self.sim, "get_joint_positions"):
            sim_q = self.sim.get_joint_positions()
        else:
            sim_q = np.zeros_like(real_state.joint_positions)

        if hasattr(self.sim, "get_joint_velocities"):
            sim_v = self.sim.get_joint_velocities()
        else:
            sim_v = np.zeros_like(real_state.joint_velocities)

        # Compute error
        pos_error = np.linalg.norm(real_state.joint_positions - sim_q)
        vel_error = np.linalg.norm(real_state.joint_velocities - sim_v)
        self._sync_error = float(pos_error + 0.1 * vel_error)

        # Update simulation state
        if hasattr(self.sim, "set_joint_positions"):
            self.sim.set_joint_positions(real_state.joint_positions)
        if hasattr(self.sim, "set_joint_velocities"):
            self.sim.set_joint_velocities(real_state.joint_velocities)

        return self._sync_error

    def predict(
        self,
        horizon: float,
        control_sequence: NDArray[np.floating],
        dt: float = 0.001,
    ) -> NDArray[np.floating]:
        """Predict future trajectory given control sequence.

        Args:
            horizon: Prediction horizon in seconds.
            control_sequence: Control inputs (n_steps, n_controls).
            dt: Simulation timestep.

        Returns:
            Predicted state trajectory (n_steps+1, n_states).
        """
        n_steps = int(horizon / dt)
        n_steps = min(n_steps, len(control_sequence))

        # Save current state
        if hasattr(self.sim, "get_joint_positions"):
            initial_q = self.sim.get_joint_positions().copy()
        else:
            initial_q = np.zeros(7)

        if hasattr(self.sim, "get_joint_velocities"):
            initial_v = self.sim.get_joint_velocities().copy()
        else:
            initial_v = np.zeros(7)

        n_dof = len(initial_q)
        trajectory = np.zeros((n_steps + 1, 2 * n_dof))
        trajectory[0] = np.concatenate([initial_q, initial_v])

        # Roll out simulation
        for i in range(n_steps):
            # Apply control
            if hasattr(self.sim, "set_joint_torques"):
                self.sim.set_joint_torques(control_sequence[i])

            # Step simulation
            if hasattr(self.sim, "step"):
                self.sim.step(dt)

            # Record state
            if hasattr(self.sim, "get_joint_positions"):
                q = self.sim.get_joint_positions()
            else:
                q = np.zeros(n_dof)

            if hasattr(self.sim, "get_joint_velocities"):
                v = self.sim.get_joint_velocities()
            else:
                v = np.zeros(n_dof)

            trajectory[i + 1] = np.concatenate([q, v])

        # Restore initial state
        if hasattr(self.sim, "set_joint_positions"):
            self.sim.set_joint_positions(initial_q)
        if hasattr(self.sim, "set_joint_velocities"):
            self.sim.set_joint_velocities(initial_v)

        return trajectory

    def detect_anomaly(self) -> AnomalyReport | None:
        """Detect discrepancy between simulation and real robot.

        Compares simulation predictions with actual robot behavior
        to detect anomalies like collisions, slipping, or model errors.

        Returns:
            Anomaly report if detected, None otherwise.
        """
        real_state = self.real.get_last_state()
        if real_state is None:
            return None

        # Get simulation state
        if hasattr(self.sim, "get_joint_positions"):
            sim_q = self.sim.get_joint_positions()
        else:
            return None

        if hasattr(self.sim, "get_joint_velocities"):
            sim_v = self.sim.get_joint_velocities()
        else:
            return None

        # Compute discrepancies
        pos_error = np.abs(real_state.joint_positions - sim_q)
        vel_error = np.abs(real_state.joint_velocities - sim_v)

        # Check for anomalies
        anomaly = None

        # Position mismatch
        if np.any(pos_error > self._anomaly_threshold):
            affected = list(np.where(pos_error > self._anomaly_threshold)[0])
            severity = float(np.max(pos_error) / 0.5)  # Normalize
            severity = min(1.0, severity)

            anomaly = AnomalyReport(
                timestamp=real_state.timestamp,
                anomaly_type=AnomalyType.MODEL_MISMATCH,
                severity=severity,
                affected_joints=affected,
                description=(
                    f"Position mismatch detected on joints {affected}. "
                    f"Max error: {np.max(pos_error):.4f} rad"
                ),
                recommended_action="Check for collisions or external forces",
                confidence=0.8,
                raw_data={
                    "position_error": pos_error.tolist(),
                    "velocity_error": vel_error.tolist(),
                },
            )

        # Velocity spike (potential collision)
        if hasattr(self.sim, "get_joint_torques"):
            real_torque = real_state.joint_torques
            sim_torque = self.sim.get_joint_torques()
            torque_error = np.abs(real_torque - sim_torque)

            if np.any(torque_error > 10):  # 10 Nm threshold
                affected = list(np.where(torque_error > 10)[0])
                anomaly = AnomalyReport(
                    timestamp=real_state.timestamp,
                    anomaly_type=AnomalyType.COLLISION,
                    severity=0.8,
                    affected_joints=affected,
                    description=f"Torque spike on joints {affected}",
                    recommended_action="Stop motion and check for obstacles",
                    confidence=0.7,
                )

        if anomaly:
            self._anomaly_history.append(anomaly)

        return anomaly

    def get_estimated_contacts(self) -> list[dict[str, Any]]:
        """Estimate contact states from force measurements.

        Uses force/torque sensor data and inverse dynamics to
        estimate contact locations and forces.

        Returns:
            List of estimated contact states.
        """
        real_state = self.real.get_last_state()
        if real_state is None:
            return []

        contacts = []

        # Estimate from F/T sensors
        if real_state.ft_wrenches:
            for sensor_name, wrench in real_state.ft_wrenches.items():
                force_mag = np.linalg.norm(wrench[:3])
                if force_mag > 1.0:  # Contact threshold
                    contacts.append({
                        "sensor": sensor_name,
                        "force": wrench[:3].tolist(),
                        "torque": wrench[3:].tolist(),
                        "magnitude": float(force_mag),
                    })

        # Estimate from contact states
        if real_state.contact_states:
            for i, in_contact in enumerate(real_state.contact_states):
                if in_contact:
                    contacts.append({
                        "contact_id": i,
                        "in_contact": True,
                    })

        return contacts

    def compute_virtual_forces(self) -> NDArray[np.floating]:
        """Compute forces that would explain state discrepancy.

        Uses inverse dynamics to estimate external forces
        acting on the robot.

        Returns:
            Estimated external forces/torques.
        """
        real_state = self.real.get_last_state()
        if real_state is None:
            return np.zeros(6)

        # Simple force estimation from torque mismatch
        if hasattr(self.sim, "get_joint_torques"):
            sim_torque = self.sim.get_joint_torques()
            real_torque = real_state.joint_torques
            torque_diff = real_torque - sim_torque

            # Map joint torques to end-effector wrench (simplified)
            # Full implementation would use Jacobian transpose
            virtual_force = np.zeros(6)
            if len(torque_diff) >= 6:
                virtual_force = torque_diff[:6]

            return virtual_force

        return np.zeros(6)

    def get_anomaly_history(
        self,
        max_age: float | None = None,
    ) -> list[AnomalyReport]:
        """Get history of detected anomalies.

        Args:
            max_age: Maximum age in seconds (None for all).

        Returns:
            List of anomaly reports.
        """
        if max_age is None:
            return self._anomaly_history.copy()

        real_state = self.real.get_last_state()
        if real_state is None:
            return self._anomaly_history.copy()

        current_time = real_state.timestamp
        return [
            a for a in self._anomaly_history
            if current_time - a.timestamp <= max_age
        ]

    def clear_anomaly_history(self) -> None:
        """Clear anomaly detection history."""
        self._anomaly_history.clear()

    def set_anomaly_threshold(self, threshold: float) -> None:
        """Set threshold for anomaly detection.

        Args:
            threshold: Position error threshold in radians.
        """
        self._anomaly_threshold = threshold
