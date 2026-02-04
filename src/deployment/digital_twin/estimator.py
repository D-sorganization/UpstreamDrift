"""State estimation for digital twin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.deployment.realtime import RobotState


@dataclass
class EstimatorConfig:
    """Configuration for state estimator.

    Attributes:
        process_noise: Process noise covariance.
        measurement_noise: Measurement noise covariance.
        use_velocity_filter: Apply low-pass filter to velocities.
        velocity_filter_alpha: Filter coefficient (0-1, lower=smoother).
        outlier_threshold: Threshold for outlier rejection.
    """

    process_noise: float = 0.001
    measurement_noise: float = 0.01
    use_velocity_filter: bool = True
    velocity_filter_alpha: float = 0.3
    outlier_threshold: float = 3.0


class StateEstimator:
    """State estimator using Kalman filtering.

    Provides filtered state estimates from noisy sensor data,
    including position, velocity, and acceleration estimation.

    Attributes:
        config: Estimator configuration.
        n_dof: Number of degrees of freedom.
    """

    def __init__(
        self,
        n_dof: int = 7,
        config: EstimatorConfig | None = None,
    ) -> None:
        """Initialize state estimator.

        Args:
            n_dof: Number of degrees of freedom.
            config: Estimator configuration.
        """
        self.config = config or EstimatorConfig()
        self.n_dof = n_dof

        # State: [q, qd, qdd] - position, velocity, acceleration
        self._state_dim = 3 * n_dof
        self._state = np.zeros(self._state_dim)
        self._covariance = np.eye(self._state_dim) * 0.1

        # Kalman filter matrices
        self._setup_kalman_matrices()

        # History for filtering
        self._position_history: list[NDArray[np.floating]] = []
        self._velocity_history: list[NDArray[np.floating]] = []
        self._filtered_velocity = np.zeros(n_dof)
        self._last_timestamp = 0.0

    def _setup_kalman_matrices(self) -> None:
        """Setup Kalman filter matrices."""
        n = self.n_dof

        # Measurement matrix (we measure position and velocity)
        self._H = np.zeros((2 * n, 3 * n))
        self._H[:n, :n] = np.eye(n)  # Position measurement
        self._H[n:, n : 2 * n] = np.eye(n)  # Velocity measurement

        # Measurement noise
        self._R = np.eye(2 * n) * self.config.measurement_noise

        # Process noise
        self._Q = np.eye(3 * n) * self.config.process_noise

    def update(
        self,
        robot_state: "RobotState",
        dt: float | None = None,
    ) -> dict[str, NDArray[np.floating]]:
        """Update state estimate with new measurement.

        Args:
            robot_state: New robot state measurement.
            dt: Time since last update (auto-computed if None).

        Returns:
            Dictionary with estimated position, velocity, acceleration.
        """
        if dt is None:
            dt = robot_state.timestamp - self._last_timestamp
            if dt <= 0:
                dt = 0.001
        self._last_timestamp = robot_state.timestamp

        n = self.n_dof

        # State transition matrix (constant acceleration model)
        F = np.eye(3 * n)
        F[:n, n : 2 * n] = np.eye(n) * dt
        F[:n, 2 * n :] = np.eye(n) * (dt**2 / 2)
        F[n : 2 * n, 2 * n :] = np.eye(n) * dt

        # Prediction step
        self._state = F @ self._state
        self._covariance = F @ self._covariance @ F.T + self._Q

        # Measurement
        z = np.concatenate([
            robot_state.joint_positions,
            robot_state.joint_velocities,
        ])

        # Outlier rejection
        z = self._reject_outliers(z)

        # Innovation
        y = z - self._H @ self._state

        # Kalman gain
        S = self._H @ self._covariance @ self._H.T + self._R
        K = self._covariance @ self._H.T @ np.linalg.inv(S)

        # Update
        self._state = self._state + K @ y
        self._covariance = (np.eye(3 * n) - K @ self._H) @ self._covariance

        # Apply velocity filter
        estimated_velocity = self._state[n : 2 * n]
        if self.config.use_velocity_filter:
            alpha = self.config.velocity_filter_alpha
            self._filtered_velocity = (
                alpha * estimated_velocity
                + (1 - alpha) * self._filtered_velocity
            )
            estimated_velocity = self._filtered_velocity

        return {
            "position": self._state[:n].copy(),
            "velocity": estimated_velocity.copy(),
            "acceleration": self._state[2 * n :].copy(),
        }

    def _reject_outliers(
        self,
        measurement: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Reject outlier measurements.

        Args:
            measurement: Raw measurement.

        Returns:
            Filtered measurement with outliers replaced.
        """
        predicted = self._H @ self._state
        residual = measurement - predicted
        std = np.sqrt(np.diag(self._H @ self._covariance @ self._H.T + self._R))

        # Replace outliers with prediction
        outliers = np.abs(residual) > self.config.outlier_threshold * std
        measurement = measurement.copy()
        measurement[outliers] = predicted[outliers]

        return measurement

    def reset(
        self,
        position: NDArray[np.floating] | None = None,
        velocity: NDArray[np.floating] | None = None,
    ) -> None:
        """Reset estimator state.

        Args:
            position: Initial position (zeros if None).
            velocity: Initial velocity (zeros if None).
        """
        n = self.n_dof
        self._state = np.zeros(3 * n)

        if position is not None:
            self._state[:n] = position
        if velocity is not None:
            self._state[n : 2 * n] = velocity

        self._covariance = np.eye(3 * n) * 0.1
        self._filtered_velocity = self._state[n : 2 * n].copy()
        self._position_history.clear()
        self._velocity_history.clear()

    def get_position(self) -> NDArray[np.floating]:
        """Get current position estimate.

        Returns:
            Estimated joint positions.
        """
        return self._state[: self.n_dof].copy()

    def get_velocity(self) -> NDArray[np.floating]:
        """Get current velocity estimate.

        Returns:
            Estimated joint velocities.
        """
        if self.config.use_velocity_filter:
            return self._filtered_velocity.copy()
        return self._state[self.n_dof : 2 * self.n_dof].copy()

    def get_acceleration(self) -> NDArray[np.floating]:
        """Get current acceleration estimate.

        Returns:
            Estimated joint accelerations.
        """
        return self._state[2 * self.n_dof :].copy()

    def get_covariance(self) -> NDArray[np.floating]:
        """Get current state covariance.

        Returns:
            State covariance matrix.
        """
        return self._covariance.copy()

    def get_position_uncertainty(self) -> NDArray[np.floating]:
        """Get position estimate uncertainty.

        Returns:
            Standard deviation of position estimates.
        """
        n = self.n_dof
        return np.sqrt(np.diag(self._covariance[:n, :n]))

    def get_velocity_uncertainty(self) -> NDArray[np.floating]:
        """Get velocity estimate uncertainty.

        Returns:
            Standard deviation of velocity estimates.
        """
        n = self.n_dof
        return np.sqrt(np.diag(self._covariance[n : 2 * n, n : 2 * n]))

    def predict(
        self,
        dt: float,
        control: NDArray[np.floating] | None = None,
    ) -> dict[str, NDArray[np.floating]]:
        """Predict future state.

        Args:
            dt: Prediction time horizon.
            control: Optional control input (acceleration).

        Returns:
            Predicted state.
        """
        n = self.n_dof

        # State transition
        F = np.eye(3 * n)
        F[:n, n : 2 * n] = np.eye(n) * dt
        F[:n, 2 * n :] = np.eye(n) * (dt**2 / 2)
        F[n : 2 * n, 2 * n :] = np.eye(n) * dt

        predicted_state = F @ self._state

        if control is not None:
            # Add control as acceleration
            predicted_state[2 * n :] += control

        return {
            "position": predicted_state[:n].copy(),
            "velocity": predicted_state[n : 2 * n].copy(),
            "acceleration": predicted_state[2 * n :].copy(),
        }
