"""IMU (Inertial Measurement Unit) sensor simulation.

This module provides a configurable IMU simulation with realistic
noise characteristics for robotics applications.

Design by Contract:
    All IMU readings contain valid acceleration and angular velocity.
    Orientation estimates (when available) are unit quaternions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.robotics.core.types import IMUReading
from src.robotics.sensing.noise_models import (
    BandwidthLimitedNoise,
    BrownianNoise,
    CompositeNoise,
    GaussianNoise,
)


@dataclass
class IMUSensorConfig:
    """Configuration for IMU sensor.

    Attributes:
        sensor_id: Unique sensor identifier.
        accel_range: Maximum measurable acceleration [m/s^2].
        gyro_range: Maximum measurable angular velocity [rad/s].
        accel_noise_std: Accelerometer noise std [m/s^2].
        gyro_noise_std: Gyroscope noise std [rad/s].
        accel_bias_drift: Accelerometer bias drift [m/s^2/step].
        gyro_bias_drift: Gyroscope bias drift [rad/s/step].
        gravity: Gravity vector in world frame [m/s^2].
        cutoff_frequency: Sensor bandwidth [Hz].
        sample_rate: Sampling rate [Hz].
        seed: Random seed for reproducibility.
    """

    sensor_id: str = "imu"
    accel_range: float = 160.0  # ~16g
    gyro_range: float = 35.0   # ~2000 deg/s
    accel_noise_std: float = 0.01
    gyro_noise_std: float = 0.001
    accel_bias_drift: float = 0.0001
    gyro_bias_drift: float = 0.00001
    gravity: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81])
    )
    cutoff_frequency: float = 200.0
    sample_rate: float = 1000.0
    seed: int | None = None


class IMUSensor:
    """Simulated IMU sensor with realistic noise.

    Provides 6-axis IMU measurements (3-axis accelerometer + 3-axis gyroscope)
    with configurable noise characteristics including:
    - Additive white Gaussian noise
    - Bias drift (random walk)
    - Bandwidth limitations

    Design by Contract:
        Invariants:
            - Acceleration readings are 3D vectors
            - Angular velocity readings are 3D vectors
            - Noise parameters are non-negative

        Postconditions:
            - read() returns valid IMUReading
            - Orientation quaternion (if computed) is unit length

    Example:
        >>> config = IMUSensorConfig(sensor_id="body_imu")
        >>> imu = IMUSensor(config)
        >>> reading = imu.read(
        ...     linear_accel=np.array([0, 0, 9.81]),
        ...     angular_vel=np.array([0, 0, 0.1]),
        ...     timestamp=0.001,
        ... )
    """

    def __init__(self, config: IMUSensorConfig | None = None) -> None:
        """Initialize IMU sensor.

        Args:
            config: Sensor configuration. Uses defaults if None.
        """
        self._config = config or IMUSensorConfig()
        self._validate_config()

        # Initialize noise models
        self._accel_noise = self._create_noise_model(
            self._config.accel_noise_std,
            self._config.accel_bias_drift,
        )
        self._gyro_noise = self._create_noise_model(
            self._config.gyro_noise_std,
            self._config.gyro_bias_drift,
        )

        # Bandwidth filters
        self._accel_filter = BandwidthLimitedNoise(
            cutoff_frequency=self._config.cutoff_frequency,
            sample_rate=self._config.sample_rate,
        )
        self._gyro_filter = BandwidthLimitedNoise(
            cutoff_frequency=self._config.cutoff_frequency,
            sample_rate=self._config.sample_rate,
        )

        # Orientation estimate state
        self._orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        self._last_timestamp: float | None = None

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self._config.accel_noise_std < 0:
            raise ValueError("accel_noise_std must be non-negative")
        if self._config.gyro_noise_std < 0:
            raise ValueError("gyro_noise_std must be non-negative")
        if self._config.cutoff_frequency <= 0:
            raise ValueError("cutoff_frequency must be positive")

    def _create_noise_model(
        self,
        noise_std: float,
        bias_drift: float,
    ) -> CompositeNoise:
        """Create composite noise model.

        Args:
            noise_std: White noise standard deviation.
            bias_drift: Bias drift rate.

        Returns:
            Composite noise model.
        """
        return CompositeNoise(
            models=[
                BrownianNoise(
                    drift_rate=bias_drift,
                    max_bias=noise_std * 10,
                    seed=self._config.seed,
                ),
                GaussianNoise(std=noise_std, seed=self._config.seed),
            ]
        )

    @property
    def sensor_id(self) -> str:
        """Get sensor identifier."""
        return self._config.sensor_id

    @property
    def config(self) -> IMUSensorConfig:
        """Get sensor configuration."""
        return self._config

    @property
    def orientation(self) -> NDArray[np.float64]:
        """Get current orientation estimate (quaternion)."""
        return self._orientation.copy()

    def read(
        self,
        linear_accel: NDArray[np.float64],
        angular_vel: NDArray[np.float64],
        timestamp: float = 0.0,
        include_orientation: bool = True,
    ) -> IMUReading:
        """Read IMU sensor with noise applied.

        Args:
            linear_accel: True linear acceleration [ax, ay, az] in sensor frame [m/s^2].
            angular_vel: True angular velocity [wx, wy, wz] in sensor frame [rad/s].
            timestamp: Measurement timestamp [s].
            include_orientation: Whether to include orientation estimate.

        Returns:
            IMUReading with noisy measurements.

        Raises:
            ValueError: If input arrays have wrong shape.
        """
        linear_accel = np.asarray(linear_accel, dtype=np.float64)
        angular_vel = np.asarray(angular_vel, dtype=np.float64)

        if linear_accel.shape != (3,):
            raise ValueError(
                f"linear_accel must be (3,), got {linear_accel.shape}"
            )
        if angular_vel.shape != (3,):
            raise ValueError(
                f"angular_vel must be (3,), got {angular_vel.shape}"
            )

        # Apply noise
        noisy_accel = self._accel_noise.apply(linear_accel)
        noisy_gyro = self._gyro_noise.apply(angular_vel)

        # Apply bandwidth filter
        filtered_accel = self._accel_filter.apply(noisy_accel)
        filtered_gyro = self._gyro_filter.apply(noisy_gyro)

        # Clip to sensor range
        filtered_accel = np.clip(
            filtered_accel,
            -self._config.accel_range,
            self._config.accel_range,
        )
        filtered_gyro = np.clip(
            filtered_gyro,
            -self._config.gyro_range,
            self._config.gyro_range,
        )

        # Update orientation estimate
        orientation = None
        if include_orientation:
            if self._last_timestamp is not None:
                dt = timestamp - self._last_timestamp
                if dt > 0:
                    self._integrate_orientation(filtered_gyro, dt)
            orientation = self._orientation.copy()

        self._last_timestamp = timestamp

        return IMUReading(
            timestamp=timestamp,
            sensor_id=self._config.sensor_id,
            linear_acceleration=filtered_accel,
            angular_velocity=filtered_gyro,
            orientation=orientation,
        )

    def _integrate_orientation(
        self,
        angular_vel: NDArray[np.float64],
        dt: float,
    ) -> None:
        """Integrate angular velocity to update orientation.

        Uses first-order quaternion integration.

        Args:
            angular_vel: Angular velocity [rad/s].
            dt: Time step [s].
        """
        # Quaternion derivative: dq/dt = 0.5 * q * omega
        # where omega = [0, wx, wy, wz]
        omega_mag = float(np.linalg.norm(angular_vel))

        if omega_mag < 1e-10:
            return

        # Compute rotation quaternion
        half_angle = 0.5 * omega_mag * dt
        axis = angular_vel / omega_mag

        dq = np.array([
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
        ])

        # Quaternion multiplication: q_new = q * dq
        self._orientation = _quaternion_multiply(self._orientation, dq)

        # Normalize
        self._orientation /= np.linalg.norm(self._orientation)

    def reset(self) -> None:
        """Reset sensor state."""
        self._accel_noise.reset()
        self._gyro_noise.reset()
        self._accel_filter.reset()
        self._gyro_filter.reset()
        self._orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self._last_timestamp = None

    def set_orientation(self, quaternion: NDArray[np.float64]) -> None:
        """Set current orientation estimate.

        Args:
            quaternion: Orientation as [w, x, y, z] quaternion.
        """
        quaternion = np.asarray(quaternion, dtype=np.float64)
        if quaternion.shape != (4,):
            raise ValueError(f"Quaternion must be (4,), got {quaternion.shape}")

        # Normalize
        self._orientation = quaternion / np.linalg.norm(quaternion)

    def get_gravity_in_sensor_frame(self) -> NDArray[np.float64]:
        """Get gravity vector in current sensor frame.

        Returns:
            Gravity vector (3,) in sensor frame [m/s^2].
        """
        # Rotate world gravity into sensor frame using orientation inverse
        q_inv = _quaternion_inverse(self._orientation)
        return _rotate_vector_by_quaternion(self._config.gravity, q_inv)


def _quaternion_multiply(
    q1: NDArray[np.float64],
    q2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Multiply two quaternions.

    Args:
        q1: First quaternion [w, x, y, z].
        q2: Second quaternion [w, x, y, z].

    Returns:
        Product quaternion q1 * q2.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quaternion_inverse(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute quaternion inverse (conjugate for unit quaternion).

    Args:
        q: Quaternion [w, x, y, z].

    Returns:
        Inverse quaternion.
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _rotate_vector_by_quaternion(
    v: NDArray[np.float64],
    q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate vector by quaternion.

    Args:
        v: Vector (3,) to rotate.
        q: Rotation quaternion [w, x, y, z].

    Returns:
        Rotated vector (3,).
    """
    # v' = q * [0, v] * q^{-1}
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    q_inv = _quaternion_inverse(q)

    result = _quaternion_multiply(
        _quaternion_multiply(q, v_quat),
        q_inv,
    )

    return result[1:4]


def create_ideal_imu(sensor_id: str = "ideal_imu") -> IMUSensor:
    """Create an ideal (noiseless) IMU sensor.

    Args:
        sensor_id: Sensor identifier.

    Returns:
        IMUSensor with zero noise.
    """
    return IMUSensor(
        IMUSensorConfig(
            sensor_id=sensor_id,
            accel_noise_std=0.0,
            gyro_noise_std=0.0,
            accel_bias_drift=0.0,
            gyro_bias_drift=0.0,
        )
    )


def create_realistic_imu(
    sensor_id: str = "imu",
    quality: str = "industrial",
    seed: int | None = None,
) -> IMUSensor:
    """Create an IMU sensor with realistic noise.

    Args:
        sensor_id: Sensor identifier.
        quality: Sensor quality level ('mems', 'industrial', 'tactical').
        seed: Random seed.

    Returns:
        IMUSensor with appropriate noise characteristics.
    """
    # Noise parameters based on typical sensor grades
    noise_params = {
        "mems": {
            "accel_noise_std": 0.1,
            "gyro_noise_std": 0.01,
            "accel_bias_drift": 0.001,
            "gyro_bias_drift": 0.0001,
        },
        "industrial": {
            "accel_noise_std": 0.01,
            "gyro_noise_std": 0.001,
            "accel_bias_drift": 0.0001,
            "gyro_bias_drift": 0.00001,
        },
        "tactical": {
            "accel_noise_std": 0.001,
            "gyro_noise_std": 0.0001,
            "accel_bias_drift": 0.00001,
            "gyro_bias_drift": 0.000001,
        },
    }

    params = noise_params.get(quality, noise_params["industrial"])

    return IMUSensor(
        IMUSensorConfig(
            sensor_id=sensor_id,
            seed=seed,
            **params,
        )
    )
