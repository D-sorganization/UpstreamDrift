"""Unit tests for sensing module.

Tests cover:
    - Noise models (Gaussian, Brownian, Quantization)
    - Force/torque sensor simulation
    - IMU sensor simulation
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from src.robotics.sensing.force_torque_sensor import (
    ForceTorqueSensor,
    ForceTorqueSensorConfig,
    create_ideal_sensor,
    create_realistic_sensor,
)
from src.robotics.sensing.imu_sensor import (
    IMUSensor,
    IMUSensorConfig,
    create_ideal_imu,
    create_realistic_imu,
)
from src.robotics.sensing.noise_models import (
    BandwidthLimitedNoise,
    BrownianNoise,
    CompositeNoise,
    GaussianNoise,
    QuantizationNoise,
    create_realistic_sensor_noise,
)


class TestIMUSensor:
    """Tests for IMUSensor class."""

    def test_create_imu(self) -> None:
        """Test creating an IMU sensor."""
        config = IMUSensorConfig(
            sensor_id="test_imu",
            accel_noise_std=0.01,
            gyro_noise_std=0.001,
        )
        imu = IMUSensor(config)

        assert imu.sensor_id == "test_imu"
        assert imu.config == config

    def test_ideal_imu_no_noise(self) -> None:
        """Test ideal IMU adds no noise."""
        imu = create_ideal_imu()

        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.1])

        reading = imu.read(accel, gyro, timestamp=0.001)

        assert_allclose(reading.linear_acceleration, accel, atol=1e-10)
        assert_allclose(reading.angular_velocity, gyro, atol=1e-10)

    def test_imu_reading_shape(self) -> None:
        """Test IMU reading has correct shape."""
        imu = IMUSensor()

        reading = imu.read(
            linear_accel=np.array([0.0, 0.0, 9.81]),
            angular_vel=np.array([0.0, 0.0, 0.1]),
        )

        assert reading.linear_acceleration.shape == (3,)
        assert reading.angular_velocity.shape == (3,)

    def test_imu_invalid_input_raises(self) -> None:
        """Test IMU raises for invalid input shape."""
        imu = IMUSensor()

        with pytest.raises(ValueError, match="must be"):
            imu.read(
                linear_accel=np.array([0.0, 0.0]),  # Wrong shape
                angular_vel=np.array([0.0, 0.0, 0.0]),
            )

    def test_imu_orientation_integration(self) -> None:
        """Test IMU integrates orientation from gyro."""
        imu = create_ideal_imu()

        # Rotate around z-axis at 1 rad/s for 1 second
        dt = 0.01
        angular_vel = np.array([0.0, 0.0, 1.0])

        for i in range(100):
            imu.read(
                linear_accel=np.zeros(3),
                angular_vel=angular_vel,
                timestamp=i * dt,
            )

        # Should have rotated ~1 radian around z
        orientation = imu.orientation

        # Convert quaternion to angle
        # For rotation around z: q = [cos(theta/2), 0, 0, sin(theta/2)]
        angle = 2 * np.arctan2(orientation[3], orientation[0])

        assert abs(angle - 1.0) < 0.1  # Approximately 1 radian

    def test_imu_reset(self) -> None:
        """Test IMU reset clears state."""
        imu = IMUSensor()

        # Do some readings
        imu.read(np.zeros(3), np.array([0, 0, 1]), timestamp=0.0)
        imu.read(np.zeros(3), np.array([0, 0, 1]), timestamp=0.1)

        imu.reset()

        # Orientation should be identity
        assert_allclose(imu.orientation, [1, 0, 0, 0])

    def test_imu_set_orientation(self) -> None:
        """Test IMU set_orientation."""
        imu = IMUSensor()

        # Set to 90 degree rotation around z
        q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        imu.set_orientation(q)

        assert_allclose(imu.orientation, q)

    def test_imu_clipping(self) -> None:
        """Test IMU clips to range."""
        config = IMUSensorConfig(
            accel_range=10.0,
            gyro_range=1.0,
            accel_noise_std=0.0,
            gyro_noise_std=0.0,
        )
        imu = IMUSensor(config)

        reading = imu.read(
            linear_accel=np.array([100.0, 0.0, 0.0]),
            angular_vel=np.array([10.0, 0.0, 0.0]),
        )

        assert abs(reading.linear_acceleration[0]) <= 10.0
        assert abs(reading.angular_velocity[0]) <= 1.0

    def test_realistic_imu_adds_noise(self) -> None:
        """Test realistic IMU adds noise."""
        imu = create_realistic_imu(quality="mems", seed=42)

        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])

        reading = imu.read(accel, gyro)

        # Should be different from true value
        assert not np.allclose(reading.linear_acceleration, accel)

    def test_gravity_in_sensor_frame(self) -> None:
        """Test gravity vector in sensor frame."""
        imu = IMUSensor()

        # At identity orientation, gravity should be in -z
        gravity = imu.get_gravity_in_sensor_frame()
        assert_allclose(gravity, [0, 0, -9.81], atol=1e-10)

        # After 90 degree rotation around y, gravity should be in -x
        q = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
        imu.set_orientation(q)

        gravity = imu.get_gravity_in_sensor_frame()
        assert abs(gravity[0] - 9.81) < 0.1  # ~9.81 in x direction
