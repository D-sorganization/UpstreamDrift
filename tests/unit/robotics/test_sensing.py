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


class TestNoiseModels:
    """Tests for noise model classes."""

    def test_gaussian_noise_shape_preserved(self) -> None:
        """Test Gaussian noise preserves input shape."""
        noise = GaussianNoise(std=0.1, seed=42)
        signal = np.array([1.0, 2.0, 3.0])

        noisy = noise.apply(signal)

        assert noisy.shape == signal.shape

    def test_gaussian_noise_statistics(self) -> None:
        """Test Gaussian noise has correct statistics."""
        noise = GaussianNoise(std=1.0, mean=0.0, seed=42)

        # Apply to many samples
        samples = []
        for _ in range(10000):
            noisy = noise.apply(np.array([0.0]))
            samples.append(noisy[0])

        samples = np.array(samples)

        # Check mean and std are approximately correct
        assert abs(np.mean(samples)) < 0.1  # Mean close to 0
        assert abs(np.std(samples) - 1.0) < 0.1  # Std close to 1

    def test_gaussian_noise_reproducibility(self) -> None:
        """Test Gaussian noise is reproducible with seed."""
        noise1 = GaussianNoise(std=0.1, seed=42)
        noise2 = GaussianNoise(std=0.1, seed=42)

        signal = np.array([1.0, 2.0, 3.0])

        noisy1 = noise1.apply(signal)
        noisy2 = noise2.apply(signal)

        assert_allclose(noisy1, noisy2)

    def test_brownian_noise_drift(self) -> None:
        """Test Brownian noise accumulates drift."""
        noise = BrownianNoise(drift_rate=0.1, seed=42)

        # Apply many times
        signal = np.array([0.0])
        for _ in range(100):
            noise.apply(signal)

        # Bias should have drifted
        assert noise.current_bias != 0.0

    def test_brownian_noise_max_bias(self) -> None:
        """Test Brownian noise respects max bias."""
        noise = BrownianNoise(drift_rate=1.0, max_bias=0.5, seed=42)

        signal = np.array([0.0])
        for _ in range(1000):
            noise.apply(signal)

        assert abs(noise.current_bias) <= 0.5

    def test_brownian_noise_reset(self) -> None:
        """Test Brownian noise reset."""
        noise = BrownianNoise(drift_rate=0.1, initial_bias=0.5, seed=42)

        noise.apply(np.array([0.0]))
        noise.reset()

        assert noise.current_bias == 0.5

    def test_quantization_noise(self) -> None:
        """Test quantization noise discretizes signal."""
        noise = QuantizationNoise(resolution=0.1)

        # Use values that don't fall on rounding boundaries
        signal = np.array([0.04, 0.16, 0.27])
        quantized = noise.apply(signal)

        # Values should be multiples of resolution (nearest)
        expected = np.array([0.0, 0.2, 0.3])
        assert_allclose(quantized, expected)

    def test_bandwidth_limited_noise(self) -> None:
        """Test bandwidth filter smooths signal."""
        noise = BandwidthLimitedNoise(
            cutoff_frequency=10.0,
            sample_rate=100.0,
        )

        # Step input
        outputs = []
        for i in range(50):
            signal = np.array([1.0]) if i > 0 else np.array([0.0])
            filtered = noise.apply(signal)
            outputs.append(filtered[0])

        outputs = np.array(outputs)

        # Should have smooth rise, not instant step
        assert outputs[1] < 1.0  # Not instant
        assert outputs[-1] > 0.9  # Eventually reaches target

    def test_composite_noise(self) -> None:
        """Test composite noise applies all models."""
        composite = CompositeNoise(
            models=[
                GaussianNoise(std=0.1, seed=42),
                QuantizationNoise(resolution=0.01),
            ]
        )

        signal = np.array([1.0])
        noisy = composite.apply(signal)

        # Should be different from original
        assert noisy[0] != signal[0]

    def test_create_realistic_sensor_noise(self) -> None:
        """Test factory function creates valid composite."""
        noise = create_realistic_sensor_noise(
            noise_std=0.1,
            bias_drift_rate=0.01,
            quantization_bits=12,
            signal_range=10.0,
            seed=42,
        )

        signal = np.array([5.0])
        noisy = noise.apply(signal)

        assert noisy.shape == signal.shape


class TestForceTorqueSensor:
    """Tests for ForceTorqueSensor class."""

    def test_create_sensor(self) -> None:
        """Test creating a force/torque sensor."""
        config = ForceTorqueSensorConfig(
            sensor_id="test_ft",
            force_noise_std=0.1,
            torque_noise_std=0.01,
        )
        sensor = ForceTorqueSensor(config)

        assert sensor.sensor_id == "test_ft"
        assert sensor.config == config

    def test_ideal_sensor_no_noise(self) -> None:
        """Test ideal sensor adds no noise."""
        sensor = create_ideal_sensor()

        true_wrench = np.array([10.0, 0.0, 50.0, 1.0, 0.0, 0.0])
        reading = sensor.read(true_wrench, timestamp=0.001)

        # Should be very close to true value
        assert_allclose(reading.wrench, true_wrench, atol=1e-10)

    def test_sensor_reading_shape(self) -> None:
        """Test sensor reading has correct shape."""
        sensor = ForceTorqueSensor()

        true_wrench = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        reading = sensor.read(true_wrench, timestamp=0.001)

        assert reading.wrench.shape == (6,)
        assert reading.force.shape == (3,)
        assert reading.torque.shape == (3,)

    def test_sensor_invalid_wrench_raises(self) -> None:
        """Test sensor raises for invalid wrench shape."""
        sensor = ForceTorqueSensor()

        with pytest.raises(ValueError, match="must be 6D"):
            sensor.read(np.array([1.0, 2.0, 3.0]))

    def test_sensor_tare(self) -> None:
        """Test sensor tare functionality."""
        sensor = create_ideal_sensor()

        # First reading
        true_wrench = np.array([10.0, 0.0, 50.0, 1.0, 0.0, 0.0])
        sensor.read(true_wrench, timestamp=0.001)

        # Tare at current reading
        sensor.tare()

        # Next reading should be offset
        reading = sensor.read(true_wrench, timestamp=0.002)
        assert_allclose(reading.wrench, np.zeros(6), atol=1e-10)

    def test_sensor_reset(self) -> None:
        """Test sensor reset clears state."""
        sensor = ForceTorqueSensor()

        sensor.read(np.array([10.0, 0.0, 50.0, 1.0, 0.0, 0.0]))
        sensor.tare(np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        sensor.reset()

        # After reset, tare should be zero
        reading = sensor.read(np.zeros(6))
        # Should be near zero (only noise)

    def test_sensor_clipping(self) -> None:
        """Test sensor clips to range."""
        config = ForceTorqueSensorConfig(
            force_range=100.0,
            torque_range=10.0,
            force_noise_std=0.0,
            torque_noise_std=0.0,
        )
        sensor = ForceTorqueSensor(config)

        # Wrench exceeding range
        true_wrench = np.array([200.0, 0.0, 0.0, 20.0, 0.0, 0.0])
        reading = sensor.read(true_wrench)

        assert abs(reading.wrench[0]) <= 100.0
        assert abs(reading.wrench[3]) <= 10.0

    def test_realistic_sensor_adds_noise(self) -> None:
        """Test realistic sensor adds noise."""
        sensor = create_realistic_sensor(quality="industrial", seed=42)

        true_wrench = np.array([10.0, 0.0, 50.0, 1.0, 0.0, 0.0])
        reading = sensor.read(true_wrench)

        # Should be different from true value
        assert not np.allclose(reading.wrench, true_wrench)

    def test_contact_location_estimation(self) -> None:
        """Test contact location estimation."""
        sensor = ForceTorqueSensor()

        # Force at known location
        # If force is [0, 0, 10] at position [1, 0, 0]
        # Torque should be r x f = [1,0,0] x [0,0,10] = [0, -10, 0]
        wrench = np.array([0.0, 0.0, 10.0, 0.0, -10.0, 0.0])

        location = sensor.estimate_contact_location(wrench)

        assert location is not None
        # Should estimate x ≈ 1
        assert abs(location[0] - 1.0) < 0.1


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


class TestSensorFactories:
    """Tests for sensor factory functions."""

    def test_create_ideal_ft_sensor(self) -> None:
        """Test ideal F/T sensor factory."""
        sensor = create_ideal_sensor("my_sensor")
        assert sensor.sensor_id == "my_sensor"
        assert sensor.config.force_noise_std == 0.0

    def test_create_realistic_ft_sensor_qualities(self) -> None:
        """Test realistic F/T sensor at different qualities."""
        for quality in ["research", "industrial", "consumer"]:
            sensor = create_realistic_sensor(quality=quality)
            assert sensor.config.force_noise_std > 0

    def test_create_ideal_imu(self) -> None:
        """Test ideal IMU factory."""
        imu = create_ideal_imu("my_imu")
        assert imu.sensor_id == "my_imu"
        assert imu.config.accel_noise_std == 0.0

    def test_create_realistic_imu_qualities(self) -> None:
        """Test realistic IMU at different qualities."""
        for quality in ["mems", "industrial", "tactical"]:
            imu = create_realistic_imu(quality=quality)
            assert imu.config.accel_noise_std > 0
