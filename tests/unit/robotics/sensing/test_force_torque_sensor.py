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
        assert np.allclose(reading.wrench, 0.0, atol=1.0)  # Allow for noise

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
