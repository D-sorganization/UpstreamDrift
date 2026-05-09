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
