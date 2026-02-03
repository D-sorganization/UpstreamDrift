"""Sensing module for robotics applications.

This module provides simulated sensors with realistic noise models:
- Force/torque sensors for contact force measurement
- IMU sensors for orientation and acceleration
- Configurable noise models for sim-to-real transfer

Example:
    >>> from src.robotics.sensing import ForceTorqueSensor, IMUSensor
    >>>
    >>> ft_sensor = ForceTorqueSensor(
    ...     sensor_id="wrist_ft",
    ...     force_noise_std=0.1,
    ...     torque_noise_std=0.01,
    ... )
    >>> reading = ft_sensor.read(true_wrench)
"""

from __future__ import annotations

from src.robotics.sensing.force_torque_sensor import (
    ForceTorqueSensor,
    ForceTorqueSensorConfig,
)
from src.robotics.sensing.imu_sensor import (
    IMUSensor,
    IMUSensorConfig,
)
from src.robotics.sensing.noise_models import (
    BrownianNoise,
    CompositeNoise,
    GaussianNoise,
    NoiseModel,
    QuantizationNoise,
)

__all__ = [
    "ForceTorqueSensor",
    "ForceTorqueSensorConfig",
    "IMUSensor",
    "IMUSensorConfig",
    "NoiseModel",
    "GaussianNoise",
    "BrownianNoise",
    "QuantizationNoise",
    "CompositeNoise",
]
