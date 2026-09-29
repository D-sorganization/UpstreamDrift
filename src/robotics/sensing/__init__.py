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
    FORCE_TORQUE_INDUSTRIAL_DEFAULTS,
    IMU_MEMS_DEFAULTS,
    BrownianNoise,
    CompositeNoise,
    GaussianNoise,
    NoiseModel,
    NoisySensor,
    QuantizationNoise,
    SensorNoiseParameters,
)

__all__ = [
    "FORCE_TORQUE_INDUSTRIAL_DEFAULTS",
    "IMU_MEMS_DEFAULTS",
    "BrownianNoise",
    "CompositeNoise",
    "ForceTorqueSensor",
    "ForceTorqueSensorConfig",
    "GaussianNoise",
    "IMUSensor",
    "IMUSensorConfig",
    "NoiseModel",
    "NoisySensor",
    "QuantizationNoise",
    "SensorNoiseParameters",
]
