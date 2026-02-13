"""Force/torque sensor simulation.

This module provides a configurable force/torque sensor simulation
with realistic noise characteristics for robotics applications.

Design by Contract:
    All sensor readings are valid 6D wrenches.
    Noise parameters are non-negative.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.robotics.core.types import ForceTorqueReading
from src.robotics.sensing.noise_models import (
    BandwidthLimitedNoise,
    BrownianNoise,
    CompositeNoise,
    GaussianNoise,
)
from src.shared.python.core.contracts import ContractChecker


@dataclass
class ForceTorqueSensorConfig:
    """Configuration for force/torque sensor.

    Attributes:
        sensor_id: Unique sensor identifier.
        force_range: Maximum measurable force [N].
        torque_range: Maximum measurable torque [Nm].
        force_noise_std: Force measurement noise std [N].
        torque_noise_std: Torque measurement noise std [Nm].
        force_bias_drift: Force bias drift rate [N/step].
        torque_bias_drift: Torque bias drift rate [Nm/step].
        cutoff_frequency: Sensor bandwidth [Hz].
        sample_rate: Sampling rate [Hz].
        seed: Random seed for reproducibility.
    """

    sensor_id: str = "ft_sensor"
    force_range: float = 1000.0
    torque_range: float = 100.0
    force_noise_std: float = 0.1
    torque_noise_std: float = 0.01
    force_bias_drift: float = 0.001
    torque_bias_drift: float = 0.0001
    cutoff_frequency: float = 100.0
    sample_rate: float = 1000.0
    seed: int | None = None


class ForceTorqueSensor(ContractChecker):
    """Simulated force/torque sensor with realistic noise.

    Provides 6-axis force/torque measurements with configurable
    noise characteristics including:
    - Additive white Gaussian noise
    - Bias drift (random walk)
    - Bandwidth limitations

    Design by Contract:
        Invariants:
            - Tare offset is always a 6D vector
            - Config has non-negative noise parameters
            - Force and torque ranges are positive

        Postconditions:
            - read() returns valid ForceTorqueReading
            - get_wrench() returns (6,) array

    Example:
        >>> config = ForceTorqueSensorConfig(
        ...     sensor_id="wrist_ft",
        ...     force_noise_std=0.5,
        ... )
        >>> sensor = ForceTorqueSensor(config)
        >>> true_wrench = np.array([10, 0, 50, 0, 0, 0])
        >>> reading = sensor.read(true_wrench, timestamp=0.001)
    """

    def __init__(self, config: ForceTorqueSensorConfig | None = None) -> None:
        """Initialize force/torque sensor.

        Args:
            config: Sensor configuration. Uses defaults if None.
        """
        self._config = config or ForceTorqueSensorConfig()
        self._validate_config()

        # Initialize noise models for forces and torques
        self._force_noise = self._create_noise_model(
            self._config.force_noise_std,
            self._config.force_bias_drift,
        )
        self._torque_noise = self._create_noise_model(
            self._config.torque_noise_std,
            self._config.torque_bias_drift,
        )

        # Bandwidth filter
        self._filter = BandwidthLimitedNoise(
            cutoff_frequency=self._config.cutoff_frequency,
            sample_rate=self._config.sample_rate,
        )

        # Tare offset
        self._tare_offset = np.zeros(6)

        # Last reading for filtering
        self._last_reading: NDArray[np.float64] | None = None

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        """Define class invariants for ForceTorqueSensor."""
        return [
            (
                lambda: self._tare_offset is not None
                and self._tare_offset.shape == (6,),
                "Tare offset must be a 6D vector",
            ),
            (
                lambda: self._config.force_noise_std >= 0
                and self._config.torque_noise_std >= 0,
                "Noise standard deviations must be non-negative",
            ),
            (
                lambda: self._config.force_range > 0 and self._config.torque_range > 0,
                "Force and torque ranges must be positive",
            ),
        ]

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self._config.force_noise_std < 0:
            raise ValueError("force_noise_std must be non-negative")
        if self._config.torque_noise_std < 0:
            raise ValueError("torque_noise_std must be non-negative")
        if self._config.cutoff_frequency <= 0:
            raise ValueError("cutoff_frequency must be positive")

    def _create_noise_model(
        self,
        noise_std: float,
        bias_drift: float,
    ) -> CompositeNoise:
        """Create composite noise model for a measurement channel.

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
    def config(self) -> ForceTorqueSensorConfig:
        """Get sensor configuration."""
        return self._config

    def read(
        self,
        true_wrench: NDArray[np.float64],
        timestamp: float = 0.0,
    ) -> ForceTorqueReading:
        """Read sensor with noise applied.

        Args:
            true_wrench: True 6D wrench [fx, fy, fz, tx, ty, tz].
            timestamp: Measurement timestamp [s].

        Returns:
            ForceTorqueReading with noisy measurement.

        Raises:
            ValueError: If true_wrench is not 6D.
        """
        true_wrench = np.asarray(true_wrench, dtype=np.float64)
        if true_wrench.shape != (6,):
            raise ValueError(f"Wrench must be 6D, got shape {true_wrench.shape}")

        # Apply noise to force and torque separately
        noisy_force = self._force_noise.apply(true_wrench[:3])
        noisy_torque = self._torque_noise.apply(true_wrench[3:])

        noisy_wrench = np.concatenate([noisy_force, noisy_torque])

        # Apply bandwidth filter
        filtered_wrench = self._filter.apply(noisy_wrench)

        # Apply tare offset
        measured_wrench = filtered_wrench - self._tare_offset

        # Clip to sensor range
        measured_wrench[:3] = np.clip(
            measured_wrench[:3],
            -self._config.force_range,
            self._config.force_range,
        )
        measured_wrench[3:] = np.clip(
            measured_wrench[3:],
            -self._config.torque_range,
            self._config.torque_range,
        )

        self._last_reading = measured_wrench.copy()

        return ForceTorqueReading(
            timestamp=timestamp,
            sensor_id=self._config.sensor_id,
            wrench=measured_wrench,
        )

    def read_raw(
        self,
        true_wrench: NDArray[np.float64],
        timestamp: float = 0.0,
    ) -> ForceTorqueReading:
        """Read sensor without filtering (just noise).

        Args:
            true_wrench: True 6D wrench.
            timestamp: Measurement timestamp.

        Returns:
            ForceTorqueReading with noisy but unfiltered measurement.
        """
        true_wrench = np.asarray(true_wrench, dtype=np.float64)
        if true_wrench.shape != (6,):
            raise ValueError(f"Wrench must be 6D, got shape {true_wrench.shape}")

        noisy_force = self._force_noise.apply(true_wrench[:3])
        noisy_torque = self._torque_noise.apply(true_wrench[3:])
        noisy_wrench = np.concatenate([noisy_force, noisy_torque])

        return ForceTorqueReading(
            timestamp=timestamp,
            sensor_id=self._config.sensor_id,
            wrench=noisy_wrench,
        )

    def tare(self, current_wrench: NDArray[np.float64] | None = None) -> None:
        """Zero the sensor (remove current reading as bias).

        Args:
            current_wrench: Current wrench to use for taring.
                           Uses last reading if None.
        """
        if current_wrench is not None:
            self._tare_offset = np.asarray(current_wrench, dtype=np.float64)  # type: ignore[assignment]
        elif self._last_reading is not None:
            self._tare_offset = self._last_reading.copy()  # type: ignore[assignment]
        else:
            self._tare_offset = np.zeros(6)

    def reset(self) -> None:
        """Reset sensor state (noise, filter, tare)."""
        self._force_noise.reset()
        self._torque_noise.reset()
        self._filter.reset()
        self._tare_offset = np.zeros(6)
        self._last_reading = None

    def estimate_contact_location(
        self,
        wrench: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        """Estimate single contact location from wrench.

        Assumes a single point contact and estimates where
        the contact force is applied.

        Args:
            wrench: Measured wrench [fx, fy, fz, tx, ty, tz].

        Returns:
            Estimated contact point (3,) relative to sensor frame,
            or None if force is too small.
        """
        wrench = np.asarray(wrench, dtype=np.float64)
        force = wrench[:3]
        torque = wrench[3:]

        force_mag = float(np.linalg.norm(force))
        if force_mag < 1e-6:
            return None

        # For a point contact at position r with force f:
        # tau = r x f
        # This gives us constraints on r
        # Using r = (f x tau) / |f|^2 as approximation
        r = np.cross(force, torque) / (force_mag**2)

        return r  # type: ignore[return-value]


def create_ideal_sensor(sensor_id: str = "ideal_ft") -> ForceTorqueSensor:
    """Create an ideal (noiseless) force/torque sensor.

    Args:
        sensor_id: Sensor identifier.

    Returns:
        ForceTorqueSensor with zero noise.
    """
    return ForceTorqueSensor(
        ForceTorqueSensorConfig(
            sensor_id=sensor_id,
            force_noise_std=0.0,
            torque_noise_std=0.0,
            force_bias_drift=0.0,
            torque_bias_drift=0.0,
        )
    )


def create_realistic_sensor(
    sensor_id: str = "ft_sensor",
    quality: str = "industrial",
    seed: int | None = None,
) -> ForceTorqueSensor:
    """Create a force/torque sensor with realistic noise.

    Args:
        sensor_id: Sensor identifier.
        quality: Sensor quality level ('research', 'industrial', 'consumer').
        seed: Random seed.

    Returns:
        ForceTorqueSensor with appropriate noise characteristics.
    """
    noise_params = {
        "research": {
            "force_noise_std": 0.01,
            "torque_noise_std": 0.001,
            "force_bias_drift": 0.0001,
            "torque_bias_drift": 0.00001,
        },
        "industrial": {
            "force_noise_std": 0.1,
            "torque_noise_std": 0.01,
            "force_bias_drift": 0.001,
            "torque_bias_drift": 0.0001,
        },
        "consumer": {
            "force_noise_std": 1.0,
            "torque_noise_std": 0.1,
            "force_bias_drift": 0.01,
            "torque_bias_drift": 0.001,
        },
    }

    params = noise_params.get(quality, noise_params["industrial"])

    return ForceTorqueSensor(
        ForceTorqueSensorConfig(
            sensor_id=sensor_id,
            seed=seed,
            **params,
        )
    )
