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


class TestNoisySensor:
    """Tests for SensorNoiseParameters / NoisySensor wrapper."""

    def test_zero_noise_returns_raw(self) -> None:
        """With all stds set to 0, measurement equals raw signal."""
        from src.robotics.sensing.noise_models import (
            NoisySensor,
            SensorNoiseParameters,
        )

        params = SensorNoiseParameters(
            white_std=0.0,
            bias_initial_std=0.0,
            bias_walk_std=0.0,
            scale_factor_std=0.0,
        )
        sensor = NoisySensor(params, seed=0)
        raw = np.array([1.0, -2.5, 3.14])
        out = sensor.measure(raw)
        assert_allclose(out, raw)

    def test_nonzero_noise_differs_from_raw(self) -> None:
        """Non-zero parameters produce deviation from raw with expected scale."""
        from src.robotics.sensing.noise_models import (
            NoisySensor,
            SensorNoiseParameters,
        )

        params = SensorNoiseParameters(
            white_std=0.1,
            bias_initial_std=0.0,
            bias_walk_std=0.0,
            scale_factor_std=0.0,
        )
        sensor = NoisySensor(params, seed=42)
        raw = np.zeros(5000)
        out = sensor.measure(raw)
        # Roughly mean 0, std ~0.1
        assert abs(float(np.mean(out))) < 0.01
        assert 0.08 < float(np.std(out)) < 0.12

    def test_bias_walk_drift_accumulates(self) -> None:
        """Bias random walk std should grow as sqrt(n_steps)."""
        from src.robotics.sensing.noise_models import (
            NoisySensor,
            SensorNoiseParameters,
        )

        params = SensorNoiseParameters(
            white_std=0.0,
            bias_initial_std=0.0,
            bias_walk_std=0.01,
            scale_factor_std=0.0,
        )
        sensor = NoisySensor(params, seed=7)
        raw = np.array([0.0])
        # Run 1000 steps, collect final bias magnitudes across many trials
        finals = []
        for trial in range(200):
            sensor = NoisySensor(params, seed=1000 + trial)
            for _ in range(1000):
                sensor.measure(raw)
            finals.append(float(np.asarray(sensor.current_bias).item()))
        # sqrt(1000) * 0.01 ~ 0.316
        final_std = float(np.std(finals))
        assert 0.2 < final_std < 0.45

    def test_scale_factor_multiplicative(self) -> None:
        """Scale factor error scales with raw signal magnitude."""
        from src.robotics.sensing.noise_models import (
            NoisySensor,
            SensorNoiseParameters,
        )

        params = SensorNoiseParameters(
            white_std=0.0,
            bias_initial_std=0.0,
            bias_walk_std=0.0,
            scale_factor_std=0.01,
        )
        sensor = NoisySensor(params, seed=3)
        # First call materializes scale factor (shape = (1,))
        sensor.measure(np.array([1.0]))
        sf = float(np.asarray(sensor.scale_factor).item())
        out_small = sensor.measure(np.array([1.0]))
        out_large = sensor.measure(np.array([100.0]))
        assert_allclose(out_small, np.array([1.0 + sf]))
        assert_allclose(out_large, np.array([100.0 * (1.0 + sf)]))

    def test_saturation_clips_output(self) -> None:
        """Saturation limit clips measurement magnitude."""
        from src.robotics.sensing.noise_models import (
            NoisySensor,
            SensorNoiseParameters,
        )

        params = SensorNoiseParameters(
            white_std=0.0,
            bias_initial_std=0.0,
            bias_walk_std=0.0,
            scale_factor_std=0.0,
            saturation_limit=5.0,
        )
        sensor = NoisySensor(params, seed=0)
        out = sensor.measure(np.array([10.0, -10.0, 3.0]))
        assert_allclose(out, np.array([5.0, -5.0, 3.0]))

    def test_temperature_drift(self) -> None:
        """Temperature delta shifts the measurement via temp coefficient."""
        from src.robotics.sensing.noise_models import (
            NoisySensor,
            SensorNoiseParameters,
        )

        params = SensorNoiseParameters(
            white_std=0.0,
            bias_initial_std=0.0,
            bias_walk_std=0.0,
            scale_factor_std=0.0,
            temperature_coefficient=0.1,
        )
        sensor = NoisySensor(params, seed=0)
        baseline = sensor.measure(np.array([1.0]), temperature_delta=0.0)
        hot = sensor.measure(np.array([1.0]), temperature_delta=10.0)
        assert_allclose(baseline, np.array([1.0]))
        assert_allclose(hot, np.array([1.0 + 1.0]))  # 0.1 * 10

    def test_reset_restores_state(self) -> None:
        """reset() returns the sensor to a deterministic fresh state."""
        from src.robotics.sensing.noise_models import (
            NoisySensor,
            SensorNoiseParameters,
        )

        params = SensorNoiseParameters(
            white_std=0.01,
            bias_initial_std=0.01,
            bias_walk_std=1e-3,
            scale_factor_std=0.0,
        )
        sensor = NoisySensor(params, seed=99)
        raw = np.array([1.0, 2.0])
        first = sensor.measure(raw.copy())
        sensor.reset()
        again = sensor.measure(raw.copy())
        assert_allclose(first, again)

    def test_parameters_validate_non_negative(self) -> None:
        """SensorNoiseParameters rejects negative stds."""
        from src.robotics.sensing.noise_models import SensorNoiseParameters

        with pytest.raises(ValueError):
            SensorNoiseParameters(white_std=-0.1)
        with pytest.raises(ValueError):
            SensorNoiseParameters(saturation_limit=0.0)

    def test_presets_are_usable(self) -> None:
        """Published-datasheet presets construct and run without error."""
        from src.robotics.sensing.noise_models import (
            FORCE_TORQUE_INDUSTRIAL_DEFAULTS,
            IMU_MEMS_DEFAULTS,
            NoisySensor,
        )

        imu = NoisySensor(IMU_MEMS_DEFAULTS, seed=0)
        ft = NoisySensor(FORCE_TORQUE_INDUSTRIAL_DEFAULTS, seed=0)
        assert imu.measure(np.zeros(3)).shape == (3,)
        assert ft.measure(np.zeros(6)).shape == (6,)
