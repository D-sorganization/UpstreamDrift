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

    def test_bandwidth_limited_noise_order2_slower_than_order1(self) -> None:
        """Higher-order filter has steeper roll-off / slower step response."""
        order1 = BandwidthLimitedNoise(
            cutoff_frequency=10.0, sample_rate=100.0, order=1
        )
        order2 = BandwidthLimitedNoise(
            cutoff_frequency=10.0, sample_rate=100.0, order=2
        )

        outputs1: list[float] = []
        outputs2: list[float] = []
        for i in range(50):
            signal = np.array([1.0]) if i > 0 else np.array([0.0])
            outputs1.append(order1.apply(signal)[0])
            outputs2.append(order2.apply(signal)[0])

        arr1 = np.array(outputs1)
        arr2 = np.array(outputs2)

        # 2nd-order filter should respond more slowly to a step than 1st-order
        # at early samples, the 2nd-order output must lag behind the 1st-order
        assert (
            arr2[3] < arr1[3]
        ), "order=2 filter should be slower than order=1 at early samples"
        # Both should eventually converge toward 1.0
        assert arr2[-1] > 0.8

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
