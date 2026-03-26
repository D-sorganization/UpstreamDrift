"""Unit tests for signal_toolkit/noise.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.signal_toolkit.core import Signal, SignalGenerator
from src.shared.python.signal_toolkit.noise import (
    DisturbanceSimulator,
    NoiseGenerator,
    NoiseType,
    add_noise_to_signal,
    generate_disturbance_profile,
)


@pytest.fixture
def t() -> np.ndarray:
    return np.linspace(0, 2.0, 400)


@pytest.fixture
def sine_signal(t: np.ndarray) -> Signal:
    return SignalGenerator.sinusoid(t, amplitude=2.0, frequency=3.0, name="sig")


@pytest.fixture
def gen() -> NoiseGenerator:
    return NoiseGenerator(seed=42)


class TestNoiseGenerator:
    """Tests for NoiseGenerator class."""

    def test_white_noise(self, gen: NoiseGenerator, t: np.ndarray) -> None:
        sig = gen.generate(t, noise_type=NoiseType.WHITE, amplitude=1.0)
        assert sig.values.shape == t.shape
        assert np.isclose(np.std(sig.values), 1.0, atol=0.2)

    def test_pink_noise(self, gen: NoiseGenerator, t: np.ndarray) -> None:
        sig = gen.generate(t, noise_type=NoiseType.PINK, amplitude=1.0)
        assert sig.values.shape == t.shape

    def test_brown_noise(self, gen: NoiseGenerator, t: np.ndarray) -> None:
        sig = gen.generate(t, noise_type=NoiseType.BROWN, amplitude=1.0)
        assert sig.values.shape == t.shape

    def test_blue_noise(self, gen: NoiseGenerator, t: np.ndarray) -> None:
        sig = gen.generate(t, noise_type=NoiseType.BLUE, amplitude=0.5)
        assert sig.values.shape == t.shape

    def test_violet_noise(self, gen: NoiseGenerator, t: np.ndarray) -> None:
        sig = gen.generate(t, noise_type=NoiseType.VIOLET, amplitude=0.5)
        assert sig.values.shape == t.shape

    def test_uniform_noise(self, gen: NoiseGenerator, t: np.ndarray) -> None:
        sig = gen.generate(t, noise_type=NoiseType.UNIFORM, amplitude=1.0)
        assert sig.values.shape == t.shape

    def test_impulse_noise(self, gen: NoiseGenerator, t: np.ndarray) -> None:
        sig = gen.generate(t, noise_type=NoiseType.IMPULSE, amplitude=5.0, probability=0.05)
        assert sig.values.shape == t.shape

    def test_quantization_noise(self, gen: NoiseGenerator, t: np.ndarray) -> None:
        sig = gen.generate(t, noise_type=NoiseType.QUANTIZATION, amplitude=1.0, levels=256)
        assert sig.values.shape == t.shape

    def test_periodic_noise(self, gen: NoiseGenerator, t: np.ndarray) -> None:
        sig = gen.generate(t, noise_type=NoiseType.PERIODIC, amplitude=1.0, frequency=60.0)
        assert sig.values.shape == t.shape

    def test_negative_amplitude_raises(self, gen: NoiseGenerator, t: np.ndarray) -> None:
        with pytest.raises((ValueError, AssertionError)):
            gen.generate(t, amplitude=-1.0)

    def test_metadata_set(self, gen: NoiseGenerator, t: np.ndarray) -> None:
        sig = gen.generate(t, noise_type=NoiseType.WHITE, amplitude=0.5)
        assert "noise_type" in sig.metadata
        assert "amplitude" in sig.metadata

    def test_reproducible_with_seed(self, t: np.ndarray) -> None:
        g1 = NoiseGenerator(seed=123)
        g2 = NoiseGenerator(seed=123)
        s1 = g1.generate(t, noise_type=NoiseType.WHITE)
        s2 = g2.generate(t, noise_type=NoiseType.WHITE)
        assert np.allclose(s1.values, s2.values)


class TestAddNoiseToSignal:
    """Tests for add_noise_to_signal function."""

    def test_add_noise_snr(self, sine_signal: Signal) -> None:
        noisy = add_noise_to_signal(sine_signal, snr_db=20.0, seed=42)
        assert noisy.values.shape == sine_signal.values.shape
        assert "snr_db" in noisy.metadata

    def test_add_noise_amplitude(self, sine_signal: Signal) -> None:
        noisy = add_noise_to_signal(sine_signal, amplitude=0.1, seed=0)
        assert noisy.values.shape == sine_signal.values.shape

    def test_add_noise_default_amplitude(self, sine_signal: Signal) -> None:
        """Default amplitude = 10% of std."""
        noisy = add_noise_to_signal(sine_signal, seed=5)
        assert noisy.values.shape == sine_signal.values.shape

    def test_add_noise_pink(self, sine_signal: Signal) -> None:
        noisy = add_noise_to_signal(sine_signal, noise_type=NoiseType.PINK, amplitude=0.5, seed=7)
        assert noisy.values.shape == sine_signal.values.shape

    def test_name_updated(self, sine_signal: Signal) -> None:
        noisy = add_noise_to_signal(sine_signal, amplitude=0.1, seed=1)
        assert "_noisy" in noisy.name


class TestGenerateDisturbanceProfile:
    """Tests for generate_disturbance_profile function."""

    def test_step_disturbance(self, t: np.ndarray) -> None:
        sig = generate_disturbance_profile(t, disturbance_type="step", step_time=1.0, magnitude=2.0)
        assert np.all(sig.values[t < 1.0] == 0.0)
        assert np.all(sig.values[t >= 1.0] == 2.0)

    def test_pulse_disturbance(self, t: np.ndarray) -> None:
        sig = generate_disturbance_profile(
            t, disturbance_type="pulse", start_time=0.5, duration=0.5, magnitude=3.0
        )
        assert sig.values.shape == t.shape

    def test_ramp_disturbance(self, t: np.ndarray) -> None:
        sig = generate_disturbance_profile(
            t, disturbance_type="ramp", start_time=0.0, end_time=2.0, end_value=5.0
        )
        assert np.isclose(sig.values[-1], 5.0, atol=1e-6)

    def test_sine_disturbance(self, t: np.ndarray) -> None:
        sig = generate_disturbance_profile(t, disturbance_type="sine", frequency=1.0, amplitude=2.0)
        assert np.isclose(np.max(np.abs(sig.values)), 2.0, atol=0.1)

    def test_random_steps_disturbance(self, t: np.ndarray) -> None:
        sig = generate_disturbance_profile(t, disturbance_type="random_steps", num_steps=5, seed=42)
        assert sig.values.shape == t.shape

    def test_chirp_disturbance(self, t: np.ndarray) -> None:
        sig = generate_disturbance_profile(t, disturbance_type="chirp", f0=0.5, f1=5.0)
        assert sig.values.shape == t.shape

    def test_metadata_contains_type(self, t: np.ndarray) -> None:
        sig = generate_disturbance_profile(t, disturbance_type="step")
        assert sig.metadata["disturbance_type"] == "step"


class TestDisturbanceSimulator:
    """Tests for DisturbanceSimulator class."""

    def test_add_noise_and_generate(self, t: np.ndarray) -> None:
        sim = DisturbanceSimulator(seed=42)
        sim.add_noise(noise_type=NoiseType.WHITE, amplitude=0.5)
        result = sim.generate(t)
        assert result.values.shape == t.shape

    def test_add_step(self, t: np.ndarray) -> None:
        sim = DisturbanceSimulator(seed=0)
        sim.add_step(step_time=1.0, magnitude=2.0)
        result = sim.generate(t)
        assert result.values.shape == t.shape

    def test_add_pulse(self, t: np.ndarray) -> None:
        sim = DisturbanceSimulator(seed=1)
        sim.add_pulse(start_time=0.5, duration=0.3, magnitude=1.5)
        result = sim.generate(t)
        assert result.values.shape == t.shape

    def test_add_periodic(self, t: np.ndarray) -> None:
        sim = DisturbanceSimulator(seed=2)
        sim.add_periodic(frequency=5.0, amplitude=0.3)
        result = sim.generate(t)
        assert result.values.shape == t.shape

    def test_chaining(self, t: np.ndarray) -> None:
        sim = DisturbanceSimulator(seed=99)
        result = (
            sim.add_noise(amplitude=0.1)
            .add_step(step_time=1.0)
            .add_pulse(start_time=0.5, duration=0.2)
            .generate(t)
        )
        assert result.values.shape == t.shape

    def test_apply_to_signal(self, sine_signal: Signal) -> None:
        sim = DisturbanceSimulator(seed=7)
        sim.add_noise(amplitude=0.2)
        disturbed = sim.apply_to_signal(sine_signal)
        assert disturbed.values.shape == sine_signal.values.shape
        assert "_disturbed" in disturbed.name

    def test_empty_disturbances(self, t: np.ndarray) -> None:
        """No disturbances → all zeros."""
        sim = DisturbanceSimulator()
        result = sim.generate(t)
        assert np.all(result.values == 0.0)
