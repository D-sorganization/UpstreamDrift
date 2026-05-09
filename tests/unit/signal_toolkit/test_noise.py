"""Tests for src.shared.python.signal_toolkit.noise (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.signal_toolkit.core import Signal
from src.shared.python.signal_toolkit.noise import (
    NoiseGenerator,
    NoiseType,
    add_noise_to_signal,
)


def _make_signal(n: int = 100) -> Signal:
    t = np.linspace(0.0, 1.0, n)
    y = np.sin(2 * np.pi * t)
    return Signal(time=t, values=y, name="test", units="m")


class TestNoiseGenerator:
    def setup_method(self) -> None:
        self.gen = NoiseGenerator(seed=42)
        self.t = np.linspace(0.0, 1.0, 200)

    def test_white_noise_returns_signal(self) -> None:
        result = self.gen.generate(self.t, NoiseType.WHITE)
        assert isinstance(result, Signal)

    def test_white_noise_correct_length(self) -> None:
        result = self.gen.generate(self.t, NoiseType.WHITE)
        assert len(result.values) == len(self.t)

    def test_pink_noise_returns_signal(self) -> None:
        result = self.gen.generate(self.t, NoiseType.PINK)
        assert isinstance(result, Signal)

    def test_brown_noise_returns_signal(self) -> None:
        result = self.gen.generate(self.t, NoiseType.BROWN)
        assert isinstance(result, Signal)

    def test_blue_noise_returns_signal(self) -> None:
        result = self.gen.generate(self.t, NoiseType.BLUE)
        assert isinstance(result, Signal)

    def test_violet_noise_returns_signal(self) -> None:
        result = self.gen.generate(self.t, NoiseType.VIOLET)
        assert isinstance(result, Signal)

    def test_uniform_noise_returns_signal(self) -> None:
        result = self.gen.generate(self.t, NoiseType.UNIFORM)
        assert isinstance(result, Signal)

    def test_zero_amplitude_gives_zeros(self) -> None:
        result = self.gen.generate(self.t, NoiseType.WHITE, amplitude=0.0)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-12)

    def test_noise_negative_amplitude_raises(self) -> None:
        with pytest.raises((ValueError, TypeError, AssertionError)):
            self.gen.generate(self.t, NoiseType.WHITE, amplitude=-1.0)

    def test_noise_reproducible_with_seed(self) -> None:
        gen1 = NoiseGenerator(seed=99)
        gen2 = NoiseGenerator(seed=99)
        r1 = gen1.generate(self.t, NoiseType.WHITE, amplitude=1.0)
        r2 = gen2.generate(self.t, NoiseType.WHITE, amplitude=1.0)
        np.testing.assert_array_equal(r1.values, r2.values)

    def test_noise_different_seeds_differ(self) -> None:
        gen1 = NoiseGenerator(seed=1)
        gen2 = NoiseGenerator(seed=2)
        r1 = gen1.generate(self.t, NoiseType.WHITE, amplitude=1.0)
        r2 = gen2.generate(self.t, NoiseType.WHITE, amplitude=1.0)
        assert not np.allclose(r1.values, r2.values)

    def test_noise_all_values_finite(self) -> None:
        for noise_type in [NoiseType.WHITE, NoiseType.PINK, NoiseType.BROWN]:
            result = self.gen.generate(self.t, noise_type, amplitude=0.5)
            assert np.all(np.isfinite(result.values)), f"Non-finite in {noise_type}"


class TestAddNoiseToSignal:
    def setup_method(self) -> None:
        self.sig = _make_signal(n=200)

    def test_noise_returns_signal(self) -> None:
        result = add_noise_to_signal(self.sig, amplitude=0.1, seed=0)
        assert isinstance(result, Signal)

    def test_output_same_length(self) -> None:
        result = add_noise_to_signal(self.sig, amplitude=0.1, seed=0)
        assert len(result.values) == len(self.sig.values)

    def test_noisy_signal_differs_from_original(self) -> None:
        result = add_noise_to_signal(self.sig, amplitude=0.5, seed=7)
        assert not np.allclose(result.values, self.sig.values)

    def test_noise_name_updated(self) -> None:
        result = add_noise_to_signal(self.sig, amplitude=0.1)
        assert "noisy" in result.name

    def test_snr_mode_runs(self) -> None:
        result = add_noise_to_signal(self.sig, snr_db=20.0, seed=0)
        assert isinstance(result, Signal)

    def test_metadata_has_noise_type(self) -> None:
        result = add_noise_to_signal(self.sig, amplitude=0.1, seed=0)
        assert "noise_type" in result.metadata

    def test_zero_amplitude_preserves_signal(self) -> None:
        result = add_noise_to_signal(self.sig, amplitude=0.0, seed=0)
        np.testing.assert_allclose(result.values, self.sig.values, atol=1e-12)
