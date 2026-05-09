"""Tests for src.shared.python.perturbation.noise (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.perturbation.noise import generate_noise

# ---------------------------------------------------------------------------
# Basic shape / type guarantees
# ---------------------------------------------------------------------------


class TestGenerateNoiseShape:
    def test_white_returns_correct_shape(self) -> None:
        result = generate_noise("white", 100, 1.0)
        assert result.shape == (100,)

    def test_pink_returns_correct_shape(self) -> None:
        result = generate_noise("pink", 64, 1.0)
        assert result.shape == (64,)

    def test_brown_returns_correct_shape(self) -> None:
        result = generate_noise("brown", 50, 1.0)
        assert result.shape == (50,)

    def test_noise_returns_ndarray(self) -> None:
        result = generate_noise("white", 10, 1.0)
        assert isinstance(result, np.ndarray)

    def test_single_sample(self) -> None:
        result = generate_noise("white", 1, 1.0)
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# Amplitude / seeding
# ---------------------------------------------------------------------------


class TestGenerateNoiseAmplitude:
    def test_zero_amplitude_white(self) -> None:
        result = generate_noise("white", 50, 0.0, seed=0)
        assert np.allclose(result, 0.0)

    def test_zero_amplitude_brown(self) -> None:
        result = generate_noise("brown", 50, 0.0, seed=0)
        assert np.allclose(result, 0.0)

    def test_seed_reproducible_white(self) -> None:
        a = generate_noise("white", 20, 1.0, seed=42)
        b = generate_noise("white", 20, 1.0, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_seed_reproducible_pink(self) -> None:
        a = generate_noise("pink", 32, 1.0, seed=7)
        b = generate_noise("pink", 32, 1.0, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_noise_different_seeds_differ(self) -> None:
        a = generate_noise("white", 20, 1.0, seed=1)
        b = generate_noise("white", 20, 1.0, seed=2)
        assert not np.allclose(a, b)

    def test_white_std_approx_amplitude(self) -> None:
        # With enough samples, std should be near the amplitude
        result = generate_noise("white", 10_000, 2.0, seed=0)
        assert abs(np.std(result) - 2.0) < 0.1


# ---------------------------------------------------------------------------
# Contract violations
# ---------------------------------------------------------------------------


class TestGenerateNoiseContracts:
    def test_zero_n_samples_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            generate_noise("white", 0, 1.0)

    def test_negative_n_samples_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            generate_noise("white", -5, 1.0)

    def test_noise_negative_amplitude_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            generate_noise("white", 10, -1.0)

    def test_unknown_noise_type_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            generate_noise("purple", 10, 1.0)
