"""DbC runtime contract tests for signal_toolkit filters, noise, and limits.

Verifies require() contracts on filter design, noise generation,
exponential/gaussian smoothing, saturation, rate limiting, and deadband.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.shared.python.core.contracts import PreconditionError
from src.shared.python.signal_toolkit.core import Signal
from src.shared.python.signal_toolkit.filters import (
    FilterDesigner,
    FilterType,
    apply_exponential_smoothing,
    apply_gaussian_smoothing,
)
from src.shared.python.signal_toolkit.limits import (
    apply_deadband,
    apply_rate_limiter,
    apply_saturation,
)
from src.shared.python.signal_toolkit.noise import (
    NoiseGenerator,
    NoiseType,
)


def _make_signal(
    duration: float = 1.0,
    fs: float = 1000.0,
) -> Signal:
    """Create a simple sinusoidal test signal."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    values = np.sin(2 * np.pi * t)
    return Signal(time=t, values=values, name="test_sin", units="V")


# ── Filter design contracts ────────────────────────────────────────


class TestFilterDesignerPreconditions(unittest.TestCase):
    """Test require() contracts on FilterDesigner."""

    def test_butterworth_order_must_be_positive(self) -> None:
        with self.assertRaises(PreconditionError):
            FilterDesigner.butterworth(
                FilterType.LOWPASS, cutoff=100.0, fs=1000.0, order=0
            )

    def test_butterworth_fs_must_be_positive(self) -> None:
        with self.assertRaises(PreconditionError):
            FilterDesigner.butterworth(
                FilterType.LOWPASS, cutoff=100.0, fs=0.0, order=4
            )

    def test_butterworth_negative_fs_rejected(self) -> None:
        with self.assertRaises(PreconditionError):
            FilterDesigner.butterworth(
                FilterType.LOWPASS, cutoff=100.0, fs=-500.0, order=4
            )

    def test_butterworth_valid_design(self) -> None:
        """Positive order and fs should succeed."""
        filt = FilterDesigner.butterworth(
            FilterType.LOWPASS, cutoff=100.0, fs=1000.0, order=4
        )
        self.assertEqual(filt.order, 4)
        self.assertEqual(filt.fs, 1000.0)


# ── Exponential smoothing contracts ────────────────────────────────


class TestExponentialSmoothingPreconditions(unittest.TestCase):
    """Test require() for alpha in (0, 1]."""

    def test_alpha_zero_rejected(self) -> None:
        sig = _make_signal()
        with self.assertRaises(PreconditionError):
            apply_exponential_smoothing(sig, alpha=0.0)

    def test_alpha_negative_rejected(self) -> None:
        sig = _make_signal()
        with self.assertRaises(PreconditionError):
            apply_exponential_smoothing(sig, alpha=-0.5)

    def test_alpha_greater_than_one_rejected(self) -> None:
        sig = _make_signal()
        with self.assertRaises(PreconditionError):
            apply_exponential_smoothing(sig, alpha=1.5)

    def test_alpha_one_accepted(self) -> None:
        """alpha=1 means no smoothing (direct passthrough)."""
        sig = _make_signal()
        result = apply_exponential_smoothing(sig, alpha=1.0)
        np.testing.assert_allclose(result.values, sig.values)

    def test_alpha_valid(self) -> None:
        sig = _make_signal()
        result = apply_exponential_smoothing(sig, alpha=0.5)
        self.assertEqual(len(result.values), len(sig.values))


# ── Gaussian smoothing contracts ───────────────────────────────────


class TestGaussianSmoothingPreconditions(unittest.TestCase):
    """Test require() for sigma > 0."""

    def test_sigma_zero_rejected(self) -> None:
        sig = _make_signal()
        with self.assertRaises(PreconditionError):
            apply_gaussian_smoothing(sig, sigma=0.0)

    def test_sigma_negative_rejected(self) -> None:
        sig = _make_signal()
        with self.assertRaises(PreconditionError):
            apply_gaussian_smoothing(sig, sigma=-1.0)

    def test_sigma_valid(self) -> None:
        sig = _make_signal()
        result = apply_gaussian_smoothing(sig, sigma=2.0)
        self.assertEqual(len(result.values), len(sig.values))


# ── Noise generation contracts ─────────────────────────────────────


class TestNoiseGeneratorPreconditions(unittest.TestCase):
    """Test require() for amplitude >= 0."""

    def test_negative_amplitude_rejected(self) -> None:
        gen = NoiseGenerator(seed=42)
        t = np.linspace(0, 1, 500)
        with self.assertRaises(PreconditionError):
            gen.generate(t, amplitude=-1.0)

    def test_zero_amplitude_accepted(self) -> None:
        """Zero amplitude should produce a zero signal."""
        gen = NoiseGenerator(seed=42)
        t = np.linspace(0, 1, 500)
        result = gen.generate(t, noise_type=NoiseType.WHITE, amplitude=0.0)
        np.testing.assert_allclose(result.values, 0.0)


class TestNoiseGeneratorPostconditions(unittest.TestCase):
    """Test ensure() for output length matching."""

    def test_white_noise_length(self) -> None:
        gen = NoiseGenerator(seed=42)
        t = np.linspace(0, 1, 500)
        result = gen.generate(t, noise_type=NoiseType.WHITE, amplitude=1.0)
        self.assertEqual(len(result.values), len(t))

    def test_pink_noise_length(self) -> None:
        gen = NoiseGenerator(seed=42)
        t = np.linspace(0, 1, 500)
        result = gen.generate(t, noise_type=NoiseType.PINK, amplitude=1.0)
        self.assertEqual(len(result.values), len(t))

    def test_brown_noise_length(self) -> None:
        gen = NoiseGenerator(seed=42)
        t = np.linspace(0, 1, 500)
        result = gen.generate(t, noise_type=NoiseType.BROWN, amplitude=1.0)
        self.assertEqual(len(result.values), len(t))

    def test_blue_noise_length(self) -> None:
        gen = NoiseGenerator(seed=42)
        t = np.linspace(0, 1, 500)
        result = gen.generate(t, noise_type=NoiseType.BLUE, amplitude=1.0)
        self.assertEqual(len(result.values), len(t))


# ── Saturation contracts ───────────────────────────────────────────


class TestSaturationPreconditions(unittest.TestCase):
    """Test require() for lower < upper."""

    def test_lower_equals_upper_rejected(self) -> None:
        sig = _make_signal()
        with self.assertRaises(PreconditionError):
            apply_saturation(sig, lower=1.0, upper=1.0)

    def test_lower_greater_than_upper_rejected(self) -> None:
        sig = _make_signal()
        with self.assertRaises(PreconditionError):
            apply_saturation(sig, lower=2.0, upper=1.0)

    def test_valid_saturation(self) -> None:
        sig = _make_signal()
        result = apply_saturation(sig, lower=-0.5, upper=0.5)
        self.assertTrue(np.all(result.values >= -0.5 - 1e-10))
        self.assertTrue(np.all(result.values <= 0.5 + 1e-10))


# ── Rate limiter contracts ─────────────────────────────────────────


class TestRateLimiterPreconditions(unittest.TestCase):
    """Test require() for max_rate > 0."""

    def test_max_rate_zero_rejected(self) -> None:
        sig = _make_signal()
        with self.assertRaises(PreconditionError):
            apply_rate_limiter(sig, max_rate=0.0)

    def test_max_rate_negative_rejected(self) -> None:
        sig = _make_signal()
        with self.assertRaises(PreconditionError):
            apply_rate_limiter(sig, max_rate=-10.0)

    def test_valid_rate_limiter(self) -> None:
        sig = _make_signal()
        result = apply_rate_limiter(sig, max_rate=100.0)
        self.assertEqual(len(result.values), len(sig.values))


# ── Deadband contracts ─────────────────────────────────────────────


class TestDeadbandPreconditions(unittest.TestCase):
    """Test require() for threshold >= 0."""

    def test_negative_threshold_rejected(self) -> None:
        sig = _make_signal()
        with self.assertRaises(PreconditionError):
            apply_deadband(sig, threshold=-0.1)

    def test_zero_threshold_accepted(self) -> None:
        """Zero deadband should be a passthrough."""
        sig = _make_signal()
        result = apply_deadband(sig, threshold=0.0, smooth=False)
        np.testing.assert_allclose(result.values, sig.values)

    def test_valid_deadband(self) -> None:
        sig = _make_signal()
        result = apply_deadband(sig, threshold=0.5, smooth=False)
        self.assertEqual(len(result.values), len(sig.values))


if __name__ == "__main__":
    unittest.main()
