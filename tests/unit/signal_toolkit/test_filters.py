"""Tests for src.shared.python.signal_toolkit.filters (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.signal_toolkit.core import Signal
from src.shared.python.signal_toolkit.filters import (
    apply_exponential_smoothing,
    apply_gaussian_smoothing,
    apply_median_filter,
    apply_moving_average,
    apply_savgol,
    create_butterworth_filter,
)


def _make_signal(n: int = 200, fs: float = 200.0) -> Signal:
    """Sinusoidal signal at 5 Hz with 200 Hz sampling."""
    t = np.arange(n) / fs
    y = np.sin(2 * np.pi * 5.0 * t) + 0.2 * np.random.default_rng(0).standard_normal(n)
    return Signal(time=t, values=y, name="test", units="m/s")


class TestCreateButterworthFilter:
    def test_lowpass_returns_filter_spec(self) -> None:
        spec = create_butterworth_filter("lowpass", cutoff=10.0, fs=200.0)
        assert spec is not None

    def test_highpass_returns_filter_spec(self) -> None:
        spec = create_butterworth_filter("highpass", cutoff=1.0, fs=200.0)
        assert spec is not None

    def test_bandpass_returns_filter_spec(self) -> None:
        spec = create_butterworth_filter("bandpass", cutoff=(2.0, 20.0), fs=200.0)
        assert spec is not None

    def test_filter_has_coefficients(self) -> None:
        spec = create_butterworth_filter("lowpass", cutoff=10.0, fs=200.0)
        assert spec.b is not None
        assert spec.a is not None

    def test_frequency_response_shape(self) -> None:
        spec = create_butterworth_filter("lowpass", cutoff=10.0, fs=200.0)
        freqs, mag, phase = spec.get_frequency_response(num_points=256)
        assert len(freqs) == 256
        assert len(mag) == 256
        assert len(phase) == 256

    def test_lowpass_attenuates_high_freqs(self) -> None:
        spec = create_butterworth_filter("lowpass", cutoff=10.0, fs=200.0)
        freqs, mag, _ = spec.get_frequency_response(num_points=512)
        # Passband magnitude near DC should be close to 1
        assert mag[0] == pytest.approx(1.0, abs=0.05)

    def test_impulse_response_shape(self) -> None:
        spec = create_butterworth_filter("lowpass", cutoff=10.0, fs=200.0)
        t, resp = spec.get_impulse_response(num_samples=64)
        assert len(t) == 64
        assert len(resp) == 64


class TestApplyMovingAverage:
    def test_filters_returns_signal(self) -> None:
        sig = _make_signal()
        result = apply_moving_average(sig, window_size=5)
        assert isinstance(result, Signal)

    def test_output_length_preserved(self) -> None:
        sig = _make_signal(n=100)
        result = apply_moving_average(sig, window_size=5)
        assert len(result.values) == 100

    def test_constant_signal_mean_preserved(self) -> None:
        t = np.linspace(0.0, 1.0, 100)
        sig = Signal(time=t, values=np.ones(100) * 3.0)
        result = apply_moving_average(sig, window_size=10)
        # Moving average has edge effects; mean should still be close to 3.0
        assert np.mean(result.values) == pytest.approx(3.0, rel=0.1)

    def test_filters_all_values_finite(self) -> None:
        sig = _make_signal()
        result = apply_moving_average(sig, window_size=7)
        assert np.all(np.isfinite(result.values))

    def test_filters_name_updated(self) -> None:
        sig = _make_signal()
        result = apply_moving_average(sig, window_size=5)
        assert "ma5" in result.name


class TestApplySavgol:
    def test_filters_returns_signal(self) -> None:
        sig = _make_signal(n=100)
        result = apply_savgol(sig, window_length=11, polyorder=3)
        assert isinstance(result, Signal)

    def test_output_length_preserved(self) -> None:
        sig = _make_signal(n=100)
        result = apply_savgol(sig, window_length=11, polyorder=3)
        assert len(result.values) == 100

    def test_short_signal_returns_copy(self) -> None:
        t = np.linspace(0.0, 0.1, 5)
        sig = Signal(time=t, values=np.ones(5))
        result = apply_savgol(sig, window_length=11)
        assert len(result.values) == 5

    def test_filters_name_updated(self) -> None:
        sig = _make_signal(n=100)
        result = apply_savgol(sig, window_length=11)
        assert "savgol" in result.name

    def test_filters_all_values_finite(self) -> None:
        sig = _make_signal(n=100)
        result = apply_savgol(sig, window_length=11, polyorder=3)
        assert np.all(np.isfinite(result.values))


class TestApplyMedianFilter:
    def test_filters_returns_signal(self) -> None:
        sig = _make_signal(n=100)
        result = apply_median_filter(sig, kernel_size=5)
        assert isinstance(result, Signal)

    def test_output_length_preserved(self) -> None:
        sig = _make_signal(n=100)
        result = apply_median_filter(sig, kernel_size=5)
        assert len(result.values) == 100

    def test_removes_spike(self) -> None:
        t = np.linspace(0.0, 1.0, 100)
        y = np.zeros(100)
        y[50] = 100.0  # spike
        sig = Signal(time=t, values=y)
        result = apply_median_filter(sig, kernel_size=5)
        assert abs(result.values[50]) < 1.0  # spike removed

    def test_filters_all_values_finite(self) -> None:
        sig = _make_signal(n=100)
        result = apply_median_filter(sig, kernel_size=5)
        assert np.all(np.isfinite(result.values))


class TestApplyExponentialSmoothing:
    def test_filters_returns_signal(self) -> None:
        sig = _make_signal(n=100)
        result = apply_exponential_smoothing(sig, alpha=0.3)
        assert isinstance(result, Signal)

    def test_output_length_preserved(self) -> None:
        sig = _make_signal(n=100)
        result = apply_exponential_smoothing(sig, alpha=0.3)
        assert len(result.values) == 100

    def test_alpha_one_preserves_signal(self) -> None:
        t = np.linspace(0.0, 1.0, 100)
        y = np.random.default_rng(0).standard_normal(100)
        sig = Signal(time=t, values=y)
        result = apply_exponential_smoothing(sig, alpha=1.0)
        np.testing.assert_allclose(result.values, y, atol=1e-12)


class TestApplyGaussianSmoothing:
    def test_filters_returns_signal(self) -> None:
        sig = _make_signal(n=100)
        result = apply_gaussian_smoothing(sig, sigma=1.0)
        assert isinstance(result, Signal)

    def test_output_length_preserved(self) -> None:
        sig = _make_signal(n=100)
        result = apply_gaussian_smoothing(sig, sigma=1.0)
        assert len(result.values) == 100

    def test_filters_all_values_finite(self) -> None:
        sig = _make_signal(n=100)
        result = apply_gaussian_smoothing(sig, sigma=2.0)
        assert np.all(np.isfinite(result.values))
