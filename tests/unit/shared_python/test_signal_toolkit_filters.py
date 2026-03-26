"""Unit tests for signal_toolkit/filters.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.signal_toolkit.core import Signal
from src.shared.python.signal_toolkit.filters import (
    AdaptiveFilter,
    FilterDesign,
    FilterDesigner,
    FilterSpec,
    FilterType,
    apply_bilateral_filter,
    apply_exponential_smoothing,
    apply_filter,
    apply_gaussian_smoothing,
    apply_median_filter,
    apply_moving_average,
    apply_savgol,
    create_butterworth_filter,
    create_chebyshev_filter,
    create_moving_average_filter,
    create_savgol_filter,
)


@pytest.fixture
def fs() -> float:
    return 500.0


@pytest.fixture
def time_500(fs: float) -> np.ndarray:
    return np.arange(0, 1.0, 1.0 / fs)


@pytest.fixture
def noisy_sine(time_500: np.ndarray) -> Signal:
    """5 Hz sine + 100 Hz noise."""
    clean = np.sin(2 * np.pi * 5.0 * time_500)
    noise = 0.1 * np.sin(2 * np.pi * 100.0 * time_500)
    return Signal(time=time_500, values=clean + noise, name="noisy_sine", units="V")


@pytest.fixture
def butter_lowpass(fs: float) -> FilterSpec:
    return FilterDesigner.butterworth(FilterType.LOWPASS, cutoff=20.0, fs=fs, order=4)


class TestFilterDesigner:
    """Tests for the FilterDesigner factory class."""

    def test_butterworth_lowpass(self, fs: float) -> None:
        spec = FilterDesigner.butterworth(FilterType.LOWPASS, cutoff=20.0, fs=fs)
        assert spec.filter_type == FilterType.LOWPASS
        assert spec.design == FilterDesign.BUTTERWORTH
        assert len(spec.b) > 0

    def test_butterworth_highpass(self, fs: float) -> None:
        spec = FilterDesigner.butterworth(FilterType.HIGHPASS, cutoff=50.0, fs=fs)
        assert spec.filter_type == FilterType.HIGHPASS

    def test_butterworth_bandpass(self, fs: float) -> None:
        spec = FilterDesigner.butterworth(FilterType.BANDPASS, cutoff=(10.0, 80.0), fs=fs)
        assert spec.filter_type == FilterType.BANDPASS

    def test_butterworth_bandstop(self, fs: float) -> None:
        spec = FilterDesigner.butterworth(FilterType.BANDSTOP, cutoff=(45.0, 55.0), fs=fs)
        assert spec.filter_type == FilterType.BANDSTOP

    def test_butterworth_notch(self, fs: float) -> None:
        spec = FilterDesigner.butterworth(FilterType.NOTCH, cutoff=(49.0, 51.0), fs=fs)
        assert spec.filter_type == FilterType.NOTCH

    def test_chebyshev1(self, fs: float) -> None:
        spec = FilterDesigner.chebyshev1(FilterType.LOWPASS, cutoff=20.0, fs=fs)
        assert spec.design == FilterDesign.CHEBYSHEV1

    def test_chebyshev2(self, fs: float) -> None:
        spec = FilterDesigner.chebyshev2(FilterType.LOWPASS, cutoff=20.0, fs=fs)
        assert spec.design == FilterDesign.CHEBYSHEV2

    def test_elliptic(self, fs: float) -> None:
        spec = FilterDesigner.elliptic(FilterType.LOWPASS, cutoff=20.0, fs=fs)
        assert spec.design == FilterDesign.ELLIPTIC

    def test_bessel(self, fs: float) -> None:
        spec = FilterDesigner.bessel(FilterType.LOWPASS, cutoff=20.0, fs=fs)
        assert spec.design == FilterDesign.BESSEL

    def test_invalid_order_raises(self, fs: float) -> None:
        with pytest.raises((ValueError, AssertionError)):
            FilterDesigner.butterworth(FilterType.LOWPASS, cutoff=20.0, fs=fs, order=0)

    def test_invalid_fs_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            FilterDesigner.butterworth(FilterType.LOWPASS, cutoff=20.0, fs=-1.0)

    def test_bandpass_scalar_cutoff_raises(self, fs: float) -> None:
        with pytest.raises(ValueError):
            FilterDesigner.butterworth(FilterType.BANDPASS, cutoff=20.0, fs=fs)


class TestFilterSpec:
    """Tests for FilterSpec methods."""

    def test_get_frequency_response(self, butter_lowpass: FilterSpec) -> None:
        freqs, mag, phase = butter_lowpass.get_frequency_response()
        assert len(freqs) == 512
        assert len(mag) == 512

    def test_get_impulse_response(self, butter_lowpass: FilterSpec) -> None:
        t, resp = butter_lowpass.get_impulse_response(num_samples=100)
        assert len(t) == 100
        assert len(resp) == 100


class TestApplyFilter:
    """Tests for apply_filter function."""

    def test_lowpass_removes_hf(self, noisy_sine: Signal, butter_lowpass: FilterSpec) -> None:
        filtered = apply_filter(noisy_sine, butter_lowpass)
        assert filtered.values.shape == noisy_sine.values.shape
        assert "filter_type" in filtered.metadata

    def test_causal_filtering(self, noisy_sine: Signal, butter_lowpass: FilterSpec) -> None:
        filtered = apply_filter(noisy_sine, butter_lowpass, zero_phase=False)
        assert filtered.values.shape == noisy_sine.values.shape

    def test_filtered_name(self, noisy_sine: Signal, butter_lowpass: FilterSpec) -> None:
        filtered = apply_filter(noisy_sine, butter_lowpass)
        assert "_filtered" in filtered.name


class TestConvenienceFunctions:
    """Tests for convenience wrapper functions."""

    def test_create_butterworth_filter(self, fs: float) -> None:
        spec = create_butterworth_filter("lowpass", 20.0, fs)
        assert spec.design == FilterDesign.BUTTERWORTH

    def test_create_chebyshev_filter(self, fs: float) -> None:
        spec = create_chebyshev_filter("lowpass", 20.0, fs)
        assert spec.design == FilterDesign.CHEBYSHEV1

    def test_create_moving_average_filter(self) -> None:
        ma = create_moving_average_filter(5)
        y = ma(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert len(y) == 5

    def test_create_savgol_filter_even_window(self) -> None:
        # Even window should be incremented to odd
        sg = create_savgol_filter(window_length=10, polyorder=3)
        data = np.sin(np.linspace(0, 4 * np.pi, 200))
        result = sg(data)
        assert result.shape == data.shape

    def test_create_savgol_filter_short_data(self) -> None:
        sg = create_savgol_filter(window_length=51, polyorder=3)
        data = np.array([1.0, 2.0, 3.0])
        result = sg(data)
        # Fallback returns unmodified for too-short data
        assert np.allclose(result, data)


class TestSmoothingFunctions:
    """Tests for various signal smoothing functions."""

    def test_apply_moving_average(self, noisy_sine: Signal) -> None:
        smoothed = apply_moving_average(noisy_sine, window_size=10)
        assert smoothed.values.shape == noisy_sine.values.shape
        assert "ma10" in smoothed.name

    def test_apply_savgol(self, noisy_sine: Signal) -> None:
        smoothed = apply_savgol(noisy_sine, window_length=11, polyorder=3)
        assert smoothed.values.shape == noisy_sine.values.shape

    def test_apply_savgol_even_window(self, noisy_sine: Signal) -> None:
        # Even window should be auto-incremented
        smoothed = apply_savgol(noisy_sine, window_length=12, polyorder=3)
        assert smoothed.values.shape == noisy_sine.values.shape

    def test_apply_savgol_short_signal_returns_copy(self) -> None:
        t = np.linspace(0, 1, 5)
        sig = Signal(time=t, values=t)
        result = apply_savgol(sig, window_length=51)
        assert np.allclose(result.values, sig.values)

    def test_apply_median_filter(self, noisy_sine: Signal) -> None:
        filtered = apply_median_filter(noisy_sine, kernel_size=5)
        assert filtered.values.shape == noisy_sine.values.shape

    def test_apply_median_filter_even_kernel(self, noisy_sine: Signal) -> None:
        filtered = apply_median_filter(noisy_sine, kernel_size=4)
        assert filtered.values.shape == noisy_sine.values.shape

    def test_apply_exponential_smoothing(self, noisy_sine: Signal) -> None:
        smoothed = apply_exponential_smoothing(noisy_sine, alpha=0.3)
        assert smoothed.values.shape == noisy_sine.values.shape
        assert smoothed.values[0] == noisy_sine.values[0]

    def test_apply_exponential_smoothing_invalid_alpha(self, noisy_sine: Signal) -> None:
        with pytest.raises((ValueError, AssertionError)):
            apply_exponential_smoothing(noisy_sine, alpha=0.0)

    def test_apply_gaussian_smoothing(self, noisy_sine: Signal) -> None:
        smoothed = apply_gaussian_smoothing(noisy_sine, sigma=2.0)
        assert smoothed.values.shape == noisy_sine.values.shape

    def test_apply_gaussian_invalid_sigma(self, noisy_sine: Signal) -> None:
        with pytest.raises((ValueError, AssertionError)):
            apply_gaussian_smoothing(noisy_sine, sigma=-1.0)

    def test_apply_bilateral_filter(self, noisy_sine: Signal) -> None:
        filtered = apply_bilateral_filter(noisy_sine, window_size=5)
        assert filtered.values.shape == noisy_sine.values.shape


class TestAdaptiveFilter:
    """Tests for AdaptiveFilter (LMS and RLS)."""

    def test_lms_basic(self, time_500: np.ndarray) -> None:
        signal = Signal(
            time=time_500,
            values=np.sin(2 * np.pi * 5.0 * time_500),
            name="sig",
        )
        ref = Signal(
            time=time_500,
            values=np.sin(2 * np.pi * 5.0 * time_500),
            name="ref",
        )
        filtered, error = AdaptiveFilter.lms(signal, ref, order=10)
        assert filtered.values.shape == time_500.shape
        assert error.values.shape == time_500.shape

    def test_rls_basic(self, time_500: np.ndarray) -> None:
        signal = Signal(
            time=time_500,
            values=np.sin(2 * np.pi * 5.0 * time_500),
            name="sig",
        )
        ref = Signal(
            time=time_500,
            values=np.sin(2 * np.pi * 5.0 * time_500),
            name="ref",
        )
        filtered, error = AdaptiveFilter.rls(signal, ref, order=10)
        assert filtered.values.shape == time_500.shape
        assert error.values.shape == time_500.shape
