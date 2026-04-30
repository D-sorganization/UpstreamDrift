"""Behavioral tests for time-domain helpers and spectral helpers.

Targets:

* ``src/shared/python/signal_toolkit/_time_domain.py``
    - ``compute_jerk``
    - ``compute_time_shift``
* ``src/shared/python/signal_toolkit/_spectral_analysis.py``
    - ``compute_psd``
    - ``compute_coherence``
    - ``compute_spectrogram``
    - ``compute_spectral_arc_length``
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.signal_toolkit._spectral_analysis import (
    compute_coherence,
    compute_psd,
    compute_spectral_arc_length,
    compute_spectrogram,
)
from src.shared.python.signal_toolkit._time_domain import (
    compute_jerk,
    compute_time_shift,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# compute_jerk
# ---------------------------------------------------------------------------


class TestComputeJerk:
    def test_zero_acceleration_zero_jerk(self) -> None:
        a = np.zeros(200)
        jerk = compute_jerk(a, fs=100.0)
        assert jerk.shape == a.shape
        assert np.allclose(jerk, 0.0, atol=1e-9)

    def test_constant_acceleration_zero_jerk(self) -> None:
        a = np.full(200, 9.81)
        jerk = compute_jerk(a, fs=100.0)
        # Smooth derivative of a constant should be effectively zero.
        assert np.max(np.abs(jerk)) < 1e-9

    def test_linearly_increasing_acceleration_constant_jerk(self) -> None:
        fs = 1000.0
        t = np.arange(0, 1.0, 1 / fs)
        slope = 4.5  # m/s^3
        a = slope * t
        jerk = compute_jerk(a, fs=fs)
        # In the interior of the array (away from filter edges),
        # the savgol first derivative should equal the slope.
        interior = jerk[50:-50]
        assert np.allclose(interior, slope, atol=1e-6)

    def test_negative_fs_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_jerk(np.zeros(20), fs=-1.0)

    def test_short_data_falls_back_to_gradient(self) -> None:
        # window_len=7 default, len < 7 → uses np.gradient.
        a = np.array([0.0, 1.0, 2.0, 3.0])
        jerk = compute_jerk(a, fs=10.0)
        expected = np.gradient(a, 0.1)
        np.testing.assert_allclose(jerk, expected)

    def test_even_window_length_made_odd(self) -> None:
        # passing window_len=8 (even) should still work; internally bumped odd.
        fs = 100.0
        t = np.arange(0, 1.0, 1 / fs)
        a = 2.0 * t
        jerk = compute_jerk(a, fs=fs, window_len=8)
        assert jerk.shape == a.shape


# ---------------------------------------------------------------------------
# compute_time_shift
# ---------------------------------------------------------------------------


class TestComputeTimeShift:
    def test_zero_shift_for_identical_signals(self) -> None:
        fs = 100.0
        t = np.arange(0, 2.0, 1 / fs)
        x = np.sin(2 * np.pi * 3.0 * t)
        shift = compute_time_shift(x, x, fs=fs)
        assert shift == pytest.approx(0.0, abs=1.0 / fs)

    def test_recovers_known_delay(self) -> None:
        fs = 100.0
        t = np.arange(0, 4.0, 1 / fs)
        x = np.sin(2 * np.pi * 2.0 * t)
        delay_samples = 17  # y(t) = x(t - delay/fs)
        y = np.zeros_like(x)
        y[delay_samples:] = x[:-delay_samples]
        shift = compute_time_shift(x, y, fs=fs)
        # Returned shift should be positive (y lags x) and ≈ delay_samples/fs.
        assert shift == pytest.approx(delay_samples / fs, abs=1.0 / fs)

    def test_recovers_negative_delay(self) -> None:
        fs = 100.0
        t = np.arange(0, 4.0, 1 / fs)
        x = np.sin(2 * np.pi * 2.0 * t)
        delay_samples = 12
        # y leads x: shift should be negative.
        y = np.zeros_like(x)
        y[:-delay_samples] = x[delay_samples:]
        shift = compute_time_shift(x, y, fs=fs)
        assert shift == pytest.approx(-delay_samples / fs, abs=1.0 / fs)

    def test_constant_signal_returns_zero(self) -> None:
        # std == 0 path → returns 0.0.
        x = np.ones(100)
        y = np.ones(100)
        assert compute_time_shift(x, y, fs=50.0) == 0.0

    def test_non_positive_fs_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_time_shift(np.zeros(10), np.zeros(10), fs=0.0)

    def test_handles_unequal_lengths(self) -> None:
        # Function should truncate to the shorter length without erroring.
        fs = 50.0
        x = np.sin(np.linspace(0, 4 * np.pi, 100))
        y = np.sin(np.linspace(0, 4 * np.pi, 80))
        shift = compute_time_shift(x, y, fs=fs)
        assert isinstance(shift, float)


# ---------------------------------------------------------------------------
# compute_psd
# ---------------------------------------------------------------------------


class TestComputePsd:
    def test_dominant_frequency_appears_in_psd_peak(self) -> None:
        fs = 1000.0
        t = np.arange(0, 4.0, 1 / fs)
        f0 = 50.0
        x = np.sin(2 * np.pi * f0 * t)
        freqs, psd = compute_psd(x, fs=fs, nperseg=512)
        assert freqs.shape == psd.shape
        peak = freqs[int(np.argmax(psd))]
        assert peak == pytest.approx(f0, abs=fs / 512)

    def test_non_positive_fs_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_psd(np.array([1.0, 2.0, 3.0]), fs=0.0)


# ---------------------------------------------------------------------------
# compute_coherence
# ---------------------------------------------------------------------------


class TestComputeCoherence:
    def test_perfect_self_coherence(self) -> None:
        fs = 500.0
        t = np.arange(0, 4.0, 1 / fs)
        x = np.sin(2 * np.pi * 10.0 * t)
        freqs, coh = compute_coherence(x, x, fs=fs, nperseg=256)
        assert freqs.shape == coh.shape
        # Coherence of a signal with itself is 1 across all freqs.
        assert np.allclose(coh, 1.0, atol=1e-6)

    def test_unequal_length_raises(self) -> None:
        with pytest.raises((ValueError, IndexError)):
            compute_coherence(np.zeros(50), np.zeros(40), fs=100.0)


# ---------------------------------------------------------------------------
# compute_spectrogram
# ---------------------------------------------------------------------------


class TestComputeSpectrogram:
    def test_shape_and_axes(self) -> None:
        fs = 1000.0
        x = np.random.default_rng(0).standard_normal(2048)
        f, t, Sxx = compute_spectrogram(x, fs=fs, nperseg=256)
        assert Sxx.shape[0] == f.shape[0]
        assert Sxx.shape[1] == t.shape[0]
        assert f[-1] == pytest.approx(fs / 2)


# ---------------------------------------------------------------------------
# compute_spectral_arc_length
# ---------------------------------------------------------------------------


class TestSpectralArcLength:
    def test_zero_data_returns_zero(self) -> None:
        assert compute_spectral_arc_length(np.zeros(64), fs=100.0) == 0.0

    def test_empty_data_returns_zero(self) -> None:
        assert compute_spectral_arc_length(np.array([]), fs=100.0) == 0.0

    def test_value_is_non_positive(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(512)
        sal = compute_spectral_arc_length(x, fs=100.0)
        assert sal <= 0.0

    def test_smoother_signal_has_larger_sal(self) -> None:
        # Smoother (less spectral variation) → SAL closer to zero
        # (i.e., greater than the noisier signal's more-negative value).
        fs = 1000.0
        t = np.arange(0, 2.0, 1 / fs)
        smooth = np.sin(2 * np.pi * 1.0 * t)
        rng = np.random.default_rng(0)
        noisy = smooth + 0.5 * rng.standard_normal(smooth.size)
        sal_smooth = compute_spectral_arc_length(smooth, fs=fs)
        sal_noisy = compute_spectral_arc_length(noisy, fs=fs)
        assert sal_smooth > sal_noisy

    def test_non_positive_fs_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_spectral_arc_length(np.ones(64), fs=0.0)
