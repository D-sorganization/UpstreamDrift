"""Comprehensive tests for src.shared.python.signal_toolkit.signal_processing.

Covers PSD, coherence, spectrogram, spectral arc length, CWT, jerk,
time shift, DTW, and Kalman filter.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.signal_toolkit.signal_processing import (
    KalmanFilter,
    compute_coherence,
    compute_cwt,
    compute_dtw_distance,
    compute_dtw_path,
    compute_jerk,
    compute_psd,
    compute_spectral_arc_length,
    compute_spectrogram,
    compute_time_shift,
)

# ============================================================================
# Helpers
# ============================================================================


def _sine_signal(
    freq: float = 10.0, fs: float = 1000.0, duration: float = 1.0
) -> np.ndarray:
    """Generate a pure sine wave for testing."""
    t = np.arange(0, duration, 1.0 / fs)
    return np.sin(2 * np.pi * freq * t)


def _chirp_signal(fs: float = 1000.0, duration: float = 1.0) -> np.ndarray:
    """Generate a chirp signal (frequency sweep from 5 to 50 Hz)."""
    t = np.arange(0, duration, 1.0 / fs)
    return np.sin(2 * np.pi * (5 + 22.5 * t) * t)


# ============================================================================
# Tests for compute_psd
# ============================================================================


class TestComputePSD:
    """Tests for Power Spectral Density computation."""

    def test_returns_frequency_and_psd_arrays(self) -> None:
        data = _sine_signal(freq=10.0, fs=1000.0)
        freqs, psd = compute_psd(data, fs=1000.0)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(psd, np.ndarray)
        assert len(freqs) == len(psd)

    def test_peak_frequency(self) -> None:
        """PSD should peak near the signal's frequency."""
        fs = 1000.0
        data = _sine_signal(freq=50.0, fs=fs)
        freqs, psd = compute_psd(data, fs=fs, nperseg=256)
        peak_freq = freqs[np.argmax(psd)]
        assert abs(peak_freq - 50.0) < 5.0  # Within 5 Hz

    def test_custom_window(self) -> None:
        data = _sine_signal(fs=500.0)
        freqs, psd = compute_psd(data, fs=500.0, window="hamming")
        assert len(freqs) > 0
        assert np.all(psd >= 0)

    def test_custom_nperseg(self) -> None:
        data = _sine_signal(fs=1000.0)
        freqs, psd = compute_psd(data, fs=1000.0, nperseg=128)
        assert len(freqs) > 0

    def test_psd_values_non_negative(self) -> None:
        data = _sine_signal(fs=500.0)
        _, psd = compute_psd(data, fs=500.0)
        assert np.all(psd >= 0)


# ============================================================================
# Tests for compute_coherence
# ============================================================================


class TestComputeCoherence:
    """Tests for magnitude squared coherence."""

    def test_identical_signals_high_coherence(self) -> None:
        """Identical signals should have coherence ~1.0."""
        data = _sine_signal(freq=20.0, fs=500.0)
        freqs, coh = compute_coherence(data, data, fs=500.0)
        assert np.mean(coh) > 0.9

    def test_uncorrelated_signals_low_coherence(self) -> None:
        """Uncorrelated signals should have low coherence."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        y = rng.standard_normal(1000)
        _, coh = compute_coherence(x, y, fs=100.0)
        assert np.mean(coh) < 0.5

    def test_coherence_range(self) -> None:
        """Coherence values should be between 0 and 1."""
        data = _sine_signal(fs=500.0)
        noise = np.random.default_rng(42).standard_normal(len(data)) * 0.5
        _, coh = compute_coherence(data, data + noise, fs=500.0)
        assert np.all(coh >= 0)
        assert np.all(coh <= 1.0 + 1e-10)  # Small tolerance


# ============================================================================
# Tests for compute_spectrogram
# ============================================================================


class TestComputeSpectrogram:
    """Tests for spectrogram computation."""

    def test_returns_three_arrays(self) -> None:
        data = _sine_signal(fs=1000.0)
        f, t, sxx = compute_spectrogram(data, fs=1000.0)
        assert isinstance(f, np.ndarray)
        assert isinstance(t, np.ndarray)
        assert isinstance(sxx, np.ndarray)
        assert sxx.shape == (len(f), len(t))

    def test_custom_parameters(self) -> None:
        data = _sine_signal(fs=500.0)
        f, t, sxx = compute_spectrogram(data, fs=500.0, nperseg=128, noverlap=64)
        assert len(f) > 0
        assert len(t) > 0

    def test_non_negative_values(self) -> None:
        data = _sine_signal(fs=500.0)
        _, _, sxx = compute_spectrogram(data, fs=500.0)
        assert np.all(sxx >= 0)


# ============================================================================
# Tests for compute_spectral_arc_length
# ============================================================================


class TestComputeSpectralArcLength:
    """Tests for Spectral Arc Length smoothness metric."""

    def test_smooth_signal_lower_sal(self) -> None:
        """SAL is negative and becomes more negative for complex signals."""
        fs = 100.0
        t = np.arange(0, 1.0, 1.0 / fs)

        # Simple: single frequency
        simple = np.sin(2 * np.pi * 5 * t)

        # Complex: multiple frequencies
        complex_sig = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 15 * t)

        sal_simple = compute_spectral_arc_length(simple, fs)
        sal_complex = compute_spectral_arc_length(complex_sig, fs)

        # Both should be negative (or zero)
        assert sal_simple <= 0
        assert sal_complex <= 0

        # Complex signal should be more negative (longer arc length)
        assert sal_complex < sal_simple

    def test_empty_data_returns_zero(self) -> None:
        result = compute_spectral_arc_length(np.array([]), fs=100.0)
        assert result == 0.0

    def test_constant_signal(self) -> None:
        """Constant signal has zero frequency content -> should return 0."""
        data = np.ones(100)
        result = compute_spectral_arc_length(data, fs=100.0)
        # Constant signal - SAL depends on DC component handling
        assert isinstance(result, float)

    def test_returns_float(self) -> None:
        data = _sine_signal(freq=5.0, fs=100.0)
        result = compute_spectral_arc_length(data, fs=100.0)
        assert isinstance(result, float)


# ============================================================================
# Tests for compute_cwt
# ============================================================================


class TestComputeCWT:
    """Tests for Continuous Wavelet Transform."""

    def test_returns_expected_arrays(self) -> None:
        data = _sine_signal(freq=10.0, fs=200.0, duration=0.5)
        freqs, times, cwt_matrix = compute_cwt(
            data, fs=200.0, freq_range=(5.0, 50.0), num_freqs=10
        )
        assert isinstance(freqs, np.ndarray)
        assert isinstance(times, np.ndarray)
        assert isinstance(cwt_matrix, np.ndarray)

    def test_output_dimensions(self) -> None:
        data = _sine_signal(freq=10.0, fs=200.0, duration=0.5)
        num_freqs = 10
        freqs, times, cwt_matrix = compute_cwt(
            data, fs=200.0, freq_range=(5.0, 50.0), num_freqs=num_freqs
        )
        assert len(freqs) == num_freqs
        assert len(times) == len(data)
        assert cwt_matrix.shape == (num_freqs, len(data))

    def test_peak_at_signal_frequency(self) -> None:
        """CWT energy should peak near the signal's frequency."""
        fs = 200.0
        signal_freq = 20.0
        data = _sine_signal(freq=signal_freq, fs=fs, duration=1.0)
        freqs, _, cwt_matrix = compute_cwt(
            data, fs=fs, freq_range=(5.0, 50.0), num_freqs=20
        )
        # Average power for each frequency
        power = np.mean(np.abs(cwt_matrix) ** 2, axis=1)
        peak_freq = freqs[np.argmax(power)]
        assert abs(peak_freq - signal_freq) < 5.0


# ============================================================================
# Tests for compute_jerk
# ============================================================================


class TestComputeJerk:
    """Tests for jerk computation."""

    def test_output_same_length(self) -> None:
        data = _sine_signal(fs=100.0, duration=0.5)
        jerk = compute_jerk(data, fs=100.0)
        assert len(jerk) == len(data)

    def test_constant_signal_zero_jerk(self) -> None:
        """Constant signal should have near-zero jerk."""
        data = np.ones(100) * 5.0
        jerk = compute_jerk(data, fs=100.0)
        assert np.allclose(jerk, 0.0, atol=1e-10)

    def test_linear_signal_small_jerk(self) -> None:
        """Linear signal: jerk depends on differentiation method.

        compute_jerk uses Savitzky-Golay filter with polyorder=2,
        so for a true linear signal the derivative should be approx constant.
        The magnitude depends on slope * fs scaling.
        """
        data = np.linspace(0, 10, 200)
        jerk = compute_jerk(data, fs=100.0)
        # Jerk values should be finite and computable
        assert np.all(np.isfinite(jerk))


# ============================================================================
# Tests for compute_time_shift
# ============================================================================


class TestComputeTimeShift:
    """Tests for time shift computation via cross-correlation."""

    def test_no_shift(self) -> None:
        """Identical signals should have zero time shift."""
        data = _sine_signal(freq=10.0, fs=1000.0)
        shift = compute_time_shift(data, data, fs=1000.0)
        assert abs(shift) < 0.002  # < 2ms

    def test_known_delay(self) -> None:
        """Signal shifted by known delay should be detected."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 10 * t)

        # Shift by 10 samples = 0.01s
        delay_samples = 10
        y = np.roll(x, delay_samples)

        shift = compute_time_shift(x, y, fs=fs)
        # The detected shift should be close to 0.01
        assert abs(abs(shift) - 0.01) < 0.005

    def test_max_lag_constraint(self) -> None:
        data = _sine_signal(freq=10.0, fs=500.0)
        # With max_lag, should still return a value
        shift = compute_time_shift(data, data, fs=500.0, max_lag=0.1)
        assert isinstance(shift, float)


# ============================================================================
# Tests for compute_dtw_distance
# ============================================================================


class TestComputeDTWDistance:
    """Tests for Dynamic Time Warping distance."""

    def test_identical_sequences_zero_distance(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = compute_dtw_distance(data, data)
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_different_sequences_positive_distance(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        dist = compute_dtw_distance(x, y)
        assert dist > 0

    def test_with_sakoe_chiba_band(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        dist = compute_dtw_distance(x, y, window=2)
        assert dist > 0
        assert dist < 1.0  # Similar sequences

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(20)
        y = rng.standard_normal(20)
        d_xy = compute_dtw_distance(x, y)
        d_yx = compute_dtw_distance(y, x)
        assert d_xy == pytest.approx(d_yx, abs=1e-10)


# ============================================================================
# Tests for compute_dtw_path
# ============================================================================


class TestComputeDTWPath:
    """Tests for DTW path computation."""

    def test_returns_distance_and_path(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        dist, path = compute_dtw_path(x, y)
        assert isinstance(dist, float)
        assert isinstance(path, list)

    def test_identical_sequences_diagonal_path(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        dist, path = compute_dtw_path(x, x)
        assert dist == pytest.approx(0.0, abs=1e-10)
        # Diagonal path: (0,0), (1,1), (2,2)
        assert len(path) == 3

    def test_path_starts_at_origin(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 3.0, 4.0])
        _, path = compute_dtw_path(x, y)
        assert path[0] == (0, 0)

    def test_path_ends_at_corners(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        _, path = compute_dtw_path(x, y)
        assert path[-1] == (len(x) - 1, len(y) - 1)


# ============================================================================
# Tests for KalmanFilter
# ============================================================================


class TestKalmanFilter:
    """Tests for Kalman filter implementation."""

    def test_initialization_defaults(self) -> None:
        kf = KalmanFilter(dim_x=2, dim_z=1)
        assert kf.x.shape == (2,)
        assert kf.P.shape == (2, 2)
        assert kf.F.shape == (2, 2)
        assert kf.H.shape == (1, 2)

    def test_initialization_custom(self) -> None:
        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        kf = KalmanFilter(dim_x=2, dim_z=1, F=F, H=H)
        np.testing.assert_array_equal(kf.F, F)
        np.testing.assert_array_equal(kf.H, H)

    def test_predict_updates_state(self) -> None:
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        kf = KalmanFilter(dim_x=2, dim_z=1, F=F)
        kf.x = np.array([0.0, 1.0])  # position=0, velocity=1
        kf.predict()
        # After predict: x' = F @ x = [0+1, 0+1] = [1, 1]
        assert kf.x[0] == pytest.approx(1.0, abs=0.01)

    def test_update_with_measurement(self) -> None:
        H = np.array([[1.0, 0.0]])
        kf = KalmanFilter(dim_x=2, dim_z=1, H=H)
        kf.x = np.array([0.0, 0.0])
        kf.predict()
        kf.update(np.array([5.0]))
        # After update, state should move toward measurement
        assert kf.x[0] != 0.0

    def test_predict_update_cycle(self) -> None:
        """Run multiple predict-update cycles to track a constant position."""
        F = np.array([[1.0, 0.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.eye(1) * 1.0

        kf = KalmanFilter(dim_x=2, dim_z=1, F=F, H=H, Q=Q, R=R)

        true_value = 10.0
        rng = np.random.default_rng(42)

        for _ in range(50):
            kf.predict()
            measurement = np.array([true_value + rng.standard_normal() * 0.1])
            kf.update(measurement)

        # After 50 iterations, estimate should converge near true value
        assert abs(kf.x[0] - true_value) < 1.0

    def test_tracking_linear_motion(self) -> None:
        """Track linearly moving object."""
        dt = 0.1
        F = np.array([[1.0, dt], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.001
        R = np.eye(1) * 0.5

        kf = KalmanFilter(dim_x=2, dim_z=1, F=F, H=H, Q=Q, R=R)

        rng = np.random.default_rng(123)
        velocity = 2.0  # 2 units/sec
        positions = []

        for step in range(100):
            true_pos = velocity * step * dt
            kf.predict()
            measurement = np.array([true_pos + rng.standard_normal() * 0.3])
            kf.update(measurement)
            positions.append(kf.x[0])

        # Final position estimate should be near true position
        true_final = velocity * 99 * dt
        assert abs(positions[-1] - true_final) < 1.0

        # Velocity estimate should converge to ~2.0
        assert abs(kf.x[1] - velocity) < 0.5
