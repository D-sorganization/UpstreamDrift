"""Unit tests for shared signal processing."""

from unittest.mock import patch

import numpy as np
import pytest

from shared.python.signal_processing import (
    KalmanFilter,
    compute_cwt,
    compute_dtw_distance,
    compute_dtw_path,
    compute_jerk,
    compute_psd,
    compute_spectral_arc_length,
    compute_spectrogram,
    compute_time_shift,
    compute_xwt,
)


class TestSignalProcessing:
    """Test cases for signal processing utilities."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Set up test data."""
        self.fs = 1000.0  # 1 kHz sampling
        self.t = np.arange(0, 1.0, 1 / self.fs)
        # Create a signal with 10 Hz and 50 Hz components
        self.signal = np.sin(2 * np.pi * 10 * self.t) + 0.5 * np.sin(
            2 * np.pi * 50 * self.t
        )

    def test_compute_psd_logic(self):
        """Test PSD computation logic (peaks)."""
        # Use boxcar window for sharper peaks
        freqs, psd = compute_psd(self.signal, self.fs, window="boxcar", nperseg=256)

        # Check output shapes
        assert len(freqs) == 129  # nperseg/2 + 1
        assert len(psd) == 129

        # Let's verify peak locations roughly
        peak_indices = np.where(psd > np.max(psd) * 0.1)[0]
        peak_freqs = freqs[peak_indices]

        has_10 = np.any(np.abs(peak_freqs - 10) < 5)
        has_50 = np.any(np.abs(peak_freqs - 50) < 5)

        assert has_10, f"Should have peak near 10 Hz. Found: {peak_freqs}"
        assert has_50, f"Should have peak near 50 Hz. Found: {peak_freqs}"

    def test_compute_psd_mock(self):
        """Test PSD computation using mock to verify call arguments."""
        with patch("shared.python.signal_processing.welch") as mock_welch:
            rng = np.random.default_rng(42)
            mock_welch.return_value = (np.arange(129), rng.random(129))
            freqs, psd = compute_psd(self.signal, self.fs, nperseg=256)

            assert len(freqs) == 129
            assert len(psd) == 129
            assert mock_welch.called

    def test_compute_spectrogram_logic(self):
        """Test spectrogram computation output shapes."""
        f, t, Sxx = compute_spectrogram(self.signal, self.fs, nperseg=256)

        # Check shapes
        assert len(f) == 129
        assert Sxx.shape[0] == 129
        assert Sxx.shape[1] > 0

    def test_compute_spectrogram_mock(self):
        """Test Spectrogram computation using mock."""
        data = np.sin(2 * np.pi * 20 * self.t + 2 * np.pi * 40 * self.t**2)
        with patch("shared.python.signal_processing.spectrogram") as mock_spec:
            rng = np.random.default_rng(42)
            mock_spec.return_value = (
                np.arange(10),
                np.arange(10),
                rng.random((10, 10)),
            )
            f, t_spec, Sxx = compute_spectrogram(data, self.fs, nperseg=128)

            assert len(f) == 10
            assert len(t_spec) == 10
            assert Sxx.shape == (10, 10)
            assert mock_spec.called

    def test_compute_spectral_arc_length_smooth(self):
        """Test SAL on a smooth signal."""
        t = np.linspace(0, 1, 100)
        smooth_signal = np.sin(np.pi * t)

        sal = compute_spectral_arc_length(smooth_signal, fs=100.0)

        # SAL should be negative
        assert sal < 0
        # Should be a finite number
        assert np.isfinite(sal)

    def test_compute_spectral_arc_length_jerky(self):
        """Test SAL comparison between smooth and jerky signals."""
        t = np.linspace(0, 1, 100)
        smooth = np.exp(-((t - 0.5) ** 2) / 0.01)

        # Set seed for reproducibility
        rng = np.random.default_rng(42)
        jerky = smooth + 0.1 * rng.standard_normal(len(t))

        sal_smooth = compute_spectral_arc_length(smooth, fs=100.0)
        sal_jerky = compute_spectral_arc_length(jerky, fs=100.0)

        assert sal_smooth != sal_jerky
        # Smooth movement should have higher SAL (closer to 0) than jerky (more negative)
        assert sal_smooth > sal_jerky

    def test_compute_spectral_arc_length_empty(self):
        """Test SAL with empty input."""
        sal = compute_spectral_arc_length(np.array([]), fs=100.0)
        assert sal == 0.0

    def test_compute_spectral_arc_length_zeros(self):
        """Test SAL with zero signal."""
        sal = compute_spectral_arc_length(np.zeros(100), fs=100.0)
        assert sal == 0.0

    def test_compute_cwt(self):
        """Test CWT computation."""
        # Just check shape and finiteness
        freqs, times, cwt = compute_cwt(self.signal, self.fs, num_freqs=10)
        assert len(freqs) == 10
        assert len(times) == len(self.signal)
        assert cwt.shape == (10, len(self.signal))
        assert np.all(np.isfinite(cwt))

    def test_compute_xwt(self):
        """Test XWT computation."""
        freqs, times, xwt = compute_xwt(self.signal, self.signal, self.fs, num_freqs=10)
        assert len(freqs) == 10
        assert xwt.shape == (10, len(self.signal))
        # Self-XWT should have real positive diagonal if it was correlation matrix but here it is elementwise product of CWTs
        # XWT = W1 * conj(W2). If W1=W2, then W1*conj(W1) = |W1|^2 which is real and non-negative
        assert np.all(
            np.abs(np.imag(xwt)) < 1e-10
        )  # imaginary part should be close to 0
        assert np.all(np.real(xwt) >= -1e-10)

    def test_compute_jerk(self):
        """Test jerk computation."""
        # Jerk of sin(t) is -cos(t)
        t = np.linspace(0, 2 * np.pi, 1000)
        acc = np.sin(t)
        jerk = compute_jerk(acc, fs=1000 / (2 * np.pi))

        # Check shape
        assert len(jerk) == len(acc)

        # Too short signal
        short_jerk = compute_jerk(np.array([1, 2, 3]), fs=1.0)
        assert len(short_jerk) == 3

    def test_compute_time_shift(self):
        """Test time shift calculation."""
        # Create a simpler, clean signal for time shift testing
        # A simple Gaussian pulse
        t = np.linspace(-1, 1, 2000)
        x = np.exp(-50 * t**2)

        # y is x delayed by 0.1s
        shift = 0.1
        # t -> t - shift
        y = np.exp(-50 * (t - shift) ** 2)

        fs = 1000.0  # Approximately 2000 samples / 2 seconds

        # Actual fs calculation based on t
        fs = 1.0 / (t[1] - t[0])

        calc_shift = compute_time_shift(x, y, fs)
        assert np.isclose(calc_shift, shift, atol=0.01)

    def test_compute_dtw_distance(self):
        """Test DTW distance."""
        s1 = np.array([0, 1, 2, 3, 2, 1, 0])
        s2 = np.array([0, 0, 1, 2, 3, 2, 1, 0])  # s1 shifted + stutters

        dist = compute_dtw_distance(s1, s2)
        assert dist >= 0
        assert dist < np.linalg.norm(
            s1 - s2[: len(s1)]
        )  # DTW should be smaller than Euclidean

        # Test with Numba disabled (mocking it if necessary, but tricky since it imports at top level)
        # We can test logic by trusting it runs whatever version is available.
        pass

    def test_compute_dtw_path(self):
        """Test DTW path."""
        s1 = np.array([0, 1])
        s2 = np.array([0, 1])
        dist, path = compute_dtw_path(s1, s2)
        assert dist == 0.0
        assert path == [(0, 0), (1, 1)]

    def test_kalman_filter(self):
        """Test Kalman Filter basic operation."""
        kf = KalmanFilter(dim_x=2, dim_z=1)
        # Constant velocity model
        kf.F = np.array([[1, 1], [0, 1]])
        kf.H = np.array([[1, 0]])

        # Initial state
        kf.x = np.array([0.0, 1.0])  # pos=0, vel=1

        kf.predict()
        assert kf.x[0] == 1.0
        assert kf.x[1] == 1.0

        # Measurement update
        kf.update(np.array([1.1]))  # Measured pos is 1.1
        # State should move towards measurement
        assert kf.x[0] > 1.0
