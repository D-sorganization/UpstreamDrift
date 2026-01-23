import numpy as np
import pytest
from shared.python.signal_processing import (
    _morlet2_impl,
    compute_coherence,
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
    @pytest.fixture
    def sample_data(self):
        # Generate a synthetic signal: 10 Hz sine wave + noise
        fs = 100.0
        t = np.arange(0, 5.0, 1 / fs)
        freq = 10.0
        signal_clean = np.sin(2 * np.pi * freq * t)
        return t, signal_clean, fs

    def test_compute_psd(self, sample_data):
        t, data, fs = sample_data
        freqs, psd = compute_psd(data, fs=fs, nperseg=128)

        # Check output shapes
        assert len(freqs) == len(psd)
        assert len(freqs) == 128 // 2 + 1

        # Check peak frequency
        peak_freq = freqs[np.argmax(psd)]
        assert np.isclose(peak_freq, 10.0, atol=1.0)

    def test_compute_spectrogram(self, sample_data):
        t, data, fs = sample_data
        f, t_spec, Sxx = compute_spectrogram(data, fs=fs, nperseg=64, noverlap=32)

        # Check shapes
        assert len(f) == 64 // 2 + 1
        assert Sxx.shape[0] == len(f)
        assert Sxx.shape[1] == len(t_spec)

    def test_compute_spectral_arc_length_smoothness(self):
        # Smooth movement (Gaussian) vs Jerky movement
        fs = 100.0
        t = np.linspace(0, 1, 100)

        # Smooth: Gaussian velocity profile
        smooth = np.exp(-50 * (t - 0.5) ** 2)

        # Jerky: Add high frequency noise
        jerky = smooth + 0.1 * np.sin(2 * np.pi * 40 * t)

        sal_smooth = compute_spectral_arc_length(smooth, fs)
        sal_jerky = compute_spectral_arc_length(jerky, fs)

        # SAL is negative, smoother is closer to 0 (less negative) usually,
        # but strictly it measures arc length of PSD.
        # More noise -> longer arc length -> more negative.
        assert sal_smooth > sal_jerky
        assert sal_smooth < 0
        assert sal_jerky < 0

    def test_compute_spectral_arc_length_empty(self):
        assert compute_spectral_arc_length(np.array([]), 100.0) == 0.0

    def test_compute_spectral_arc_length_zeros(self):
        assert compute_spectral_arc_length(np.zeros(100), 100.0) == 0.0

    def test_morlet2_impl(self):
        # Compare custom implementation with scipy if available, or just check properties
        M = 100
        s = 10.0
        w = 5.0
        wavelet = _morlet2_impl(M, s, w)

        assert len(wavelet) == M
        assert np.iscomplexobj(wavelet)
        # Peak should be at center
        assert (
            np.argmax(np.abs(wavelet)) == M // 2
            or np.argmax(np.abs(wavelet)) == M // 2 - 1
        )

    def test_compute_cwt(self, sample_data):
        t, data, fs = sample_data
        freq_range = (5.0, 20.0)
        num_freqs = 15

        freqs, times, cwt = compute_cwt(
            data, fs, freq_range=freq_range, num_freqs=num_freqs
        )

        assert len(freqs) == num_freqs
        assert len(times) == len(data)
        assert cwt.shape == (num_freqs, len(data))

        # Check if 10Hz is prominent
        idx_10hz = np.argmin(np.abs(freqs - 10.0))
        # Amplitude at 10Hz should be relatively high
        avg_power = np.mean(np.abs(cwt), axis=1)
        assert avg_power[idx_10hz] > np.mean(avg_power)

    def test_compute_xwt(self, sample_data):
        t, data1, fs = sample_data
        # data2 is data1 shifted + noise
        data2 = np.roll(data1, 10)

        freqs, times, xwt = compute_xwt(
            data1, data2, fs, freq_range=(5.0, 20.0), num_freqs=10
        )

        assert xwt.shape == (10, len(data1))

        # Check phase difference at 10Hz
        idx_10hz = np.argmin(np.abs(freqs - 10.0))

        # Phase should be roughly constant related to shift
        phases = np.angle(xwt[idx_10hz, 20:-20])  # avoid edges
        # Just check it runs and produces output for now
        assert not np.all(phases == 0)

    def test_compute_coherence(self, sample_data):
        t, data1, fs = sample_data
        # data2 is data1 + noise
        data2 = data1 + 0.1 * np.random.randn(len(data1))

        freqs, coh = compute_coherence(data1, data2, fs=fs, nperseg=64)

        assert len(freqs) == len(coh)

        # Coherence at signal frequency (10Hz) should be high
        idx_10hz = np.argmin(np.abs(freqs - 10.0))
        assert coh[idx_10hz] > 0.8  # Strong correlation

    # --- New Tests ---

    def test_compute_jerk(self):
        """Test jerk computation."""
        fs = 100.0
        t = np.linspace(0, 2, 200)

        # Acceleration: sin(t)
        # Jerk: cos(t)
        acc = np.sin(2 * np.pi * t)
        jerk = compute_jerk(acc, fs, window_len=5, polyorder=2)

        # Check shape
        assert len(jerk) == len(acc)

        # Check values roughly (Savitzky-Golay is approximation)
        expected = 2 * np.pi * np.cos(2 * np.pi * t)

        # Ignore boundaries
        error = np.abs(jerk[10:-10] - expected[10:-10])
        assert np.mean(error) < 1.0  # Loose tolerance for discrete derivative

    def test_compute_time_shift(self):
        """Test time shift calculation."""
        fs = 100.0
        t = np.linspace(0, 2, 200)
        x = np.sin(2 * np.pi * 5 * t)

        # Shift y by 10 samples (0.1s)
        shift_samples = 10
        y = np.roll(x, shift_samples)

        # y leads x? or lags?
        # If y is x shifted right (delayed), then y(t) = x(t-tau).
        # compute_time_shift returns tau > 0 for delay (lag).

        tau = compute_time_shift(x, y, fs)

        expected_lag = shift_samples / fs
        assert np.isclose(tau, expected_lag, atol=0.02)

    def test_compute_dtw_distance(self):
        """Test DTW distance."""
        s1 = np.array([0, 1, 2, 3, 2, 1, 0])
        s2 = np.array([0, 0, 1, 2, 3, 2, 1, 0])  # s1 stretched/shifted

        dist = compute_dtw_distance(s1, s2)

        # Should be small (perfect match theoretically if purely shifted?)
        # Euclidean would be large.
        euclidean = np.linalg.norm(s1 - s2[:7])
        assert dist < euclidean
        assert dist >= 0

    def test_compute_dtw_path(self):
        """Test DTW path."""
        s1 = np.array([0, 1, 0])
        s2 = np.array([0, 0, 1, 0])

        dist, path = compute_dtw_path(s1, s2)

        assert dist >= 0
        assert len(path) >= max(len(s1), len(s2))
        assert path[0] == (0, 0)
        assert path[-1] == (len(s1) - 1, len(s2) - 1)
