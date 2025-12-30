"""Unit tests for shared signal processing."""

import numpy as np
import pytest
from unittest.mock import patch

from shared.python.signal_processing import (
    compute_psd,
    compute_spectral_arc_length,
    compute_spectrogram,
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

        # Check that we have peaks at roughly 10 Hz and 50 Hz
        # Finding indices of peaks requires scipy.signal which is what we are testing the wrapper for,
        # but let's use a simple argmax logic or manual search around expected bins.

        # Resolution = fs/nperseg = 1000/256 approx 3.9 Hz
        # 10 Hz should be around index 10 / 3.9 ~= 2.5 -> index 2 or 3
        # 50 Hz should be around index 50 / 3.9 ~= 12.8 -> index 12 or 13

        # We can also just mock the underlying scipy call if we trust scipy,
        # but verifying the wrapper actually produces expected math is valuable.

        # Let's verify peak locations roughly
        peak_indices = np.where(psd > np.max(psd) * 0.1)[0]
        peak_freqs = freqs[peak_indices]

        has_10 = np.any(np.abs(peak_freqs - 10) < 5)
        has_50 = np.any(np.abs(peak_freqs - 50) < 5)

        assert has_10, f"Should have peak near 10 Hz. Found: {peak_freqs}"
        assert has_50, f"Should have peak near 50 Hz. Found: {peak_freqs}"

    def test_compute_psd_mock(self):
        """Test PSD computation using mock to verify call arguments."""
        with patch("scipy.signal.welch") as mock_welch:
            mock_welch.return_value = (np.arange(129), np.random.rand(129))
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
        with patch("scipy.signal.spectrogram") as mock_spec:
            mock_spec.return_value = (np.arange(10), np.arange(10), np.random.rand(10, 10))
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
