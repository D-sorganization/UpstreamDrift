"""Unit tests for shared signal processing."""

import unittest

import numpy as np
import scipy.signal

from shared.python.signal_processing import (
    compute_psd,
    compute_spectral_arc_length,
    compute_spectrogram,
)


class TestSignalProcessing(unittest.TestCase):
    """Test cases for signal processing utilities."""

    def setUp(self):
        """Set up test data."""
        self.fs = 1000.0  # 1 kHz sampling
        self.t = np.arange(0, 1.0, 1 / self.fs)
        # Create a signal with 10 Hz and 50 Hz components
        self.signal = np.sin(2 * np.pi * 10 * self.t) + 0.5 * np.sin(
            2 * np.pi * 50 * self.t
        )

    def test_compute_psd(self):
        """Test PSD computation."""
        # Use boxcar window for sharper peaks
        freqs, psd = compute_psd(self.signal, self.fs, window="boxcar", nperseg=256)

        # Check output shapes
        self.assertEqual(len(freqs), 129)  # nperseg/2 + 1
        self.assertEqual(len(psd), 129)

        # Check that we have peaks at roughly 10 Hz and 50 Hz
        # Finding indices of peaks
        peaks, _ = scipy.signal.find_peaks(psd, height=0.01)  # Lower threshold
        peak_freqs = freqs[peaks]

        # There should be peaks near 10 and 50
        # Given resolution = fs/nperseg = 1000/256 approx 3.9 Hz
        has_10 = np.any(np.abs(peak_freqs - 10) < 5)
        has_50 = np.any(np.abs(peak_freqs - 50) < 5)

        self.assertTrue(has_10, f"Should have peak near 10 Hz. Found: {peak_freqs}")
        self.assertTrue(has_50, f"Should have peak near 50 Hz. Found: {peak_freqs}")

    def test_compute_spectrogram(self):
        """Test spectrogram computation."""
        f, t, Sxx = compute_spectrogram(self.signal, self.fs, nperseg=256)

        # Check shapes
        self.assertEqual(len(f), 129)
        self.assertTrue(Sxx.shape[0] == 129)
        self.assertTrue(Sxx.shape[1] > 0)

    def test_compute_spectral_arc_length_smooth(self):
        """Test SAL on a smooth signal."""
        # A simple smooth movement (e.g. half sine)
        t = np.linspace(0, 1, 100)
        smooth_signal = np.sin(np.pi * t)

        sal = compute_spectral_arc_length(smooth_signal, fs=100.0)

        # SAL should be negative
        self.assertLess(sal, 0)

        # Should be a finite number
        self.assertTrue(np.isfinite(sal))

    def test_compute_spectral_arc_length_jerky(self):
        """Test SAL on a jerky signal."""
        t = np.linspace(0, 1, 100)
        # Smooth signal + noise
        jerky_signal = np.sin(np.pi * t) + 0.1 * np.random.randn(100)

        smooth_sal = compute_spectral_arc_length(np.sin(np.pi * t), fs=100.0)
        jerky_sal = compute_spectral_arc_length(jerky_signal, fs=100.0)

        self.assertNotEqual(smooth_sal, jerky_sal)

    def test_compute_spectral_arc_length_empty(self):
        """Test SAL with empty input."""
        sal = compute_spectral_arc_length(np.array([]), fs=100.0)
        self.assertEqual(sal, 0.0)

    def test_compute_spectral_arc_length_zeros(self):
        """Test SAL with zero signal."""
        sal = compute_spectral_arc_length(np.zeros(100), fs=100.0)
        self.assertEqual(sal, 0.0)
