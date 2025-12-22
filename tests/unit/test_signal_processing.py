"""Unit tests for signal processing utilities."""

import numpy as np

from shared.python import signal_processing


def test_compute_psd():
    """Test Power Spectral Density computation."""
    fs = 1000.0
    t = np.arange(0, 1.0, 1 / fs)
    # 50 Hz sine wave
    data = np.sin(2 * np.pi * 50 * t)

    freqs, psd = signal_processing.compute_psd(data, fs, nperseg=256)

    # Peak should be around 50 Hz
    peak_idx = np.argmax(psd)
    peak_freq = freqs[peak_idx]

    # Frequency resolution is fs / nperseg ~= 4 Hz
    assert abs(peak_freq - 50.0) < 5.0


def test_compute_spectrogram():
    """Test Spectrogram computation."""
    fs = 1000.0
    t = np.arange(0, 1.0, 1 / fs)
    # Chirp signal: 20Hz to 100Hz
    data = np.sin(2 * np.pi * 20 * t + 2 * np.pi * 40 * t**2)

    f, t_spec, Sxx = signal_processing.compute_spectrogram(data, fs, nperseg=128)

    assert len(f) > 0
    assert len(t_spec) > 0
    assert Sxx.shape == (len(f), len(t_spec))


def test_compute_spectral_arc_length():
    """Test Spectral Arc Length (smoothness metric)."""
    fs = 100.0
    t = np.arange(0, 1.0, 1 / fs)

    # Smooth movement (Gaussian-like)
    smooth = np.exp(-((t - 0.5) ** 2) / 0.01)

    # Jerky movement (add noise)
    # Set seed for reproducibility
    np.random.seed(42)
    jerky = smooth + 0.1 * np.random.randn(len(t))

    sal_smooth = signal_processing.compute_spectral_arc_length(smooth, fs)
    sal_jerky = signal_processing.compute_spectral_arc_length(jerky, fs)

    # Smooth movement should have higher SAL (closer to 0) than jerky (more negative)
    # Because SAL is negative, closer to 0 is smoother.
    assert sal_smooth > sal_jerky


def test_compute_spectral_arc_length_empty():
    """Test SAL with empty data."""
    assert signal_processing.compute_spectral_arc_length(np.array([]), 100.0) == 0.0


def test_compute_spectral_arc_length_zero():
    """Test SAL with zero data."""
    assert signal_processing.compute_spectral_arc_length(np.zeros(100), 100.0) == 0.0
