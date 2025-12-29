from unittest.mock import patch

import numpy as np

from shared.python import signal_processing


def test_compute_psd():
    """Test Power Spectral Density computation."""
    fs = 1000.0
    t = np.arange(0, 1.0, 1 / fs)
    data = np.sin(2 * np.pi * 50 * t)

    with patch("scipy.signal.welch") as mock_welch:
        mock_welch.return_value = (np.arange(129), np.random.rand(129))
        freqs, psd = signal_processing.compute_psd(data, fs, nperseg=256)

        assert len(freqs) == 129
        assert len(psd) == 129
        assert mock_welch.called


def test_compute_spectrogram():
    """Test Spectrogram computation."""
    fs = 1000.0
    t = np.arange(0, 1.0, 1 / fs)
    data = np.sin(2 * np.pi * 20 * t + 2 * np.pi * 40 * t**2)

    with patch("scipy.signal.spectrogram") as mock_spec:
        mock_spec.return_value = (np.arange(10), np.arange(10), np.random.rand(10, 10))
        f, t_spec, Sxx = signal_processing.compute_spectrogram(data, fs, nperseg=128)

        assert len(f) == 10
        assert len(t_spec) == 10
        assert Sxx.shape == (10, 10)
        assert mock_spec.called


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
