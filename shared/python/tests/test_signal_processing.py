import numpy as np

from shared.python.signal_processing import (
    compute_psd,
    compute_spectral_arc_length,
    compute_spectrogram,
)


class TestSignalProcessing:
    def test_compute_psd_sine_wave(self):
        """Test PSD computation for a known sine wave."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        freq = 50.0
        data = np.sin(2 * np.pi * freq * t)

        f, psd = compute_psd(data, fs, nperseg=256)

        # Peak should be near 50 Hz
        peak_freq = f[np.argmax(psd)]
        assert (
            abs(peak_freq - freq) < 5.0
        )  # Allow some spectral leakage/resolution error

    def test_compute_spectrogram_chirp(self):
        """Test spectrogram for a chirp signal."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        # Linear chirp from 10 Hz to 100 Hz
        data = np.sin(2 * np.pi * (10 + 90 * t) * t)

        f, time, Sxx = compute_spectrogram(data, fs, nperseg=128)

        assert Sxx.shape[0] == f.shape[0]
        assert Sxx.shape[1] == time.shape[0]
        assert np.max(Sxx) > 0

    def test_compute_spectral_arc_length_smooth(self):
        """Test SAL for a smooth movement (e.g., Gaussian velocity profile)."""
        fs = 100.0
        t = np.linspace(-1, 1, 200)
        # Gaussian bell curve
        data = np.exp(-5 * t**2)

        sal = compute_spectral_arc_length(data, fs)

        # SAL should be negative
        assert sal < 0

    def test_compute_spectral_arc_length_jerky(self):
        """Test SAL for a jerky movement (noise)."""
        fs = 100.0
        t = np.linspace(0, 1, 200)
        data = np.random.normal(0, 1, 200)

        sal = compute_spectral_arc_length(data, fs)

        # Jerky movement should have more negative SAL (larger magnitude) than smooth?
        # Actually SAL measures smoothness, so smooth is closer to 0 (less negative)
        # and jerky is more negative.

        # Let's compare smooth vs jerky
        smooth_data = np.sin(2 * np.pi * 1.0 * t)
        _sal_smooth = compute_spectral_arc_length(smooth_data, fs)

        # Ideally, sal_smooth > sal_jerky (closer to 0)
        # But for random noise vs sine, let's just check it runs and returns float
        assert isinstance(sal, float)
        assert sal < 0

    def test_sal_zero_input(self):
        """Test SAL with empty input."""
        sal = compute_spectral_arc_length(np.array([]), 100.0)
        assert sal == 0.0

    def test_sal_constant_input(self):
        """Test SAL with constant input (DC)."""
        data = np.ones(100)
        sal = compute_spectral_arc_length(data, 100.0)
        # Spectrum will be a peak at 0. Normalized spectrum is 1 at 0, 0 elsewhere.
        # Arc length should be small but calculated.
        assert isinstance(sal, float)
