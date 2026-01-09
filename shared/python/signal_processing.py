"""Signal processing utilities for biomechanical data analysis.

This module provides common signal processing functions used across
different physics engines for vibration analysis, frequency domain
analysis, and signal quality assessment.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def compute_psd(
    data: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density using Welch's method.

    Args:
        data: Input time series data
        fs: Sampling frequency in Hz
        window: Window function to use (default: 'hann')
        nperseg: Length of each segment (default: None -> 256)

    Returns:
        tuple: (frequencies, psd_values)
    """
    freqs, psd = signal.welch(data, fs=fs, window=window, nperseg=nperseg)
    return freqs, psd


def compute_spectrogram(
    data: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Spectrogram.

    Args:
        data: Input time series data
        fs: Sampling frequency in Hz
        window: Window function to use
        nperseg: Length of each segment
        noverlap: Number of points to overlap between segments

    Returns:
        tuple: (frequencies, times, Sxx)
    """
    f, t, Sxx = signal.spectrogram(
        data,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    return f, t, Sxx


def compute_spectral_arc_length(
    data: np.ndarray,
    fs: float,
    pad_level: int = 4,
    fc: float = 20.0,
    amp_th: float = 0.05,
) -> float:
    """Compute Spectral Arc Length (SAL) smoothness metric.

    A lower SAL value indicates a smoother movement.
    Based on Balasubramanian et al. (2015).

    Args:
        data: Velocity profile
        fs: Sampling frequency
        pad_level: Zero padding level (power of 2)
        fc: Cut-off frequency for normalization
        amp_th: Amplitude threshold (fraction of peak)

    Returns:
        float: SAL value (negative dimensionless metric)
    """
    # Number of points
    n = len(data)
    if n == 0:
        return 0.0

    # Zero padding
    n_padded = int(pow(2, np.ceil(np.log2(n)) + pad_level))

    # FFT
    spectrum = np.fft.fft(data, n_padded)
    spectrum_mag = np.abs(spectrum)

    max_mag = np.max(spectrum_mag)
    if max_mag == 0:
        return 0.0

    # Normalize magnitude
    spectrum_norm = spectrum_mag / max_mag

    # Frequency axis
    freqs = np.fft.fftfreq(n_padded, 1 / fs)

    # Select frequencies up to fc
    mask = (freqs >= 0) & (freqs <= fc)
    freqs_sel = freqs[mask]
    spectrum_sel = spectrum_norm[mask]

    # Select magnitudes above threshold
    # Note: The original paper defines the support region based on amplitude threshold
    # We apply it to filter out noise
    valid_indices = spectrum_sel >= amp_th
    if not np.any(valid_indices):
        return 0.0

    # Calculate arc length
    # Scale frequency to [0, 1] for the integral
    freq_norm = freqs_sel / fc

    # Calculate gradient
    # Optimization: Manual slicing is faster than np.diff
    d_mag = spectrum_sel[1:] - spectrum_sel[:-1]
    d_freq = freq_norm[1:] - freq_norm[:-1]

    # Arc length
    sal = -np.sum(np.sqrt(d_freq**2 + d_mag**2))

    return float(sal)
