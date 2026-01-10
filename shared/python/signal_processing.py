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


def _morlet2_impl(M: int, s: float, w: float = 5.0) -> np.ndarray:
    """Complex Morlet wavelet implementation.

    Fallback if scipy.signal.morlet2 is unavailable.
    """
    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / s
    output: np.ndarray = np.exp(1j * w * x) * np.exp(-0.5 * x**2) * np.pi ** (-0.25)
    return output


def compute_cwt(
    data: np.ndarray,
    fs: float,
    freq_range: tuple[float, float] = (1.0, 50.0),
    num_freqs: int = 50,
    w0: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Continuous Wavelet Transform using Morlet wavelet.

    Args:
        data: Input time series
        fs: Sampling frequency
        freq_range: (min_freq, max_freq)
        num_freqs: Number of frequency scales
        w0: Omega0 parameter for Morlet wavelet (default 6.0)

    Returns:
        (freqs, times, cwt_matrix)
        freqs: Array of frequencies
        times: Array of time points
        cwt_matrix: Complex CWT coefficients (freqs x time)
    """
    # Create frequency vector (log space usually better for wavelets, but linspace ok)
    # Using logspace for better multiscale analysis
    min_freq, max_freq = freq_range
    freqs = np.geomspace(min_freq, max_freq, num=num_freqs)

    # Convert frequencies to scales
    # For Morlet: scale = w0 * fs / (2 * pi * freq)
    scales = w0 * fs / (2 * np.pi * freqs)

    cwt_matrix = np.zeros((num_freqs, len(data)), dtype=np.complex128)

    # Use internal implementation or scipy's
    if hasattr(signal, "morlet2"):
        morlet_func = signal.morlet2
    else:
        morlet_func = _morlet2_impl

    for i, s in enumerate(scales):
        # Window length for wavelet: typically 6-10 sigmas.
        # Morlet2 std dev is s.
        # Support is roughly [-5s, 5s] or so.
        M = int(2 * 5 * s + 1)  # Sufficient width

        wavelet = morlet_func(M, s, w=w0)

        # Convolve (using 'same' mode to keep length)
        cwt_matrix[i, :] = signal.fftconvolve(data, wavelet, mode="same")

        # Normalize by 1/sqrt(s)
        cwt_matrix[i, :] /= np.sqrt(s)

    times = np.arange(len(data)) / fs

    return freqs, times, cwt_matrix


def compute_xwt(
    data1: np.ndarray,
    data2: np.ndarray,
    fs: float,
    freq_range: tuple[float, float] = (1.0, 50.0),
    num_freqs: int = 50,
    w0: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Cross Wavelet Transform.

    XWT = W1 * conj(W2)

    Args:
        data1: First time series
        data2: Second time series
        fs: Sampling frequency
        freq_range: (min_freq, max_freq)
        num_freqs: Number of frequency scales
        w0: Omega0 parameter

    Returns:
        (freqs, times, xwt_matrix)
        xwt_matrix is complex. Magnitude is cross-power, Angle is relative phase.
    """
    f1, t1, w1 = compute_cwt(data1, fs, freq_range, num_freqs, w0)
    f2, t2, w2 = compute_cwt(data2, fs, freq_range, num_freqs, w0)

    # Ensure dimensions match
    min_len = min(w1.shape[1], w2.shape[1])
    w1 = w1[:, :min_len]
    w2 = w2[:, :min_len]
    times = t1[:min_len]

    xwt_matrix = w1 * np.conj(w2)

    return f1, times, xwt_matrix
