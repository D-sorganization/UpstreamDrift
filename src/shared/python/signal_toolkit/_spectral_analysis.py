from __future__ import annotations

import numpy as np
from scipy.signal import (
    coherence,
    spectrogram,
    welch,
)

from src.shared.python.core.contracts import ensure, precondition, require


@precondition(
    lambda data, fs, window="hann", nperseg=None: fs > 0,
    "Sampling frequency must be positive",
)
@precondition(
    lambda data, fs, window="hann", nperseg=None: len(data) > 0,
    "Input data must be non-empty",
)
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
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")
    freqs, psd = welch(data, fs=fs, window=window, nperseg=nperseg)
    return freqs, psd


def compute_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Magnitude Squared Coherence.

    Design by Contract:
        Preconditions:
            - fs > 0 (sampling frequency must be positive)
            - x and y must be non-empty
            - x and y must have the same length

    Args:
        x: First time series
        y: Second time series
        fs: Sampling frequency in Hz
        window: Window function to use (default: 'hann')
        nperseg: Length of each segment (default: None -> 256)

    Returns:
        tuple: (frequencies, coherence_values)
    """
    if not (x is not None):
        raise ValueError("x must be provided")
    if not (x is not None):
        raise ValueError("x must be provided")
    require(fs > 0, "Sampling frequency must be positive", fs)
    require(len(x) > 0, "Input x must be non-empty")
    require(len(y) > 0, "Input y must be non-empty")
    require(
        len(x) == len(y),
        "x and y must have the same length",
        {"x_len": len(x), "y_len": len(y)},
    )
    freqs, coh = coherence(x, y, fs=fs, window=window, nperseg=nperseg)
    return freqs, coh


@precondition(
    lambda data, fs, window="hann", nperseg=256, noverlap=None: fs > 0,
    "Sampling frequency must be positive",
)
@precondition(
    lambda data, fs, window="hann", nperseg=256, noverlap=None: nperseg > 0,
    "Segment length must be positive",
)
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
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")
    f, t, Sxx = spectrogram(
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
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")

    # Number of points
    n = len(data)
    if n == 0:
        return 0.0

    # Zero padding
    n_padded = int(pow(2, np.ceil(np.log2(n)) + pad_level))

    # FFT
    # PERFORMANCE: Use rfft (real input FFT) to avoid computing negative frequencies
    # This reduces computation by ~50% and memory usage by ~50%
    spectrum = np.fft.rfft(data, n_padded)
    spectrum_mag = np.abs(spectrum)

    max_mag = np.max(spectrum_mag)
    if max_mag == 0:
        return 0.0

    # Normalize magnitude
    spectrum_norm = spectrum_mag / max_mag

    # Frequency axis optimization:
    # Instead of generating full fftfreq and masking (which creates large temporary arrays),
    # we calculate the index limit directly.
    # rfft returns positive frequencies at indices 0 to n_padded//2 + 1.
    # df = fs / n_padded
    df = fs / n_padded
    limit_idx = int(np.floor(fc / df)) + 1 if df > 0 else 1

    # limit_idx must be at most n_padded // 2 + 1 (Nyquist limit for positive freqs)
    # This matches the size of rfft output exactly.
    limit_idx = min(limit_idx, len(spectrum_mag))

    # We only need the positive part of spectrum up to fc
    spectrum_sel = spectrum_norm[:limit_idx]

    # Select magnitudes above threshold
    # Note: The original paper defines the support region based on amplitude threshold
    # We apply it to filter out noise
    if not np.any(spectrum_sel >= amp_th):
        return 0.0

    # Calculate gradient
    # Optimization: Manual slicing is faster than np.diff
    d_mag = spectrum_sel[1:] - spectrum_sel[:-1]

    # Optimization: d_freq is constant (df / fc), so we use a scalar instead of an array.
    # This avoids creating two arrays (freq_norm and d_freq) and performing N subtractions.
    # freq_norm = freqs_sel / fc
    # d_freq = freq_norm[1:] - freq_norm[:-1] = df / fc
    d_freq = df / fc

    # Arc length
    sal = -np.sum(np.sqrt(d_freq**2 + d_mag**2))

    result = float(sal)
    ensure(result <= 0, "SAL must be non-positive (arc length is negated)", result)
    return result
