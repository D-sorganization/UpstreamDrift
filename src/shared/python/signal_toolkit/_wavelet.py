from __future__ import annotations

import functools

import numpy as np
from scipy import fft, signal

from src.shared.python.core.contracts import precondition


@functools.lru_cache(maxsize=128)
def _morlet2_impl(M: int, s: float, w: float = 5.0) -> np.ndarray:
    """Complex Morlet wavelet implementation with caching.

    Fallback if scipy.signal.morlet2 is unavailable.

    PERFORMANCE FIX: Added LRU cache to avoid recomputing wavelets.
    """
    if not (M is not None):
        raise ValueError("M must be provided")
    if not (M is not None):
        raise ValueError("M must be provided")
    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / s
    output: np.ndarray = np.exp(1j * w * x) * np.exp(-0.5 * x**2) * np.pi ** (-0.25)
    # Convert to tuple for caching (numpy arrays aren't hashable)
    return output


# =============================================================================
# PERFORMANCE: Cached wavelet generation for CWT
# =============================================================================


@functools.lru_cache(maxsize=256)
def _get_cached_wavelet(M: int, s_int: int, w0_int: int, n_fft: int) -> np.ndarray:
    """Generate and cache wavelet FFT for CWT computation.

    PERFORMANCE: Caches wavelet FFTs to avoid recomputation. The cache key uses
    integers (scale * 1000, w0 * 100) to enable hashable lookup while maintaining
    sufficient precision.

    Args:
        M: Wavelet length
        s_int: Scale * 1000 (integer for hashing)
        w0_int: w0 * 100 (integer for hashing)
        n_fft: FFT length

    Returns:
        Wavelet FFT (complex array)
    """
    if not (M is not None):
        raise ValueError("M must be provided")
    if not (M is not None):
        raise ValueError("M must be provided")
    s = s_int / 1000.0
    w0 = w0_int / 100.0

    # Use scipy's morlet2 if available, else our implementation
    if hasattr(signal, "morlet2"):
        wavelet = signal.morlet2(M, s, w=w0)
    else:
        wavelet = _morlet2_impl(M, s, w=w0)

    # Return the FFT
    return np.asarray(fft.fft(wavelet, n=n_fft))


def _validate_cwt_inputs(
    fs: float, freq_range: tuple[float, float]
) -> tuple[float, float]:
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")
    min_freq, max_freq = freq_range
    if min_freq <= 0:
        raise ValueError(f"Minimum frequency must be positive, got {min_freq}")
    return min_freq, max_freq


def _prepare_cwt_fft(
    data: np.ndarray,
    freqs: np.ndarray,
    w0: float,
    fs: float,
) -> tuple[int, int, np.ndarray]:
    if not (data is not None):
        raise ValueError("data must be provided")
    if not (data is not None):
        raise ValueError("data must be provided")
    n_data = len(data)

    # Determine maximum wavelet width (corresponds to smallest frequency / largest scale)
    min_f = np.min(freqs)
    max_s = w0 * fs / (2 * np.pi * min_f)
    max_M = int(2 * 5 * max_s + 1)

    # Determine optimal FFT size for the largest convolution
    # Padding to N + M - 1 ensures linear convolution avoids circular aliasing
    target_len = n_data + max_M - 1
    n_fft = fft.next_fast_len(target_len)

    # Compute FFT of data (must use full fft as we will multiply with complex wavelet)
    data_fft = fft.fft(data, n=n_fft)

    return n_data, n_fft, data_fft


def _convolve_wavelet_at_scale(
    data_fft: np.ndarray,
    n_fft: int,
    n_data: int,
    s: float,
    w0: float,
) -> np.ndarray:
    if not (data_fft is not None):
        raise ValueError("data_fft must be provided")
    if not (data_fft is not None):
        raise ValueError("data_fft must be provided")
    M = int(2 * 5 * s + 1)

    # Use cached wavelet FFT to avoid recomputation
    s_int = int(round(s * 1000))
    w0_int = int(round(w0 * 100))
    wavelet_fft = _get_cached_wavelet(M, s_int, w0_int, n_fft)

    prod = data_fft * wavelet_fft
    conv_res = fft.ifft(prod, n=n_fft)

    # Center crop to match 'same' mode
    # start = (M-1) // 2
    start_idx = (M - 1) // 2

    if start_idx + n_data <= len(conv_res):
        row = conv_res[start_idx : start_idx + n_data]
    else:
        raise RuntimeError(
            "Unexpected CWT convolution length: "
            f"start_idx + n_data = {start_idx + n_data}, len(conv_res) = {len(conv_res)}. "
            "Check wavelet padding and FFT size logic."
        )

    # Normalize by 1/sqrt(s)
    row /= np.sqrt(s)
    return row


@precondition(
    lambda data, fs, freq_range=(1.0, 50.0), num_freqs=50, w0=6.0: fs > 0,
    "Sampling frequency must be positive",
)
@precondition(
    lambda data, fs, freq_range=(1.0, 50.0), num_freqs=50, w0=6.0: num_freqs > 0,
    "Number of frequency scales must be positive",
)
@precondition(
    lambda data, fs, freq_range=(1.0, 50.0), num_freqs=50, w0=6.0: len(data) > 0,
    "Input data must be non-empty",
)
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
    if not (data is not None):
        raise ValueError("data must be provided")
    if not (data is not None):
        raise ValueError("data must be provided")
    _validate_cwt_inputs(fs, freq_range)

    freqs = np.geomspace(freq_range[0], freq_range[1], num=num_freqs)
    scales = w0 * fs / (2 * np.pi * freqs)

    n_data, n_fft, data_fft = _prepare_cwt_fft(data, freqs, w0, fs)
    cwt_matrix = np.zeros((num_freqs, n_data), dtype=np.complex128)

    for i, s in enumerate(scales):
        cwt_matrix[i, :] = _convolve_wavelet_at_scale(data_fft, n_fft, n_data, s, w0)

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
    if not (data1 is not None):
        raise ValueError("data1 must be provided")
    if not (data1 is not None):
        raise ValueError("data1 must be provided")
    f1, t1, w1 = compute_cwt(data1, fs, freq_range, num_freqs, w0)
    f2, t2, w2 = compute_cwt(data2, fs, freq_range, num_freqs, w0)

    # Ensure dimensions match
    min_len = min(w1.shape[1], w2.shape[1])
    w1 = w1[:, :min_len]
    w2 = w2[:, :min_len]
    times = t1[:min_len]

    xwt_matrix = w1 * np.conj(w2)

    return f1, times, xwt_matrix
