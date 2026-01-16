"""Signal processing utilities for biomechanical data analysis.

This module provides common signal processing functions used across
different physics engines for vibration analysis, frequency domain
analysis, and signal quality assessment.

Performance optimizations:
- Numba JIT compilation for DTW and other tight loops
- LRU caching for wavelet generation in CWT
- Parallelization hooks for lag matrix computation
"""

from __future__ import annotations

import functools
import logging
from typing import cast

import numpy as np
from scipy import fft, signal
from scipy.signal import (
    coherence,
    correlate,
    correlation_lags,
    savgol_filter,
    spectrogram,
    welch,
)

# Performance: Optional Numba JIT compilation
try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Create a no-op decorator when numba is not available
    def jit(*args: object, **kwargs: object) -> object:  # type: ignore[misc]
        """No-op decorator when numba is not installed."""

        def decorator(func: object) -> object:  # type: ignore[misc]
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


_LOGGER = logging.getLogger(__name__)


# =============================================================================
# PERFORMANCE: Numba-optimized DTW kernel
# =============================================================================


@jit(nopython=True, cache=True, fastmath=True)
def _dtw_core(series1: np.ndarray, series2: np.ndarray, window: int) -> float:
    """Numba-optimized DTW distance computation core.

    This inner kernel runs ~100x faster than pure Python due to JIT compilation.

    Args:
        series1: First sequence (1D float64 array)
        series2: Second sequence (1D float64 array)
        window: Sakoe-Chiba band width

    Returns:
        DTW distance (float)
    """
    n = len(series1)
    m = len(series2)

    # Use large float instead of inf for numba compatibility
    INF = 1e30

    # Allocate cost matrix
    dtw_matrix = np.full((n + 1, m + 1), INF, dtype=np.float64)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        # Sakoe-Chiba band limits
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)

        for j in range(j_start, j_end):
            cost = (series1[i - 1] - series2[j - 1]) ** 2

            # Take minimum of three options
            min_prev = dtw_matrix[i - 1, j]  # Insertion
            if dtw_matrix[i, j - 1] < min_prev:
                min_prev = dtw_matrix[i, j - 1]  # Deletion
            if dtw_matrix[i - 1, j - 1] < min_prev:
                min_prev = dtw_matrix[i - 1, j - 1]  # Match

            dtw_matrix[i, j] = cost + min_prev

    return float(np.sqrt(dtw_matrix[n, m]))


@jit(nopython=True, cache=True, fastmath=True)
def _dtw_path_core(
    series1: np.ndarray, series2: np.ndarray, window: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Numba-optimized DTW path computation core.

    Args:
        series1: First sequence (1D float64 array)
        series2: Second sequence (1D float64 array)
        window: Sakoe-Chiba band width (-1 for none)

    Returns:
        tuple: (distance, path_i, path_j)
        path_i, path_j are arrays of indices (reversed order)
    """
    n = len(series1)
    m = len(series2)

    # Use large float instead of inf for numba compatibility
    INF = 1e30

    # Allocate cost matrix
    dtw_matrix = np.full((n + 1, m + 1), INF, dtype=np.float64)
    dtw_matrix[0, 0] = 0.0

    w = window if window >= 0 else max(n, m)

    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m + 1, i + w + 1)

        for j in range(j_start, j_end):
            cost = (series1[i - 1] - series2[j - 1]) ** 2

            # Find minimum of previous cells
            min_prev = dtw_matrix[i - 1, j]  # Insertion

            val_del = dtw_matrix[i, j - 1]  # Deletion
            if val_del < min_prev:
                min_prev = val_del

            val_match = dtw_matrix[i - 1, j - 1]  # Match
            if val_match < min_prev:
                min_prev = val_match

            dtw_matrix[i, j] = cost + min_prev

    distance = float(np.sqrt(dtw_matrix[n, m]))

    # Backtrack
    # Max path length is n + m
    max_len = n + m
    path_i = np.empty(max_len, dtype=np.int32)
    path_j = np.empty(max_len, dtype=np.int32)

    idx = 0
    i, j = n, m
    while i > 0 and j > 0:
        path_i[idx] = i - 1
        path_j[idx] = j - 1
        idx += 1

        v_ins = dtw_matrix[i - 1, j]
        v_del = dtw_matrix[i, j - 1]
        v_match = dtw_matrix[i - 1, j - 1]

        # Preference order for backtracking: Match, then Insertion, then Deletion
        min_val = v_match
        if v_ins < min_val:
            min_val = v_ins
        if v_del < min_val:
            min_val = v_del

        if min_val == v_match:
            i -= 1
            j -= 1
        elif min_val == v_ins:
            i -= 1
        else:
            j -= 1

    return distance, path_i[:idx], path_j[:idx]


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

    Args:
        x: First time series
        y: Second time series
        fs: Sampling frequency in Hz
        window: Window function to use (default: 'hann')
        nperseg: Length of each segment (default: None -> 256)

    Returns:
        tuple: (frequencies, coherence_values)
    """
    freqs, coh = coherence(x, y, fs=fs, window=window, nperseg=nperseg)
    return freqs, coh


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
    if df > 0:
        limit_idx = int(np.floor(fc / df)) + 1
    else:
        limit_idx = 1

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

    return float(sal)


@functools.lru_cache(maxsize=128)
def _morlet2_impl(M: int, s: float, w: float = 5.0) -> np.ndarray:
    """Complex Morlet wavelet implementation with caching.

    Fallback if scipy.signal.morlet2 is unavailable.

    PERFORMANCE FIX: Added LRU cache to avoid recomputing wavelets.
    """
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
    s = s_int / 1000.0
    w0 = w0_int / 100.0

    # Use scipy's morlet2 if available, else our implementation
    if hasattr(signal, "morlet2"):
        wavelet = signal.morlet2(M, s, w=w0)
    else:
        wavelet = _morlet2_impl(M, s, w=w0)

    # Return the FFT
    return np.asarray(fft.fft(wavelet, n=n_fft))


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

    n_data = len(data)
    cwt_matrix = np.zeros((num_freqs, n_data), dtype=np.complex128)

    # PERFORMANCE: Wavelet generation is now handled by _get_cached_wavelet()
    # which uses LRU cache to avoid recomputation

    # Optimization: Pre-compute FFT of data once to avoid recomputation in fftconvolve
    # 1. Determine maximum wavelet width (corresponds to smallest frequency / largest scale)
    min_f = np.min(freqs)
    max_s = w0 * fs / (2 * np.pi * min_f)
    max_M = int(2 * 5 * max_s + 1)

    # 2. Determine optimal FFT size for the largest convolution
    # Padding to N + M - 1 ensures linear convolution avoids circular aliasing
    target_len = n_data + max_M - 1
    n_fft = fft.next_fast_len(target_len)

    # 3. Compute FFT of data (must use full fft as we will multiply with complex wavelet)
    data_fft = fft.fft(data, n=n_fft)

    for i, s in enumerate(scales):
        # Window length for wavelet: typically 6-10 sigmas.
        # Morlet2 std dev is s.
        # Support is roughly [-5s, 5s] or so.
        M = int(2 * 5 * s + 1)  # Sufficient width

        # PERFORMANCE: Use cached wavelet FFT to avoid recomputation
        # Convert scale and w0 to integers for hashable cache key
        s_int = int(round(s * 1000))
        w0_int = int(round(w0 * 100))
        wavelet_fft = _get_cached_wavelet(M, s_int, w0_int, n_fft)

        # b. Multiply in frequency domain
        prod = data_fft * wavelet_fft

        # c. Inverse FFT
        conv_res = fft.ifft(prod, n=n_fft)

        # d. Center crop to match 'same' mode
        # The linear convolution (length N+M-1) starts at index 0 because inputs were zero-padded at end.
        # mode='same' requires extracting the center section of length N from the full convolution.
        # The center of the full convolution is at index (N+M-1-1)/2.
        # We want the slice [start, start + N].
        # start = (N+M-1)//2 - (N-1)//2 (integer math)
        # Simplified: start = (M-1) // 2
        start_idx = (M - 1) // 2

        # Ensure we stay within bounds (though with correct padding, it should be fine)
        if start_idx + n_data <= len(conv_res):
            cwt_matrix[i, :] = conv_res[start_idx : start_idx + n_data]
        else:
            # This branch is expected to be unreachable if the padding/FFT sizing logic is correct.
            # Raise an error rather than silently changing the alignment, so any upstream issue
            # with convolution length or padding is detected during development.
            raise RuntimeError(
                "Unexpected CWT convolution length: "
                f"start_idx + n_data = {start_idx + n_data}, len(conv_res) = {len(conv_res)}. "
                "Check wavelet padding and FFT size logic."
            )

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


def compute_jerk(
    data: np.ndarray,
    fs: float,
    window_len: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Compute jerk (derivative of acceleration) from position, velocity or acceleration.

    If input is position, 3rd derivative.
    If input is velocity, 2nd derivative.
    If input is acceleration, 1st derivative.

    This function assumes input is ACCELERATION. If you have position/velocity,
    differentiate them first or chain this function.

    Uses Savitzky-Golay filter for smooth differentiation.

    Args:
        data: Acceleration time series
        fs: Sampling frequency
        window_len: Window length for Savitzky-Golay (odd integer)
        polyorder: Polynomial order

    Returns:
        Jerk time series (same length as input)
    """
    if len(data) < window_len:
        # Fallback to simple finite difference if too short
        dt = 1.0 / fs
        return cast("np.ndarray", np.gradient(data, dt))

    if window_len % 2 == 0:
        window_len += 1

    # deriv=1 means first derivative
    # delta = 1.0/fs (spacing)
    jerk = savgol_filter(
        data, window_length=window_len, polyorder=polyorder, deriv=1, delta=1.0 / fs
    )

    return cast("np.ndarray", jerk)


def compute_time_shift(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    max_lag: float | None = None,
) -> float:
    """Compute time shift between two signals using cross-correlation.

    Returns the time lag tau such that y(t) approx x(t - tau).
    Positive tau means y lags x (x leads y).
    Negative tau means y leads x (x lags y).

    Args:
        x: Reference signal
        y: Comparison signal
        fs: Sampling frequency
        max_lag: Maximum lag to search in seconds (default: None = full)

    Returns:
        Time shift in seconds.
    """
    if len(x) != len(y):
        # Truncate to min length
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]

    # Detrend/Normalize
    x = x - np.mean(x)
    y = y - np.mean(y)

    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0

    corr = correlate(x, y, mode="full")
    lags = correlation_lags(len(x), len(y), mode="full")

    if max_lag is not None:
        lag_samples = int(max_lag * fs)
        mask = (lags >= -lag_samples) & (lags <= lag_samples)
        corr = corr[mask]
        lags = lags[mask]

    if len(corr) == 0:
        return 0.0

    # We want to find the shift that maximizes positive correlation (alignment)
    # Using abs() can bias towards anti-phase alignment if overlap is larger
    peak_idx = np.argmax(corr)
    lag_sample = lags[peak_idx]

    # In scipy.signal.correlate(x, y), lag k corresponds to sum(x[n] * y[n-k]).
    # If x(t) = s(t) and y(t) = s(t - tau), then y is shifted right by tau (delayed).
    # y[n] \approx x[n - D].
    # The correlation sum is sum_n x[n] * x[n - D - k].
    # This peaks when -D - k = 0 => k = -D.
    # So the lag k returned by correlate is -D.
    # We want to return +D (delay).
    # So we must negate the result.

    return float(-lag_sample / fs)


def compute_dtw_distance(
    series1: np.ndarray,
    series2: np.ndarray,
    window: int | None = None,
) -> float:
    """Compute Dynamic Time Warping (DTW) distance between two sequences.

    Uses Euclidean distance as the local cost measure.
    Implements Sakoe-Chiba band constraint if window is specified.

    PERFORMANCE: Uses Numba JIT-compiled kernel when available (~100x speedup).

    Args:
        series1: First sequence (1D array)
        series2: Second sequence (1D array)
        window: Sakoe-Chiba band width (None for no constraint)

    Returns:
        DTW distance (float)
    """
    n = len(series1)
    m = len(series2)

    # Sakoe-Chiba band constraint
    w = window if window is not None else max(n, m)

    # PERFORMANCE: Use Numba-optimized kernel if available
    if NUMBA_AVAILABLE:
        # Ensure arrays are float64 for numba
        s1 = np.asarray(series1, dtype=np.float64)
        s2 = np.asarray(series2, dtype=np.float64)
        return float(_dtw_core(s1, s2, w))

    # Fallback: Pure Python implementation
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        # Determine band limits
        j_start = max(1, i - w)
        j_end = min(m + 1, i + w + 1)

        for j in range(j_start, j_end):
            cost = (series1[i - 1] - series2[j - 1]) ** 2
            # Take minimum of (match, insertion, deletion)
            last_min = min(
                dtw_matrix[i - 1, j],  # Insertion
                dtw_matrix[i, j - 1],  # Deletion
                dtw_matrix[i - 1, j - 1],  # Match
            )
            dtw_matrix[i, j] = cost + last_min

    return float(np.sqrt(dtw_matrix[n, m]))


def compute_dtw_path(
    series1: np.ndarray,
    series2: np.ndarray,
    window: int | None = None,
) -> tuple[float, list[tuple[int, int]]]:
    """Compute DTW distance and optimal warping path.

    Args:
        series1: First sequence
        series2: Second sequence
        window: Sakoe-Chiba band width

    Returns:
        Tuple (distance, path). Path is list of (i, j) indices.
    """
    # Ensure inputs are float64 arrays for Numba
    s1 = np.asarray(series1, dtype=np.float64)
    s2 = np.asarray(series2, dtype=np.float64)

    w_val = window if window is not None else -1

    # Use Numba kernel (which works as pure python too via no-op jit)
    dist, pi, pj = _dtw_path_core(s1, s2, w_val)

    # Convert structure to list of tuples
    # pi, pj are in reverse order from backtracking
    path = []
    # Loop backwards to reverse
    for k in range(len(pi) - 1, -1, -1):
        path.append((int(pi[k]), int(pj[k])))

    return dist, path


class KalmanFilter:
    """Simple linear Kalman filter implementation.

    Supports n-dimensional state and measurement vectors.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        F: np.ndarray | None = None,
        H: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        P: np.ndarray | None = None,
        x: np.ndarray | None = None,
    ):
        """Initialize Kalman Filter.

        Args:
            dim_x: State dimension
            dim_z: Measurement dimension
            F: State transition matrix (dim_x, dim_x)
            H: Measurement function (dim_z, dim_x)
            Q: Process noise covariance (dim_x, dim_x)
            R: Measurement noise covariance (dim_z, dim_z)
            P: Error covariance matrix (dim_x, dim_x)
            x: Initial state (dim_x,)
        """
        self.dim_x = dim_x
        self.dim_z = dim_z

        # State transition matrix
        self.F = F if F is not None else np.eye(dim_x)

        # Measurement function
        self.H = H if H is not None else np.zeros((dim_z, dim_x))

        # Process noise covariance
        self.Q = Q if Q is not None else np.eye(dim_x)

        # Measurement noise covariance
        self.R = R if R is not None else np.eye(dim_z)

        # Error covariance
        self.P = P if P is not None else np.eye(dim_x)

        # State
        self.x = x if x is not None else np.zeros(dim_x)

    def predict(self) -> None:
        """Predict next state (prior)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        """Update state with measurement (posterior).

        Args:
            z: Measurement vector
        """
        # System uncertainty
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        # Use solve instead of inv for stability
        try:
            K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(self.dim_z))
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if S is singular
            K = self.P @ self.H.T @ np.linalg.pinv(S)

        # Residual
        y = z - self.H @ self.x

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I_mat = np.eye(self.dim_x)
        self.P = (I_mat - K @ self.H) @ self.P
