from __future__ import annotations

from typing import cast

import numpy as np
from scipy.signal import (
    correlate,
    correlation_lags,
    savgol_filter,
)


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
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")

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
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")

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
