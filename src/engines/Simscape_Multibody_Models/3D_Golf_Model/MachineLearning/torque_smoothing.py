"""Configurable smoothing for sequential torque rows prior to polynomial fit.

This module addresses the gap identified in #3980: the bridge from frame-search
piecewise-constant torque rows to the smooth sixth-order polynomial inputs
consumed by the Simscape model. It exposes four smoothing methods plus a
diagnostic helper that flags polynomial fits whose maximum residual against
the smoothed torque grid exceeds a configurable threshold.

Design by contract:
    * All smoothing methods preserve array length and dtype (float64).
    * NaN/Inf values are forbidden in inputs and outputs; a ``ValueError`` is
      raised when smoothing would produce non-finite values.
    * ``savitzky_golay`` requires an odd window strictly greater than the
      polyorder; both preconditions raise ``ValueError`` on violation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, filtfilt, savgol_filter

LOGGER = logging.getLogger(__name__)

SmoothingMethod = Literal["moving_average", "savitzky_golay", "lowpass", "spline"]
VALID_METHODS: tuple[SmoothingMethod, ...] = (
    "moving_average",
    "savitzky_golay",
    "lowpass",
    "spline",
)


@dataclass(frozen=True)
class SmoothingConfig:
    """Configuration for torque smoothing.

    Attributes
    ----------
    method:
        One of :data:`VALID_METHODS`.
    window:
        Window size for ``moving_average`` and ``savitzky_golay``. Must be a
        positive integer; must be odd for ``savitzky_golay``.
    polyorder:
        Polynomial order for ``savitzky_golay``. Must be < ``window``.
    cutoff_hz:
        Cutoff frequency in Hz for ``lowpass`` Butterworth filter.
    butter_order:
        Order of the Butterworth filter.
    spline_s:
        Smoothing factor ``s`` for ``UnivariateSpline``. ``None`` lets scipy
        pick a default.
    """

    method: SmoothingMethod = "moving_average"
    window: int = 5
    polyorder: int = 3
    cutoff_hz: float = 25.0
    butter_order: int = 4
    spline_s: float | None = None


def _ensure_finite(values: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must not contain NaN or Inf")


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("moving_average window must be >= 1")
    if window == 1:
        return values.astype(float, copy=True)
    kernel = np.ones(window, dtype=float) / float(window)
    # Pad with edge values so output matches input length.
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    # Convolution output length = len(padded) - window + 1; trim to match.
    return smoothed[: len(values)].astype(float)


def _savitzky_golay(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    if window % 2 == 0:
        raise ValueError("savitzky_golay window must be odd")
    if polyorder >= window:
        raise ValueError("savitzky_golay polyorder must be < window")
    effective_window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if effective_window < polyorder + 1:
        # Series too short to filter; return a copy.
        return values.astype(float, copy=True)
    return savgol_filter(values, effective_window, polyorder).astype(float)


def _lowpass(
    values: np.ndarray, time: np.ndarray, cutoff_hz: float, order: int
) -> np.ndarray:
    if len(time) < 2:
        return values.astype(float, copy=True)
    dt = float(np.median(np.diff(time)))
    if dt <= 0.0:
        raise ValueError("lowpass requires strictly increasing time samples")
    fs = 1.0 / dt
    nyq = 0.5 * fs
    if cutoff_hz <= 0.0 or cutoff_hz >= nyq:
        # Out-of-band cutoff: fall back to identity to avoid filter instability.
        return values.astype(float, copy=True)
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    # filtfilt requires len(signal) > padlen ~ 3*max(len(a), len(b)).
    if len(values) <= 3 * max(len(a), len(b)):
        return values.astype(float, copy=True)
    return filtfilt(b, a, values).astype(float)


def _spline(values: np.ndarray, time: np.ndarray, s: float | None) -> np.ndarray:
    if len(time) < 4:
        return values.astype(float, copy=True)
    spline = UnivariateSpline(time, values, s=s if s is not None else len(values))
    return spline(time).astype(float)


def smooth_torque(
    time: np.ndarray, values: np.ndarray, config: SmoothingConfig
) -> np.ndarray:
    """Apply the configured smoothing method to a single torque column.

    Parameters
    ----------
    time:
        Monotonically increasing time samples (seconds).
    values:
        Torque samples corresponding to ``time``.
    config:
        Smoothing configuration. See :class:`SmoothingConfig`.

    Returns
    -------
    np.ndarray
        Smoothed torque, same length as ``values``, dtype float64.

    Raises
    ------
    ValueError
        If ``config.method`` is unrecognized, if inputs contain NaN/Inf, if
        the smoothing produces non-finite values, or if method-specific
        preconditions (e.g., odd Savitzky-Golay window) are violated.
    """
    if config.method not in VALID_METHODS:
        raise ValueError(
            f"Unknown smoothing method '{config.method}'. "
            f"Expected one of {VALID_METHODS}."
        )
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.shape != values.shape:
        raise ValueError("time and values must have identical shape")
    _ensure_finite(time, "time")
    _ensure_finite(values, "values")

    if config.method == "moving_average":
        smoothed = _moving_average(values, config.window)
    elif config.method == "savitzky_golay":
        smoothed = _savitzky_golay(values, config.window, config.polyorder)
    elif config.method == "lowpass":
        smoothed = _lowpass(values, time, config.cutoff_hz, config.butter_order)
    else:  # spline
        smoothed = _spline(values, time, config.spline_s)

    _ensure_finite(smoothed, f"{config.method}(values)")
    if smoothed.shape != values.shape:
        raise ValueError(
            f"Smoothing method '{config.method}' changed shape "
            f"{values.shape} -> {smoothed.shape}"
        )
    return smoothed


def polynomial_residual_diagnostic(
    time: np.ndarray,
    smoothed: np.ndarray,
    coefficients: np.ndarray,
    threshold: float,
) -> dict[str, float | bool]:
    """Quantify how well the polynomial fits the smoothed torque grid.

    Parameters
    ----------
    time:
        Time samples used to evaluate the polynomial.
    smoothed:
        Smoothed torque grid being approximated.
    coefficients:
        Polynomial coefficients in ``np.polyval`` order (highest degree first).
    threshold:
        Maximum acceptable absolute residual (same units as torque).

    Returns
    -------
    dict
        Keys: ``max_abs_residual``, ``rmse``, ``threshold``, ``exceeds_threshold``.
    """
    fitted = np.polyval(coefficients, time)
    residual = smoothed - fitted
    max_abs = float(np.max(np.abs(residual)))
    # ⚡ Bolt: np.vdot avoids intermediate array allocation compared to np.mean(x**2)
    rmse = float(np.sqrt(np.vdot(residual, residual) / residual.size))
    exceeds = max_abs > threshold
    if exceeds:
        LOGGER.warning(
            "Polynomial residual %.6g exceeds threshold %.6g (rmse=%.6g)",
            max_abs,
            threshold,
            rmse,
        )
    return {
        "max_abs_residual": max_abs,
        "rmse": rmse,
        "threshold": float(threshold),
        "exceeds_threshold": bool(exceeds),
    }
