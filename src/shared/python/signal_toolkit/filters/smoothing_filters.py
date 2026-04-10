"""Smoothing and basic convolution-based signal filters."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.signal import filtfilt, lfilter, medfilt, savgol_filter

from src.shared.python.core.contracts import require  # type: ignore[import-untyped]

from ..core import Signal
from .filter_design import FilterSpec


def apply_filter(
    signal: Signal,
    filter_spec: FilterSpec,
    zero_phase: bool = True,
) -> Signal:
    """Apply a filter to a signal.

    Args:
        signal: Input signal.
        filter_spec: Filter specification.
        zero_phase: If True, use zero-phase filtering (filtfilt).

    Returns:
        Filtered signal.
    """
    if not (signal is not None):
        raise ValueError("signal must be provided")
    if not (signal is not None):
        raise ValueError("signal must be provided")
    if zero_phase:
        # Zero-phase filtering (no phase distortion)
        filtered_values = filtfilt(filter_spec.b, filter_spec.a, signal.values)
    else:
        # Causal filtering (introduces phase shift)
        filtered_values = lfilter(filter_spec.b, filter_spec.a, signal.values)

    return Signal(
        time=signal.time,
        values=filtered_values,
        name=f"{signal.name}_filtered",
        units=signal.units,
        metadata={
            **signal.metadata,
            "filter_type": filter_spec.filter_type.value,
            "filter_design": filter_spec.design.value,
            "cutoff": filter_spec.cutoff,
        },
    )


def create_moving_average_filter(
    window_size: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a moving average filter function.

    Args:
        window_size: Size of the moving average window.

    Returns:
        Function that applies moving average to values.
    """
    kernel = np.ones(window_size) / window_size

    def apply(values: np.ndarray) -> np.ndarray:
        return np.convolve(values, kernel, mode="same")

    return apply


def create_savgol_filter(
    window_length: int = 11,
    polyorder: int = 3,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a Savitzky-Golay filter function.

    Args:
        window_length: Window length (must be odd).
        polyorder: Polynomial order.

    Returns:
        Function that applies Savitzky-Golay filter to values.
    """
    if not (window_length is not None):
        raise ValueError("window_length must be provided")
    if not (window_length is not None):
        raise ValueError("window_length must be provided")
    if window_length % 2 == 0:
        window_length += 1

    def apply(values: np.ndarray) -> np.ndarray:
        if len(values) < window_length:
            return values
        return savgol_filter(values, window_length, polyorder)

    return apply


def apply_moving_average(
    signal: Signal,
    window_size: int,
) -> Signal:
    """Apply moving average filter to a signal.

    Args:
        signal: Input signal.
        window_size: Size of moving average window.

    Returns:
        Filtered signal.
    """
    if not (signal is not None):
        raise ValueError("signal must be provided")
    if not (signal is not None):
        raise ValueError("signal must be provided")
    filter_func = create_moving_average_filter(window_size)
    filtered_values = filter_func(signal.values)

    return Signal(
        time=signal.time,
        values=filtered_values,
        name=f"{signal.name}_ma{window_size}",
        units=signal.units,
        metadata={**signal.metadata, "filter": "moving_average", "window": window_size},
    )


def apply_savgol(
    signal: Signal,
    window_length: int = 11,
    polyorder: int = 3,
) -> Signal:
    """Apply Savitzky-Golay filter to a signal.

    Args:
        signal: Input signal.
        window_length: Window length (must be odd).
        polyorder: Polynomial order.

    Returns:
        Filtered signal.
    """
    if not (signal is not None):
        raise ValueError("signal must be provided")
    if not (signal is not None):
        raise ValueError("signal must be provided")
    if window_length % 2 == 0:
        window_length += 1

    if len(signal.values) < window_length:
        return signal.copy()

    filtered_values = savgol_filter(signal.values, window_length, polyorder)

    return Signal(
        time=signal.time,
        values=filtered_values,
        name=f"{signal.name}_savgol",
        units=signal.units,
        metadata={
            **signal.metadata,
            "filter": "savgol",
            "window": window_length,
            "order": polyorder,
        },
    )


def apply_median_filter(
    signal: Signal,
    kernel_size: int = 5,
) -> Signal:
    """Apply median filter to a signal.

    Useful for removing impulse noise.

    Args:
        signal: Input signal.
        kernel_size: Size of median filter kernel (must be odd).

    Returns:
        Filtered signal.
    """
    if not (signal is not None):
        raise ValueError("signal must be provided")
    if not (signal is not None):
        raise ValueError("signal must be provided")
    if kernel_size % 2 == 0:
        kernel_size += 1

    filtered_values = medfilt(signal.values, kernel_size)

    return Signal(
        time=signal.time,
        values=filtered_values,
        name=f"{signal.name}_median",
        units=signal.units,
        metadata={**signal.metadata, "filter": "median", "kernel": kernel_size},
    )


def apply_exponential_smoothing(
    signal: Signal,
    alpha: float = 0.3,
) -> Signal:
    """Apply exponential smoothing to a signal.

    Args:
        signal: Input signal.
        alpha: Smoothing factor (0 < alpha <= 1). Higher = less smoothing.

    Returns:
        Smoothed signal.
    """
    if not (signal is not None):
        raise ValueError("signal must be provided")
    if not (signal is not None):
        raise ValueError("signal must be provided")
    require(0.0 < alpha <= 1.0, f"alpha must be in (0, 1], got {alpha}")
    values = signal.values
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]

    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]

    return Signal(
        time=signal.time,
        values=smoothed,
        name=f"{signal.name}_ema",
        units=signal.units,
        metadata={**signal.metadata, "filter": "exponential", "alpha": alpha},
    )
