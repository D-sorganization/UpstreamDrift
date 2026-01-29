"""Signal limiting and saturation with smoothing.

This module provides functions for applying limits, saturation, rate limiting,
deadband, and hysteresis to signals with optional smoothing to prevent
discontinuities in control applications.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import numpy as np

from src.shared.python.signal_toolkit.core import Signal


class SaturationMode(Enum):
    """Saturation/clipping mode types."""

    HARD = "hard"  # Hard clipping (discontinuous)
    SOFT = "soft"  # Soft clipping with polynomial transition
    TANH = "tanh"  # Hyperbolic tangent (smooth)
    SIGMOID = "sigmoid"  # Logistic sigmoid
    ATAN = "atan"  # Arc tangent
    CUBIC = "cubic"  # Cubic polynomial
    EXPONENTIAL = "exponential"  # Exponential knee


def apply_saturation(
    signal: Signal,
    lower: float = -1.0,
    upper: float = 1.0,
    mode: SaturationMode = SaturationMode.HARD,
    smoothness: float = 1.0,
) -> Signal:
    """Apply saturation/limiting to a signal.

    Args:
        signal: Input signal.
        lower: Lower saturation limit.
        upper: Upper saturation limit.
        mode: Type of saturation curve.
        smoothness: Smoothness parameter for soft modes (larger = sharper transition).

    Returns:
        Signal with saturation applied.
    """
    values = signal.values.copy()
    result = _apply_saturation_values(values, lower, upper, mode, smoothness)

    return Signal(
        time=signal.time.copy(),
        values=result,
        name=f"{signal.name}_saturated",
        units=signal.units,
        metadata=dict(signal.metadata),
    )


def _apply_saturation_values(
    values: np.ndarray,
    lower: float,
    upper: float,
    mode: SaturationMode,
    smoothness: float,
) -> np.ndarray:
    """Apply saturation to raw values array.

    Args:
        values: Input values array.
        lower: Lower limit.
        upper: Upper limit.
        mode: Saturation mode.
        smoothness: Smoothness parameter.

    Returns:
        Saturated values array.
    """
    if mode == SaturationMode.HARD:
        return np.clip(values, lower, upper)

    # Normalize to [-1, 1] range for smooth functions
    center = (upper + lower) / 2
    half_range = (upper - lower) / 2

    if half_range == 0:
        return np.full_like(values, center)

    normalized = (values - center) / half_range

    if mode == SaturationMode.TANH:
        # tanh saturates smoothly
        result = np.tanh(smoothness * normalized) / np.tanh(smoothness)

    elif mode == SaturationMode.SIGMOID:
        # Logistic sigmoid: 2 / (1 + exp(-k*x)) - 1
        result = 2 / (1 + np.exp(-smoothness * normalized)) - 1

    elif mode == SaturationMode.ATAN:
        # Arctangent: (2/pi) * atan(k*x)
        result = (2 / np.pi) * np.arctan(smoothness * normalized)

    elif mode == SaturationMode.SOFT:
        # Soft clipping with polynomial transition
        result = _soft_clip(normalized, smoothness)

    elif mode == SaturationMode.CUBIC:
        # Cubic polynomial: 1.5*x - 0.5*x^3 for |x| < 1
        result = _cubic_clip(normalized, smoothness)

    elif mode == SaturationMode.EXPONENTIAL:
        # Exponential knee
        result = _exponential_clip(normalized, smoothness)

    else:
        result = np.clip(normalized, -1, 1)

    # Scale back to original range
    return result * half_range + center


def _soft_clip(x: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Soft clipping with smooth polynomial transition.

    Uses a quintic polynomial to ensure C2 continuity at the transition.

    Args:
        x: Normalized input values.
        k: Transition sharpness (larger = sharper).

    Returns:
        Soft-clipped values in [-1, 1].
    """
    result = np.zeros_like(x)
    threshold = 1.0 / k

    # Linear region
    linear_mask = np.abs(x) < threshold
    result[linear_mask] = x[linear_mask]

    # Transition region (use smooth polynomial)
    for sign in [1, -1]:
        mask = x * sign >= threshold
        if np.any(mask):
            # Map [threshold, inf) to [threshold, 1] smoothly
            y = (x[mask] * sign - threshold) / (1 - threshold + 1e-10)
            # Quintic hermite for smooth transition
            # f(0) = threshold, f(1) = 1, f'(0) = 1, f'(1) = 0, f''(0) = 0, f''(1) = 0
            t = np.clip(y, 0, 1)
            t3 = t * t * t
            t4 = t3 * t
            t5 = t4 * t
            # Quintic hermite interpolation
            h = 6 * t5 - 15 * t4 + 10 * t3
            result[mask] = sign * (threshold + (1 - threshold) * h)

    return result


def _cubic_clip(x: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Cubic soft clipping.

    f(x) = 1.5*x - 0.5*x^3 for |x| < 1, scaled by k.

    Args:
        x: Normalized input values.
        k: Sharpness parameter.

    Returns:
        Cubic-clipped values.
    """
    x_scaled = x * k
    mask = np.abs(x_scaled) < 1

    result = np.sign(x_scaled) * np.ones_like(x_scaled)
    result[mask] = 1.5 * x_scaled[mask] - 0.5 * x_scaled[mask] ** 3
    result /= k  # Scale back

    return np.clip(result, -1, 1)


def _exponential_clip(x: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Exponential knee soft clipping.

    Uses exponential decay near limits for smooth transition.

    Args:
        x: Normalized input values.
        k: Knee sharpness.

    Returns:
        Exponentially clipped values.
    """
    # f(x) = sign(x) * (1 - exp(-k*|x|)) / (1 - exp(-k))
    x_abs = np.abs(x)
    normalizer = 1 - np.exp(-k)
    if normalizer < 1e-10:
        normalizer = 1e-10

    result = np.sign(x) * (1 - np.exp(-k * x_abs)) / normalizer
    return np.clip(result, -1, 1)


def apply_rate_limiter(
    signal: Signal,
    max_rate: float,
    smooth_transition: bool = True,
    transition_time: float = 0.01,
) -> Signal:
    """Apply rate limiting to prevent rapid changes.

    Args:
        signal: Input signal.
        max_rate: Maximum rate of change (units per second).
        smooth_transition: Whether to smooth the rate transitions.
        transition_time: Time constant for smooth transitions.

    Returns:
        Rate-limited signal.
    """
    values = signal.values.copy()
    dt = signal.dt

    result = np.zeros_like(values)
    result[0] = values[0]

    max_delta = max_rate * dt

    for i in range(1, len(values)):
        delta = values[i] - result[i - 1]

        if smooth_transition:
            # Smooth rate limiting using exponential approach
            if abs(delta) > max_delta:
                # Apply exponential smoothing when rate exceeds limit
                alpha = min(1.0, dt / transition_time)
                target = result[i - 1] + np.sign(delta) * max_delta
                result[i] = result[i - 1] + alpha * (target - result[i - 1])
            else:
                result[i] = values[i]
        else:
            # Hard rate limiting
            if delta > max_delta:
                result[i] = result[i - 1] + max_delta
            elif delta < -max_delta:
                result[i] = result[i - 1] - max_delta
            else:
                result[i] = values[i]

    return Signal(
        time=signal.time.copy(),
        values=result,
        name=f"{signal.name}_rate_limited",
        units=signal.units,
        metadata=dict(signal.metadata),
    )


def apply_deadband(
    signal: Signal,
    threshold: float,
    center: float = 0.0,
    smooth: bool = True,
    smoothness: float = 10.0,
) -> Signal:
    """Apply deadband (dead zone) around a center value.

    Values within the deadband are mapped to the center value.

    Args:
        signal: Input signal.
        threshold: Half-width of the deadband.
        center: Center of the deadband.
        smooth: Whether to smooth the transition.
        smoothness: Smoothness parameter for transitions.

    Returns:
        Signal with deadband applied.
    """
    values = signal.values.copy()
    offset = values - center

    if smooth:
        # Smooth deadband using tanh
        # f(x) = x * tanh(k * (|x| - threshold)) / (|x| + epsilon) for |x| > threshold
        epsilon = 1e-10
        x_abs = np.abs(offset)

        # Smoothly transition from 0 to 1 as we leave the deadband
        transition = 0.5 * (np.tanh(smoothness * (x_abs - threshold)) + 1)

        # Output is proportional to distance from deadband edge
        result = np.sign(offset) * np.maximum(0, x_abs - threshold) * transition
        result += center

    else:
        # Hard deadband
        result = np.where(
            np.abs(offset) > threshold,
            center + np.sign(offset) * (np.abs(offset) - threshold),
            center,
        )

    return Signal(
        time=signal.time.copy(),
        values=result,
        name=f"{signal.name}_deadband",
        units=signal.units,
        metadata=dict(signal.metadata),
    )


def apply_hysteresis(
    signal: Signal,
    threshold_up: float,
    threshold_down: float,
    output_high: float = 1.0,
    output_low: float = 0.0,
    initial_state: bool = False,
    smooth: bool = False,
    smoothness: float = 10.0,
) -> Signal:
    """Apply hysteresis (Schmitt trigger) to the signal.

    Args:
        signal: Input signal.
        threshold_up: Threshold for low-to-high transition.
        threshold_down: Threshold for high-to-low transition.
        output_high: Output value when in high state.
        output_low: Output value when in low state.
        initial_state: Initial state (True = high).
        smooth: Whether to smooth transitions.
        smoothness: Smoothness for transitions.

    Returns:
        Signal with hysteresis applied.
    """
    values = signal.values
    result = np.zeros_like(values)

    state = initial_state

    for i, val in enumerate(values):
        if state:
            # Currently high, check for low transition
            if val < threshold_down:
                state = False
        else:
            # Currently low, check for high transition
            if val > threshold_up:
                state = True

        result[i] = output_high if state else output_low

    if smooth:
        # Apply smoothing to the transitions

        kernel_size = max(3, int(smoothness))
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = np.convolve(
            result,
            np.ones(kernel_size) / kernel_size,
            mode="same",
        )
        result = smoothed

    return Signal(
        time=signal.time.copy(),
        values=result,
        name=f"{signal.name}_hysteresis",
        units=signal.units,
        metadata=dict(signal.metadata),
    )


def apply_backlash(
    signal: Signal,
    backlash_width: float,
    smooth: bool = True,
    smoothness: float = 5.0,
) -> Signal:
    """Apply mechanical backlash model to the signal.

    Simulates gear backlash where the output doesn't respond
    until the input moves by the backlash width.

    Args:
        signal: Input signal.
        backlash_width: Total backlash width.
        smooth: Whether to smooth transitions.
        smoothness: Smoothness parameter.

    Returns:
        Signal with backlash applied.
    """
    values = signal.values
    half_width = backlash_width / 2
    result = np.zeros_like(values)
    result[0] = values[0]

    output = values[0]

    for i in range(1, len(values)):
        delta = values[i] - (output + half_width)
        if delta > 0:
            output = values[i] - half_width
        else:
            delta = values[i] - (output - half_width)
            if delta < 0:
                output = values[i] + half_width

        result[i] = output

    if smooth:
        # Apply low-pass filter for smoothing
        alpha = 1.0 / (1.0 + smoothness)
        smoothed = np.zeros_like(result)
        smoothed[0] = result[0]
        for i in range(1, len(result)):
            smoothed[i] = alpha * result[i] + (1 - alpha) * smoothed[i - 1]
        result = smoothed

    return Signal(
        time=signal.time.copy(),
        values=result,
        name=f"{signal.name}_backlash",
        units=signal.units,
        metadata=dict(signal.metadata),
    )


def create_saturation_function(
    lower: float = -1.0,
    upper: float = 1.0,
    mode: SaturationMode = SaturationMode.TANH,
    smoothness: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a reusable saturation function.

    Args:
        lower: Lower limit.
        upper: Upper limit.
        mode: Saturation mode.
        smoothness: Smoothness parameter.

    Returns:
        Function that applies saturation to values.
    """

    def saturate(values: np.ndarray) -> np.ndarray:
        return _apply_saturation_values(values, lower, upper, mode, smoothness)

    return saturate


def visualize_saturation_curves(
    lower: float = -1.0,
    upper: float = 1.0,
    smoothness: float = 1.0,
    num_points: int = 1000,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Generate data for visualizing different saturation curves.

    Args:
        lower: Lower limit.
        upper: Upper limit.
        smoothness: Smoothness parameter.
        num_points: Number of points to generate.

    Returns:
        Dictionary mapping mode name to (input, output) arrays.
    """
    # Generate input values that go beyond limits
    margin = (upper - lower) * 0.5
    x = np.linspace(lower - margin, upper + margin, num_points)

    curves = {}
    for mode in SaturationMode:
        y = _apply_saturation_values(x, lower, upper, mode, smoothness)
        curves[mode.value] = (x, y)

    return curves
