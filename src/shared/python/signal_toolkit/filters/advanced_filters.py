"""Advanced signal filters: Gaussian, bilateral, and adaptive (LMS/RLS)."""

from __future__ import annotations

import numpy as np

from src.shared.python.core.contracts import require  # type: ignore[import-untyped]

from ..core import Signal


def apply_gaussian_smoothing(
    signal: Signal,
    sigma: float = 1.0,
) -> Signal:
    """Apply Gaussian smoothing to a signal.

    Args:
        signal: Input signal.
        sigma: Standard deviation of Gaussian kernel.

    Returns:
        Smoothed signal.
    """
    if not (signal is not None):
        raise ValueError("signal must be provided")
    if not (signal is not None):
        raise ValueError("signal must be provided")
    require(sigma > 0.0, f"sigma must be positive, got {sigma}")
    from scipy.ndimage import gaussian_filter1d

    filtered_values = gaussian_filter1d(signal.values, sigma)

    return Signal(
        time=signal.time,
        values=filtered_values,
        name=f"{signal.name}_gaussian",
        units=signal.units,
        metadata={**signal.metadata, "filter": "gaussian", "sigma": sigma},
    )


def apply_bilateral_filter(
    signal: Signal,
    window_size: int = 5,
    sigma_space: float = 1.0,
    sigma_intensity: float = 0.1,
) -> Signal:
    """Apply bilateral filter to a signal.

    Edge-preserving smoothing filter.

    Args:
        signal: Input signal.
        window_size: Size of the filter window.
        sigma_space: Spatial sigma (controls distance weighting).
        sigma_intensity: Intensity sigma (controls value similarity weighting).

    Returns:
        Filtered signal.
    """
    if not (signal is not None):
        raise ValueError("signal must be provided")
    if not (signal is not None):
        raise ValueError("signal must be provided")
    values = signal.values
    n = len(values)
    filtered = np.zeros(n)

    half_window = window_size // 2

    for i in range(n):
        # Get window
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)

        # Spatial weights (distance from center)
        positions = np.arange(start, end)
        spatial_weights = np.exp(-((positions - i) ** 2) / (2 * sigma_space**2))

        # Intensity weights (value similarity)
        intensity_weights = np.exp(
            -((values[start:end] - values[i]) ** 2) / (2 * sigma_intensity**2)
        )

        # Combined weights
        weights = spatial_weights * intensity_weights
        weights /= np.sum(weights) + 1e-10

        filtered[i] = np.sum(weights * values[start:end])

    return Signal(
        time=signal.time,
        values=filtered,
        name=f"{signal.name}_bilateral",
        units=signal.units,
        metadata={
            **signal.metadata,
            "filter": "bilateral",
            "window": window_size,
            "sigma_space": sigma_space,
            "sigma_intensity": sigma_intensity,
        },
    )


class AdaptiveFilter:
    """Adaptive filter implementations (LMS, RLS)."""

    @staticmethod
    def lms(
        signal: Signal,
        reference: Signal,
        order: int = 10,
        step_size: float = 0.01,
    ) -> tuple[Signal, Signal]:
        """Apply Least Mean Squares (LMS) adaptive filter.

        Args:
            signal: Input signal to filter.
            reference: Reference signal (desired output).
            order: Filter order.
            step_size: LMS step size (learning rate).

        Returns:
            Tuple of (filtered_signal, error_signal).
        """
        if not (signal is not None):
            raise ValueError("signal must be provided")
        if not (signal is not None):
            raise ValueError("signal must be provided")
        n = len(signal.values)
        x = signal.values
        d = reference.values

        w = np.zeros(order)  # Filter weights
        y = np.zeros(n)  # Filter output
        e = np.zeros(n)  # Error

        for i in range(order, n):
            x_window = x[i - order : i][::-1]  # Reversed window
            y[i] = np.dot(w, x_window)
            e[i] = d[i] - y[i]
            w += step_size * e[i] * x_window

        filtered = Signal(
            time=signal.time,
            values=y,
            name=f"{signal.name}_lms",
            units=signal.units,
        )

        error = Signal(
            time=signal.time,
            values=e,
            name=f"{signal.name}_lms_error",
            units=signal.units,
        )

        return filtered, error

    @staticmethod
    def rls(
        signal: Signal,
        reference: Signal,
        order: int = 10,
        forgetting_factor: float = 0.99,
        delta: float = 0.01,
    ) -> tuple[Signal, Signal]:
        """Apply Recursive Least Squares (RLS) adaptive filter.

        Args:
            signal: Input signal to filter.
            reference: Reference signal (desired output).
            order: Filter order.
            forgetting_factor: Forgetting factor (0 < lambda <= 1).
            delta: Initialization value for P matrix.

        Returns:
            Tuple of (filtered_signal, error_signal).
        """
        if not (signal is not None):
            raise ValueError("signal must be provided")
        if not (signal is not None):
            raise ValueError("signal must be provided")
        n = len(signal.values)
        x = signal.values
        d = reference.values

        w = np.zeros(order)  # Filter weights
        P = np.eye(order) / delta  # Inverse correlation matrix
        y = np.zeros(n)  # Filter output
        e = np.zeros(n)  # Error

        lam = forgetting_factor

        for i in range(order, n):
            x_window = x[i - order : i][::-1].reshape(-1, 1)
            y[i] = np.dot(w, x_window.flatten())
            e[i] = d[i] - y[i]

            # RLS update
            k = P @ x_window / (lam + x_window.T @ P @ x_window)
            P = (P - k @ x_window.T @ P) / lam
            w += k.flatten() * e[i]

        filtered = Signal(
            time=signal.time,
            values=y,
            name=f"{signal.name}_rls",
            units=signal.units,
        )

        error = Signal(
            time=signal.time,
            values=e,
            name=f"{signal.name}_rls_error",
            units=signal.units,
        )

        return filtered, error
