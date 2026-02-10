"""Differentiation and integration utilities for signals.

This module provides tools for computing derivatives, integrals,
tangent lines, and visualizing calculus operations on signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import integrate
from scipy.signal import savgol_filter

from src.shared.python.signal_toolkit.core import Signal


class DifferentiationMethod(Enum):
    """Methods for computing derivatives."""

    FORWARD = "forward"  # Forward difference
    BACKWARD = "backward"  # Backward difference
    CENTRAL = "central"  # Central difference
    SAVGOL = "savgol"  # Savitzky-Golay filter
    GRADIENT = "gradient"  # NumPy gradient (central, edge-aware)
    SPLINE = "spline"  # Spline interpolation derivative


class IntegrationMethod(Enum):
    """Methods for computing integrals."""

    TRAPEZOID = "trapezoid"  # Trapezoidal rule
    SIMPSON = "simpson"  # Simpson's rule
    CUMULATIVE = "cumulative"  # Cumulative trapezoidal


@dataclass
class TangentLine:
    """Represents a tangent line at a point on a signal.

    Attributes:
        t_point: Time value where tangent is computed.
        y_point: Signal value at the point.
        slope: Derivative (slope) at the point.
        t_range: Time range for the tangent line [t_start, t_end].
        line_values: Values of the tangent line over t_range.
    """

    t_point: float
    y_point: float
    slope: float
    t_range: np.ndarray
    line_values: np.ndarray

    def get_equation_string(self) -> str:
        """Get the equation of the tangent line."""
        return f"y = {self.slope:.4f}*(t - {self.t_point:.4f}) + {self.y_point:.4f}"


@dataclass
class IntegralResult:
    """Result of an integration operation.

    Attributes:
        value: The definite integral value.
        lower_bound: Lower integration bound.
        upper_bound: Upper integration bound.
        cumulative_signal: Signal with cumulative integral values.
        area_positive: Area of positive regions.
        area_negative: Area of negative regions.
    """

    value: float
    lower_bound: float
    upper_bound: float
    cumulative_signal: Signal | None
    area_positive: float
    area_negative: float


class Differentiator:
    """Class for computing signal derivatives."""

    def __init__(
        self,
        method: DifferentiationMethod = DifferentiationMethod.SAVGOL,
        window_length: int = 7,
        polyorder: int = 2,
    ) -> None:
        """Initialize the differentiator.

        Args:
            method: Differentiation method to use.
            window_length: Window length for Savitzky-Golay filter (odd integer).
            polyorder: Polynomial order for Savitzky-Golay filter.
        """
        self.method = method
        self.window_length = window_length
        self.polyorder = polyorder

    def differentiate(
        self,
        signal: Signal,
        order: int = 1,
    ) -> Signal:
        """Compute the derivative of a signal.

        Args:
            signal: Input signal.
            order: Order of derivative (1 = first derivative, 2 = second, etc.).

        Returns:
            Signal containing the derivative.
        """
        result = signal.copy()
        result.name = f"d{order}({signal.name})/dt{order}"
        result.units = f"{signal.units}/s^{order}" if signal.units else ""

        for _ in range(order):
            result.values = self._differentiate_once(result)

        return result

    def _differentiate_once(self, signal: Signal) -> np.ndarray:
        """Compute first derivative of signal values.

        Args:
            signal: Input signal.

        Returns:
            Array of derivative values.
        """
        t = signal.time
        y = signal.values
        dt = signal.dt

        if self.method == DifferentiationMethod.FORWARD:
            # Forward difference: (y[i+1] - y[i]) / dt
            dy = np.zeros_like(y)
            dy[:-1] = (y[1:] - y[:-1]) / dt
            dy[-1] = dy[-2]  # Repeat last value

        elif self.method == DifferentiationMethod.BACKWARD:
            # Backward difference: (y[i] - y[i-1]) / dt
            dy = np.zeros_like(y)
            dy[1:] = (y[1:] - y[:-1]) / dt
            dy[0] = dy[1]  # Repeat first value

        elif self.method == DifferentiationMethod.CENTRAL:
            # Central difference: (y[i+1] - y[i-1]) / (2*dt)
            dy = np.zeros_like(y)
            dy[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
            dy[0] = (y[1] - y[0]) / dt  # Forward at start
            dy[-1] = (y[-1] - y[-2]) / dt  # Backward at end

        elif self.method == DifferentiationMethod.GRADIENT:
            # NumPy gradient (handles edges properly)
            dy = np.gradient(y, t)

        elif self.method == DifferentiationMethod.SAVGOL:
            # Savitzky-Golay filter derivative
            window = min(self.window_length, len(y))
            if window % 2 == 0:
                window -= 1
            window = max(window, self.polyorder + 2)

            if len(y) > window:
                dy = savgol_filter(
                    y,
                    window_length=window,
                    polyorder=self.polyorder,
                    deriv=1,
                    delta=dt,
                )
            else:
                # Fallback to gradient for short signals
                dy = np.gradient(y, t)

        elif self.method == DifferentiationMethod.SPLINE:
            # Spline interpolation derivative
            from scipy.interpolate import UnivariateSpline

            try:
                spline = UnivariateSpline(t, y, s=0)
                dy = spline.derivative()(t)
            except (RuntimeError, ValueError, OSError):
                dy = np.gradient(y, t)

        else:
            dy = np.gradient(y, t)

        return dy

    def compute_at_point(
        self,
        signal: Signal,
        t_point: float,
        order: int = 1,
    ) -> float:
        """Compute derivative at a specific time point.

        Args:
            signal: Input signal.
            t_point: Time at which to evaluate derivative.
            order: Order of derivative.

        Returns:
            Derivative value at the point.
        """
        derivative_signal = self.differentiate(signal, order)

        # Interpolate to exact point
        idx = np.searchsorted(signal.time, t_point)
        idx = np.clip(idx, 0, len(signal.time) - 1)

        # Linear interpolation for exact value
        if idx > 0 and idx < len(signal.time) - 1:
            t0, t1 = signal.time[idx - 1], signal.time[idx]
            y0, y1 = derivative_signal.values[idx - 1], derivative_signal.values[idx]
            alpha = (t_point - t0) / (t1 - t0) if t1 != t0 else 0
            return y0 + alpha * (y1 - y0)

        return derivative_signal.values[idx]


class Integrator:
    """Class for computing signal integrals."""

    def __init__(
        self,
        method: IntegrationMethod = IntegrationMethod.TRAPEZOID,
    ) -> None:
        """Initialize the integrator.

        Args:
            method: Integration method to use.
        """
        self.method = method

    def integrate(
        self,
        signal: Signal,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        initial_value: float = 0.0,
    ) -> IntegralResult:
        """Compute the integral of a signal.

        Args:
            signal: Input signal.
            lower_bound: Lower integration bound (default: signal start).
            upper_bound: Upper integration bound (default: signal end).
            initial_value: Initial value for cumulative integral.

        Returns:
            IntegralResult with integral value and related data.
        """
        t = signal.time
        y = signal.values

        if lower_bound is None:
            lower_bound = t[0]
        if upper_bound is None:
            upper_bound = t[-1]

        # Find indices for bounds
        lower_idx = np.searchsorted(t, lower_bound)
        upper_idx = np.searchsorted(t, upper_bound)

        lower_idx = np.clip(lower_idx, 0, len(t) - 1)
        upper_idx = np.clip(upper_idx, 0, len(t))

        t_range = t[lower_idx:upper_idx]
        y_range = y[lower_idx:upper_idx]

        # Compute definite integral
        if self.method == IntegrationMethod.TRAPEZOID:
            value = np.trapezoid(y_range, t_range)

        elif self.method == IntegrationMethod.SIMPSON:
            if len(t_range) >= 3:
                value = integrate.simpson(y_range, x=t_range)
            else:
                value = np.trapezoid(y_range, t_range)

        else:
            value = np.trapezoid(y_range, t_range)

        # Compute cumulative integral
        cumulative = integrate.cumulative_trapezoid(y, t, initial=initial_value)
        cumulative_signal = Signal(
            time=t,
            values=cumulative,
            name=f"integral({signal.name})",
            units=f"{signal.units}*s" if signal.units else "s",
        )

        # Compute positive and negative areas
        area_positive = np.trapezoid(np.maximum(y_range, 0), t_range)
        area_negative = np.trapezoid(np.minimum(y_range, 0), t_range)

        return IntegralResult(
            value=value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            cumulative_signal=cumulative_signal,
            area_positive=area_positive,
            area_negative=abs(area_negative),
        )

    def cumulative_integral(
        self,
        signal: Signal,
        initial_value: float = 0.0,
    ) -> Signal:
        """Compute cumulative (running) integral of a signal.

        Args:
            signal: Input signal.
            initial_value: Initial value of the integral.

        Returns:
            Signal with cumulative integral values.
        """
        cumulative = integrate.cumulative_trapezoid(
            signal.values,
            signal.time,
            initial=initial_value,
        )

        return Signal(
            time=signal.time,
            values=cumulative,
            name=f"integral({signal.name})",
            units=f"{signal.units}*s" if signal.units else "s",
            metadata=dict(signal.metadata),
        )


def compute_derivative(
    signal: Signal,
    order: int = 1,
    method: DifferentiationMethod = DifferentiationMethod.SAVGOL,
    **kwargs,
) -> Signal:
    """Convenience function to compute signal derivative.

    Args:
        signal: Input signal.
        order: Order of derivative.
        method: Differentiation method.
        **kwargs: Additional arguments for the differentiator.

    Returns:
        Signal containing the derivative.
    """
    diff = Differentiator(method=method, **kwargs)
    return diff.differentiate(signal, order)


def compute_integral(
    signal: Signal,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    method: IntegrationMethod = IntegrationMethod.TRAPEZOID,
) -> IntegralResult:
    """Convenience function to compute signal integral.

    Args:
        signal: Input signal.
        lower_bound: Lower integration bound.
        upper_bound: Upper integration bound.
        method: Integration method.

    Returns:
        IntegralResult with integral value and data.
    """
    integrator = Integrator(method=method)
    return integrator.integrate(signal, lower_bound, upper_bound)


def compute_tangent_line(
    signal: Signal,
    t_point: float,
    line_width: float | None = None,
    method: DifferentiationMethod = DifferentiationMethod.SAVGOL,
) -> TangentLine:
    """Compute tangent line at a specific point on the signal.

    Args:
        signal: Input signal.
        t_point: Time at which to compute tangent.
        line_width: Width of tangent line to generate (default: 10% of signal duration).
        method: Differentiation method for slope computation.

    Returns:
        TangentLine object with tangent information.
    """
    # Clamp t_point to signal range
    t_point = np.clip(t_point, signal.time[0], signal.time[-1])

    # Get signal value at point
    idx = np.searchsorted(signal.time, t_point)
    idx = np.clip(idx, 0, len(signal.time) - 1)

    # Interpolate for exact y value
    if idx > 0 and idx < len(signal.time):
        t0, t1 = signal.time[idx - 1], signal.time[idx]
        y0, y1 = signal.values[idx - 1], signal.values[idx]
        if t1 != t0:
            alpha = (t_point - t0) / (t1 - t0)
            y_point = y0 + alpha * (y1 - y0)
        else:
            y_point = signal.values[idx]
    else:
        y_point = signal.values[idx]

    # Compute derivative at point
    diff = Differentiator(method=method)
    slope = diff.compute_at_point(signal, t_point)

    # Generate tangent line
    if line_width is None:
        line_width = 0.1 * signal.duration

    t_start = max(t_point - line_width / 2, signal.time[0])
    t_end = min(t_point + line_width / 2, signal.time[-1])
    t_range = np.linspace(t_start, t_end, 100)

    # Tangent line: y = slope * (t - t_point) + y_point
    line_values = slope * (t_range - t_point) + y_point

    return TangentLine(
        t_point=t_point,
        y_point=y_point,
        slope=slope,
        t_range=t_range,
        line_values=line_values,
    )


def compute_all_tangent_lines(
    signal: Signal,
    num_points: int = 10,
    method: DifferentiationMethod = DifferentiationMethod.SAVGOL,
) -> list[TangentLine]:
    """Compute tangent lines at multiple evenly-spaced points.

    Args:
        signal: Input signal.
        num_points: Number of tangent lines to compute.
        method: Differentiation method.

    Returns:
        List of TangentLine objects.
    """
    t_points = np.linspace(signal.time[0], signal.time[-1], num_points + 2)[1:-1]
    line_width = signal.duration / (num_points + 1) * 0.8

    tangents = []
    for t in t_points:
        tangent = compute_tangent_line(signal, t, line_width, method)
        tangents.append(tangent)

    return tangents


def compute_curvature(
    signal: Signal,
    method: DifferentiationMethod = DifferentiationMethod.SAVGOL,
) -> Signal:
    """Compute curvature of a signal.

    Curvature = |y''| / (1 + y'^2)^(3/2)

    Args:
        signal: Input signal.
        method: Differentiation method.

    Returns:
        Signal containing curvature values.
    """
    diff = Differentiator(method=method)

    y_prime = diff.differentiate(signal, order=1).values
    y_double_prime = diff.differentiate(signal, order=2).values

    # Curvature formula
    curvature = np.abs(y_double_prime) / (1 + y_prime**2) ** 1.5

    return Signal(
        time=signal.time,
        values=curvature,
        name=f"curvature({signal.name})",
        units="1/units",
        metadata=dict(signal.metadata),
    )


def compute_arc_length(
    signal: Signal,
    method: DifferentiationMethod = DifferentiationMethod.SAVGOL,
) -> float:
    """Compute arc length of a signal curve.

    Arc length = integral(sqrt(1 + (dy/dt)^2) dt)

    Args:
        signal: Input signal.
        method: Differentiation method.

    Returns:
        Total arc length.
    """
    diff = Differentiator(method=method)
    y_prime = diff.differentiate(signal, order=1).values

    # Arc length element: ds = sqrt(1 + (dy/dt)^2) * dt
    ds = np.sqrt(1 + y_prime**2)

    return np.trapezoid(ds, signal.time)


def find_extrema(
    signal: Signal,
    method: DifferentiationMethod = DifferentiationMethod.SAVGOL,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Find local maxima and minima of a signal.

    Args:
        signal: Input signal.
        method: Differentiation method for zero-crossing detection.

    Returns:
        Tuple of (maxima, minima) where each is a list of (time, value) tuples.
    """
    diff = Differentiator(method=method)
    derivative = diff.differentiate(signal, order=1)

    # Find zero crossings of derivative
    dy = derivative.values
    sign_changes = np.diff(np.sign(dy))

    maxima = []
    minima = []

    for i in np.where(sign_changes != 0)[0]:
        t = signal.time[i]
        y = signal.values[i]

        if sign_changes[i] < 0:  # Positive to negative = maximum
            maxima.append((t, y))
        else:  # Negative to positive = minimum
            minima.append((t, y))

    return maxima, minima


def find_inflection_points(
    signal: Signal,
    method: DifferentiationMethod = DifferentiationMethod.SAVGOL,
) -> list[tuple[float, float]]:
    """Find inflection points where concavity changes.

    Args:
        signal: Input signal.
        method: Differentiation method.

    Returns:
        List of (time, value) tuples for inflection points.
    """
    diff = Differentiator(method=method)
    second_derivative = diff.differentiate(signal, order=2)

    # Find zero crossings of second derivative
    d2y = second_derivative.values
    sign_changes = np.diff(np.sign(d2y))

    inflections = []
    for i in np.where(sign_changes != 0)[0]:
        t = signal.time[i]
        y = signal.values[i]
        inflections.append((t, y))

    return inflections
