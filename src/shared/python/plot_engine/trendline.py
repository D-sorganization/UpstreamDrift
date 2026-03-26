"""Shared trendline computation module.

Provides trendline fitting (linear, polynomial, exponential, power)
with equation string generation and R-squared computation. Used by
both the matplotlib renderer and Plotly converter.

Extracted from data_processor/core/signal_processing.py for DRY reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class TrendlineResult:
    """Result of a trendline computation."""

    trend_type: str
    coefficients: list[float]
    equation: str
    r_squared: float
    x_pred: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)


def compute_trendline(
    x: np.ndarray,
    y: np.ndarray,
    trend_type: Literal["linear", "polynomial", "exponential", "power"] = "linear",
    degree: int = 2,
    n_points: int = 200,
) -> TrendlineResult:
    """Compute a trendline for the given data.

    Args:
        x: X-axis values.
        y: Y-axis values.
        trend_type: Type of trendline to fit.
        degree: Polynomial degree (only for polynomial type).
        n_points: Number of points in the prediction curve.

    Returns:
        TrendlineResult with coefficients, equation, R², and prediction arrays.

    Raises:
        ValueError: If insufficient data points or invalid data for the fit type.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Remove NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) < 2:
        raise ValueError("At least 2 valid data points required for trendline")

    x_pred = np.linspace(x_valid.min(), x_valid.max(), n_points)

    if trend_type == "linear":
        return _linear(x_valid, y_valid, x_pred)
    if trend_type == "polynomial":
        return _polynomial(x_valid, y_valid, x_pred, degree)
    if trend_type == "exponential":
        return _exponential(x_valid, y_valid, x_pred)
    if trend_type == "power":
        return _power(x_valid, y_valid, x_pred)
    raise ValueError(f"Unknown trend type: {trend_type}")


def _r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared (coefficient of determination)."""
    if not (y is not None):
        raise ValueError("y must be provided")
    if not (y is not None):
        raise ValueError("y must be provided")
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0


def _linear(x: np.ndarray, y: np.ndarray, x_pred: np.ndarray) -> TrendlineResult:
    """Linear trendline: y = mx + b."""
    if not (x is not None):
        raise ValueError("x must be provided")
    if not (x is not None):
        raise ValueError("x must be provided")
    coeffs = np.polyfit(x, y, 1)
    m, b = coeffs
    y_fit = np.polyval(coeffs, x)
    y_pred = np.polyval(coeffs, x_pred)
    r2 = _r_squared(y, y_fit)

    sign = "+" if b >= 0 else "-"
    equation = f"y = {m:.4g}x {sign} {abs(b):.4g}"

    return TrendlineResult(
        trend_type="linear",
        coefficients=[float(m), float(b)],
        equation=equation,
        r_squared=r2,
        x_pred=x_pred,
        y_pred=y_pred,
    )


def _polynomial(
    x: np.ndarray, y: np.ndarray, x_pred: np.ndarray, degree: int
) -> TrendlineResult:
    """Polynomial trendline: y = a_n*x^n + ... + a_1*x + a_0."""
    if not (x is not None):
        raise ValueError("x must be provided")
    if not (x is not None):
        raise ValueError("x must be provided")
    degree = min(degree, len(x) - 1)
    coeffs = np.polyfit(x, y, degree)
    y_fit = np.polyval(coeffs, x)
    y_pred = np.polyval(coeffs, x_pred)
    r2 = _r_squared(y, y_fit)

    # Build equation string
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) < 1e-10:
            continue
        if power == 0:
            terms.append(f"{c:.4g}")
        elif power == 1:
            terms.append(f"{c:.4g}x")
        else:
            terms.append(f"{c:.4g}x^{power}")
    equation = "y = " + " + ".join(terms) if terms else "y = 0"
    # Clean up "+ -" to "- "
    equation = equation.replace("+ -", "- ")

    return TrendlineResult(
        trend_type="polynomial",
        coefficients=[float(c) for c in coeffs],
        equation=equation,
        r_squared=r2,
        x_pred=x_pred,
        y_pred=y_pred,
    )


def _exponential(x: np.ndarray, y: np.ndarray, x_pred: np.ndarray) -> TrendlineResult:
    """Exponential trendline: y = a * exp(b * x)."""
    mask = y > 0
    if np.sum(mask) < 2:
        raise ValueError("Exponential fit requires at least 2 positive y values")

    x_pos = x[mask]
    y_pos = y[mask]

    # Log-transform for initial estimate
    log_y = np.log(y_pos)
    init_coeffs = np.polyfit(x_pos, log_y, 1)
    b_init = init_coeffs[0]
    a_init = np.exp(init_coeffs[1])

    def exp_func(xv: np.ndarray, a: float, b: float) -> np.ndarray:
        result: np.ndarray = a * np.exp(b * xv)
        return result

    try:
        popt, _ = curve_fit(exp_func, x_pos, y_pos, p0=[a_init, b_init], maxfev=5000)
        a, b = popt
    except RuntimeError:
        a, b = a_init, b_init

    y_fit = exp_func(x_pos, a, b)
    y_pred_arr = exp_func(x_pred, a, b)
    r2 = _r_squared(y_pos, y_fit)

    equation = f"y = {a:.4g} * exp({b:.4g}x)"

    return TrendlineResult(
        trend_type="exponential",
        coefficients=[float(a), float(b)],
        equation=equation,
        r_squared=r2,
        x_pred=x_pred,
        y_pred=y_pred_arr,
    )


def _power(x: np.ndarray, y: np.ndarray, x_pred: np.ndarray) -> TrendlineResult:
    """Power trendline: y = a * x^b."""
    mask = (x > 0) & (y > 0)
    if np.sum(mask) < 2:
        raise ValueError("Power fit requires at least 2 positive x and y values")

    x_pos = x[mask]
    y_pos = y[mask]

    log_x = np.log(x_pos)
    log_y = np.log(y_pos)
    coeffs = np.polyfit(log_x, log_y, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])

    y_fit = a * (x_pos**b)
    # For prediction, only use positive x values
    x_pred_pos = x_pred[x_pred > 0]
    y_pred_arr = a * (x_pred_pos**b)

    r2 = _r_squared(y_pos, y_fit)
    equation = f"y = {a:.4g} * x^{b:.4g}"

    return TrendlineResult(
        trend_type="power",
        coefficients=[float(a), float(b)],
        equation=equation,
        r_squared=r2,
        x_pred=x_pred_pos,
        y_pred=y_pred_arr,
    )
