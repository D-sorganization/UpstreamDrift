"""Error-budget numerical formatting for matched shot pattern reports and UI.

Provides pure Python formatting utilities for Strokes Gained (SG) values and
95% Monte Carlo confidence intervals. Preserves:
- <= 0.1% relative error for all nonzero estimates and endpoints
- <= 1.0% relative error for nonzero interval widths (endpoints bounded by 0.5% width)
- Nonzero effect visibility (no small nonzero effect is displayed as zero)
- Interval-sign relationships and zero-crossing integrity
- Unavailable / malformed data contracts without converting missing values to zero

Note: Meaningful finite display precision must not be confused with physical certainty.
The intervals reported describe Monte Carlo sampling uncertainty in simulated models,
not empirical player performance or deterministic physical law.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypeGuard


def is_finite_real(val: Any) -> TypeGuard[float]:
    """Return True if val is a finite real number (excluding bool)."""
    return (
        isinstance(val, (int, float))
        and not isinstance(val, bool)
        and math.isfinite(val)
    )


def format_sg_value(val: Any, *, absolute_tolerance: float | None = None) -> str:
    """Keep compact SG text within 0.1% relative and optional absolute error.

    Preconditions:
      - val is any value; non-finite or non-numeric values render as "unavailable"
      - absolute_tolerance is None or a nonnegative finite real error budget; zero requests exact round-trip
    Postconditions:
      - Returns a signed string ("+0.000", "+0.020", "-5.056e-04", etc.) or "unavailable"
      - If numeric, relative error is <= 0.1% and any supplied absolute budget is met
      - Minimum 3 decimal places for ordinary values (|val| >= 0.001)
      - Scientific notation for small values (|val| < 0.001) to prevent zeroing
    """
    if absolute_tolerance is not None and (
        not is_finite_real(absolute_tolerance) or absolute_tolerance < 0
    ):
        raise ValueError("absolute_tolerance must be nonnegative and finite")
    if not is_finite_real(val):
        return "unavailable"
    value = float(val)
    if value == 0.0:
        return "+0.000"
    error_budget = abs(value) * 0.001
    if absolute_tolerance is not None:
        error_budget = min(error_budget, absolute_tolerance)
    notation = "e" if abs(value) < 0.001 else "f"
    for precision in range(3, 18):
        rendered = format(value, f"+.{precision}{notation}")
        if abs(float(rendered) - value) <= error_budget:
            return rendered
    # Seventeen significant digits round-trip any finite Python float, including
    # an unusually narrow interval whose error budget is below fixed precision.
    return format(value, "+.17g")


def format_sg_interval(
    estimate: Any,
    lower_95: Any,
    upper_95: Any,
    *,
    check_reversed: bool = True,
) -> str:
    """Format an estimate and 95% Monte Carlo interval with width-budgeted precision.

    Preconditions:
      - estimate, lower_95, upper_95 are finite real numbers
      - If check_reversed is True, lower_95 <= upper_95
    Postconditions:
      - Returns a formatted string 'est [low, high]'
      - Nonzero interval width is retained without collapsing to identical endpoints
      - Each endpoint has an error budget <= 0.5% of width, guaranteeing <= 1.0% width error
    """
    if not (
        is_finite_real(estimate)
        and is_finite_real(lower_95)
        and is_finite_real(upper_95)
    ):
        raise ValueError("interval values must be finite real numbers")

    est = float(estimate)
    low = float(lower_95)
    high = float(upper_95)

    if check_reversed and low > high:
        raise ValueError("paired effect interval is reversed")

    width = high - low
    tolerance = 0.005 * width if 0.0 < width < math.inf else None

    est_str = format_sg_value(est, absolute_tolerance=tolerance)
    low_str = format_sg_value(low, absolute_tolerance=tolerance)
    high_str = format_sg_value(high, absolute_tolerance=tolerance)

    return f"{est_str} [{low_str}, {high_str}]"


def format_sg_interval_record(
    data: Any,
    *,
    unavailable_text: str = "Unavailable",
    check_reversed: bool = True,
) -> str:
    """Format a dictionary/mapping interval record containing estimate, lower_95, upper_95.

    Preconditions:
      - data is None, a Mapping, or any object
    Postconditions:
      - Returns formatted interval 'est [low, high]' or unavailable_text
      - Raises ValueError if check_reversed is True and lower_95 > upper_95
    """
    if not isinstance(data, Mapping):
        return unavailable_text
    try:
        est = data["estimate"]
        low = data["lower_95"]
        high = data["upper_95"]
    except KeyError:
        return unavailable_text

    if not (is_finite_real(est) and is_finite_real(low) and is_finite_real(high)):
        return unavailable_text

    return format_sg_interval(est, low, high, check_reversed=check_reversed)
