# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Centralized parameter validation for body/exercise inputs.

Issue #400: numeric parameters were previously accepted without checking
physical bounds, allowing impossible configurations (negative mass,
zero height, extreme durations) to flow into the optimizer and fail
deep in the call stack with cryptic errors.

This module is the single source of truth for valid ranges. Both the
GUI and the CLI should call :func:`validate_all` before constructing a
:class:`BodyModel` or invoking the optimizer.

Design Principles:
    DBC -- every validator raises ``ValueError`` with a self-describing
        message naming the offending parameter, the bad value and the
        valid range.
    DRY -- ranges and the inclusive bounds check live in one place.
"""

from __future__ import annotations

import math
from typing import Final

# ---------------------------------------------------------------------------
# Physical bounds. Values are inclusive on both ends.
# Sources: WHO anthropometric references (mass/height); Olympic lifting
# regulations (bar mass); empirical optimizer convergence (duration,
# smoothness).
# ---------------------------------------------------------------------------

BODY_MASS_RANGE: Final[tuple[float, float]] = (30.0, 300.0)  # kg
HEIGHT_RANGE: Final[tuple[float, float]] = (1.4, 2.2)  # m
BAR_MASS_RANGE: Final[tuple[float, float]] = (0.0, 500.0)  # kg
DURATION_RANGE: Final[tuple[float, float]] = (0.5, 10.0)  # s
SMOOTHNESS_RANGE: Final[tuple[float, float]] = (0.1, 100.0)  # unitless


def _check_finite(name: str, value: float) -> float:
    """Reject NaN / Inf inputs with a uniform message.

    Args:
        name: Human-readable parameter name used in the error message.
        value: The candidate value.

    Returns:
        ``value`` unchanged when finite.

    Raises:
        ValueError: If ``value`` is NaN or infinite, or not a real number.
        TypeError: If ``value`` is not numeric.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def _check_range(
    name: str,
    value: float,
    bounds: tuple[float, float],
    unit: str,
) -> None:
    """Raise ``ValueError`` if ``value`` is outside the inclusive ``bounds``."""
    low, high = bounds
    if not (low <= value <= high):
        raise ValueError(f"{name} must be in [{low}, {high}] {unit}, got {value}")


def validate_body_mass(mass: float) -> float:
    """Validate body mass in kg.

    Returns the value as a float on success.

    Raises:
        ValueError: If ``mass`` is non-finite or outside ``BODY_MASS_RANGE``.
    """
    v = _check_finite("body_mass", mass)
    _check_range("body_mass", v, BODY_MASS_RANGE, "kg")
    return v


def validate_height(height: float) -> float:
    """Validate body height in metres."""
    v = _check_finite("height", height)
    _check_range("height", v, HEIGHT_RANGE, "m")
    return v


def validate_bar_mass(mass: float) -> float:
    """Validate barbell mass in kg (0 = empty hand)."""
    v = _check_finite("bar_mass", mass)
    _check_range("bar_mass", v, BAR_MASS_RANGE, "kg")
    return v


def validate_duration(duration: float) -> float:
    """Validate movement duration in seconds."""
    v = _check_finite("duration", duration)
    _check_range("duration", v, DURATION_RANGE, "s")
    return v


def validate_smoothness(smoothness: float) -> float:
    """Validate smoothness weight (unitless)."""
    v = _check_finite("smoothness", smoothness)
    _check_range("smoothness", v, SMOOTHNESS_RANGE, "(unitless)")
    return v


def validate_all(
    *,
    body_mass: float,
    height: float,
    bar_mass: float,
    duration: float,
    smoothness: float,
) -> None:
    """Validate every numeric input in one call.

    Keyword-only to keep call sites unambiguous. Raises the first
    :class:`ValueError` encountered (validators run in declaration order).
    """
    validate_body_mass(body_mass)
    validate_height(height)
    validate_bar_mass(bar_mass)
    validate_duration(duration)
    validate_smoothness(smoothness)
