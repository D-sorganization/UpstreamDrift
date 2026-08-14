"""Shared invariant guards for the domain value objects (issue #8608).

One implementation of "this number must be a positive finite metre count", so
the six value objects cannot drift into six spellings of the same rejection.
Every guard ``raise``s: ``python -O`` strips ``assert``, and a guard that
disappears under optimisation is not a guard (ADR-0032).
"""

from __future__ import annotations

import math

from ..exceptions import DomainInvariantError

__all__ = [
    "require_finite",
    "require_non_negative",
    "require_open_range",
    "require_positive",
    "require_positive_int",
    "require_unit_interval",
]


def require_finite(value: float, name: str) -> float:
    """Return ``value`` as a float, refusing NaN and infinities.

    Args:
        value: Quantity under test.
        name: Field name, quoted verbatim in the error message.

    Returns:
        ``float(value)``.

    Raises:
        DomainInvariantError: The value is not a finite number.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DomainInvariantError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise DomainInvariantError(f"{name} must be finite, got {number!r}")
    return number


def require_positive(value: float, name: str) -> float:
    """Return ``value`` as a strictly positive finite float.

    Args:
        value: Quantity under test.
        name: Field name, quoted verbatim in the error message.

    Returns:
        ``float(value)``.

    Raises:
        DomainInvariantError: The value is not finite or is not positive.
    """
    number = require_finite(value, name)
    if number <= 0.0:
        raise DomainInvariantError(f"{name} must be positive, got {number!r}")
    return number


def require_non_negative(value: float, name: str) -> float:
    """Return ``value`` as a non-negative finite float.

    Args:
        value: Quantity under test.
        name: Field name, quoted verbatim in the error message.

    Returns:
        ``float(value)``.

    Raises:
        DomainInvariantError: The value is not finite or is negative.
    """
    number = require_finite(value, name)
    if number < 0.0:
        raise DomainInvariantError(f"{name} must not be negative, got {number!r}")
    return number


def require_positive_int(value: int, name: str) -> int:
    """Return ``value`` as a strictly positive integer.

    Args:
        value: Quantity under test.
        name: Field name, quoted verbatim in the error message.

    Returns:
        ``int(value)``.

    Raises:
        DomainInvariantError: The value is not an integer or is not positive.
    """
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise DomainInvariantError(f"{name} must be an integer, got {value!r}") from exc
    if number <= 0:
        raise DomainInvariantError(f"{name} must be positive, got {number!r}")
    return number


def require_unit_interval(value: float, name: str) -> float:
    """Return ``value`` as a float inside the closed interval ``[0, 1]``.

    Args:
        value: Quantity under test.
        name: Field name, quoted verbatim in the error message.

    Returns:
        ``float(value)``.

    Raises:
        DomainInvariantError: The value lies outside ``[0, 1]``.
    """
    number = require_finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise DomainInvariantError(f"{name} must lie within [0, 1], got {number!r}")
    return number


def require_open_range(value: float, name: str, low: float, high: float) -> float:
    """Return ``value`` as a float strictly inside ``(low, high)``.

    Args:
        value: Quantity under test.
        name: Field name, quoted verbatim in the error message.
        low: Exclusive lower bound.
        high: Exclusive upper bound.

    Returns:
        ``float(value)``.

    Raises:
        DomainInvariantError: The value lies outside the open interval.
    """
    number = require_finite(value, name)
    if not low < number < high:
        raise DomainInvariantError(
            f"{name} must lie strictly between {low} and {high}, got {number!r}"
        )
    return number
