"""Freshness comparison for generated grip-contact JSON (HRG-GS-M12, #13334).

The Simscape golden vectors and scenario trajectories are computed with numpy
and BLAS, whose last-ulp results differ between library versions and CPUs.
The scenario Hermite coefficients come from finite differences, which amplify
that noise. Measured between two generation environments (numpy 2.5.3 and the
environment that produced 1509faa671): golden vectors differed in 71 of 12,854
numbers by at most 6.7e-14 relative, and scenario arrays needed a bound of
1e-9 of each array's largest magnitude. A byte-for-byte freshness check
therefore cannot pass on every runner, so these files are compared
structurally, with each numeric array bounded by :data:`FRESHNESS_SCALE_RTOL`
of its own scale (10x the measured need). Real generator changes are orders of
magnitude larger. The generated MATLAB kernel and parameter JSON stay
byte-for-byte: they matched exactly across environments, and the R2025b receipt
binds the kernel bytes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["FRESHNESS_SCALE_RTOL", "generated_json_matches"]

FRESHNESS_SCALE_RTOL = 1.0e-8
_ABS_FLOOR = 1.0e-15


def _numeric_array(value: list[Any]) -> np.ndarray | None:
    """``value`` as a float array when it is a rectangular all-number list."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if _has_bool(value):
        return None
    return arr


def _has_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, list):
        return any(_has_bool(v) for v in value)
    return False


def _arrays_match(a: np.ndarray, b: np.ndarray, scale_rtol: float) -> bool:
    if a.shape != b.shape:
        return False
    if a.size == 0:
        return True
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    bound = scale_rtol * float(np.abs(a).max()) + _ABS_FLOOR
    return bool(np.abs(a - b).max() <= bound)


def _match(expected: Any, actual: Any, scale_rtol: float) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and expected.keys() == actual.keys()
            and all(_match(expected[k], actual[k], scale_rtol) for k in expected)
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        a, b = _numeric_array(expected), _numeric_array(actual)
        if a is not None and b is not None:
            return _arrays_match(a, b, scale_rtol)
        return all(_match(e, v, scale_rtol) for e, v in zip(expected, actual, strict=True))
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, int) and isinstance(actual, int):
        return expected == actual
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        if math.isnan(expected):
            return math.isnan(actual)
        return abs(expected - actual) <= scale_rtol * abs(expected) + _ABS_FLOOR
    return bool(expected == actual) and type(expected) is type(actual)


def generated_json_matches(
    expected: Any, actual: Any, scale_rtol: float = FRESHNESS_SCALE_RTOL
) -> bool:
    """Whether ``actual`` is the same generated JSON payload as ``expected``.

    Preconditions:
        ``scale_rtol`` is positive and finite.

    Postconditions:
        True only when keys, list lengths, strings, booleans and integers are
        identical and every numeric array (or float) differs from the expected
        one by at most ``scale_rtol`` times its largest magnitude plus 1e-15.
    """
    if not (math.isfinite(scale_rtol) and scale_rtol > 0.0):
        raise ValueError(f"scale_rtol must be positive and finite, got {scale_rtol}")
    return _match(expected, actual, scale_rtol)
