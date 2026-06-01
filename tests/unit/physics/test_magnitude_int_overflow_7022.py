"""Regression tests for integer-overflow in math.sqrt(np.dot) magnitude helpers (#7022).

Replacing ``np.linalg.norm(x)`` with ``math.sqrt(np.dot(x, x))`` is faster but
silently overflows when ``x`` has an integer dtype, because ``np.dot`` keeps the
integer dtype and can wrap before the square root. ``np.linalg.norm`` promoted to
float implicitly. The shared ``_magnitude`` helpers must cast to float first.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.physics._friction_laws import _magnitude as friction_magnitude
from src.shared.python.physics._terrain_physics import _magnitude as terrain_magnitude
from src.shared.python.physics.aerodynamics._rust_facade import (
    _magnitude as aero_magnitude,
)

_HELPERS = pytest.mark.parametrize(
    "magnitude",
    [friction_magnitude, terrain_magnitude, aero_magnitude],
)


@_HELPERS
def test_integer_array_does_not_overflow(magnitude) -> None:
    """An int32 vector whose squared sum exceeds int32 max must not wrap."""
    # 60000**2 + 60000**2 == 7.2e9 > int32 max (~2.147e9): would overflow if the
    # dot product stayed integer-typed.
    vec = np.array([60000, 60000, 0], dtype=np.int32)
    expected = float(np.linalg.norm(vec.astype(float)))
    assert magnitude(vec) == pytest.approx(expected, rel=1e-9)


@_HELPERS
def test_matches_linalg_norm_for_floats(magnitude) -> None:
    """The helper must be numerically equivalent to np.linalg.norm for floats."""
    vec = np.array([3.0, 4.0, 12.0])
    assert magnitude(vec) == pytest.approx(float(np.linalg.norm(vec)), rel=1e-12)


@_HELPERS
def test_zero_vector(magnitude) -> None:
    """The zero vector has zero magnitude (no division/precision surprises)."""
    assert magnitude(np.zeros(3)) == 0.0
