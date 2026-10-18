"""``assumption_digest`` is stable across platforms (NSX-99, #12728).

A digest must not change with last-bit floating-point noise (BLAS or numpy
build differences), yet must change when a declared assumption changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.sensitivity_identity import (
    assumption_digest,
)

pytestmark = pytest.mark.unit

_BASE = np.array([[0.3, 0.004], [1.0e-7, -2.5]])


def _noisy(values: np.ndarray) -> np.ndarray:
    return np.nextafter(values, np.inf)


def test_last_bit_noise_in_an_array_keeps_the_digest() -> None:
    assert assumption_digest({"a": _BASE}) == assumption_digest({"a": _noisy(_BASE)})


def test_last_bit_noise_in_a_float_keeps_the_digest() -> None:
    assert assumption_digest({"x": 0.1 + 0.2}) == assumption_digest({"x": 0.3})


def test_a_real_change_changes_the_digest() -> None:
    changed = _BASE.copy()
    changed[0, 0] *= 1.0 + 1.0e-9
    assert assumption_digest({"a": _BASE}) != assumption_digest({"a": changed})
    assert assumption_digest({"x": 0.3}) != assumption_digest({"x": 0.3000001})


def test_signed_zero_and_non_finite_values_are_stable() -> None:
    assert assumption_digest({"a": np.array([0.0])}) == assumption_digest(
        {"a": np.array([-0.0])}
    )
    inf = np.array([np.inf, -np.inf, np.nan])
    assert assumption_digest({"a": inf}) == assumption_digest({"a": inf.copy()})


def test_array_shape_is_part_of_the_digest() -> None:
    assert assumption_digest({"a": _BASE}) != assumption_digest(
        {"a": _BASE.reshape(4, 1)}
    )
