"""Accuracy and contract tests for ``grip_contact.deterministic_math`` (#13552)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.grip_contact.deterministic_math import (
    closest_point_segment3,
    det_atan2,
    det_exp,
    det_sin_cos,
    det_tanh,
    dot3,
    matmul3,
    matvec3,
    norm3,
)

_EPS = np.finfo(float).eps


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), np.finfo(float).tiny)


@pytest.mark.unit
def test_exp_and_tanh_track_the_library_within_a_few_ulp() -> None:
    rng = np.random.default_rng(13552)
    xs = np.concatenate(
        [
            rng.uniform(-40.0, 5.0, 2000),
            rng.uniform(-1e-3, 1e-3, 2000),
            rng.uniform(-30.0, 30.0, 2000),
            [0.0, -0.0, 1e-300, -1e-300, 20.0, -20.0, 25.0],
        ]
    )
    for x in map(float, xs):
        assert _rel(det_exp(x), math.exp(x)) <= 2 * _EPS, x
        if math.tanh(x) != 0.0:
            assert _rel(det_tanh(x), math.tanh(x)) <= 4 * _EPS, x
    assert det_exp(-800.0) == 0.0
    assert det_tanh(0.0) == 0.0


@pytest.mark.unit
def test_sin_cos_and_atan2_track_the_library_within_a_few_ulp() -> None:
    rng = np.random.default_rng(13334)
    for x in map(float, rng.uniform(-10.0, 10.0, 4000)):
        s, c = det_sin_cos(x)
        assert abs(s - math.sin(x)) <= 4 * _EPS, x
        assert abs(c - math.cos(x)) <= 4 * _EPS, x
    pts = rng.normal(size=(4000, 2))
    for y, x in map(tuple, pts):
        assert abs(det_atan2(y, x) - math.atan2(y, x)) <= 4 * _EPS * math.pi
    for y, x in [(0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 0.0)]:
        assert det_atan2(y, x) == pytest.approx(math.atan2(y, x), abs=1e-15)
    with pytest.raises(ValueError, match="finite"):
        det_sin_cos(math.inf)


@pytest.mark.unit
def test_linear_algebra_helpers_match_numpy() -> None:
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    v, w = rng.normal(size=3), rng.normal(size=3)
    assert dot3(v, w) == pytest.approx(float(v @ w), rel=1e-14)
    assert norm3(v) == pytest.approx(float(np.linalg.norm(v)), rel=1e-14)
    np.testing.assert_allclose(matvec3(a, v), a @ v, rtol=1e-13)
    np.testing.assert_allclose(matmul3(a, b), a @ b, rtol=1e-13, atol=1e-15)
    p0, p1 = np.zeros(3), np.array([1.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        closest_point_segment3(np.array([0.25, 1.0, 0.0]), p0, p1),
        [0.25, 0.0, 0.0],
    )
    np.testing.assert_array_equal(
        closest_point_segment3(np.array([2.0, 1.0, 0.0]), p0, p1), p1
    )
    np.testing.assert_array_equal(closest_point_segment3(v, p0, p0.copy()), p0)
