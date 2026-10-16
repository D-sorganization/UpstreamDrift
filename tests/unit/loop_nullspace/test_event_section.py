"""Event projection and saltation against the NSX-80 symbolic goldens (NSX-83)."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.loop_nullspace.event_section import (
    GRAZING_TRANSVERSALITY,
    event_projection,
    event_time_shift_row,
    saltation_matrix,
    transversality,
)

pytestmark = pytest.mark.unit

GOLDEN = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "loop_nullspace"
    / "goldens"
    / "variability_sensitivity.json"
)


def _num(text: str) -> float:
    return float(Fraction(text))


@pytest.fixture(scope="module")
def salt() -> dict:
    return json.loads(GOLDEN.read_text())["saltation"]


def _vec(rows: list) -> np.ndarray:
    return np.array([_num(r[0]) for r in rows])


def test_event_projection_matches_golden(salt: dict) -> None:
    f_minus = _vec(salt["f_minus"])
    grad_g = _vec(salt["grad_g"])
    expected = np.array([[_num(x) for x in r] for r in salt["event_projection"]["Pi"]])
    pi = event_projection(grad_g, f_minus)
    np.testing.assert_allclose(pi, expected, atol=1e-14)
    np.testing.assert_allclose(pi @ f_minus, 0.0, atol=1e-14)
    np.testing.assert_allclose(grad_g @ pi, 0.0, atol=1e-14)
    assert not np.allclose(pi, np.eye(2))


def test_saltation_matches_golden_cases(salt: dict) -> None:
    f_minus = _vec(salt["f_minus"])
    grad_g = _vec(salt["grad_g"])
    for case in salt["cases"]:
        r_x = np.array([[_num(x) for x in r] for r in case["reset_jacobian"]])
        s_expected = np.array([[_num(x) for x in r] for r in case["S"]])
        f_plus = _vec(case["f_plus"])
        s_mat = saltation_matrix(r_x, f_minus, f_plus, grad_g)
        np.testing.assert_allclose(s_mat, s_expected, atol=1e-14)
        np.testing.assert_allclose(s_mat @ f_minus, f_plus, atol=1e-14)


def test_timing_row_is_the_event_time_shift(salt: dict) -> None:
    f_minus = _vec(salt["f_minus"])
    grad_g = _vec(salt["grad_g"])
    row = event_time_shift_row(grad_g, f_minus)
    dx = np.array([0.3, -0.2])
    # delta t* = -(g_x dx)/(g_x f)
    assert row @ dx == pytest.approx(-(grad_g @ dx) / (grad_g @ f_minus))
    # Pi = I + f * row (the timing term is what distinguishes it from I).
    np.testing.assert_allclose(
        event_projection(grad_g, f_minus), np.eye(2) + np.outer(f_minus, row)
    )


def test_explicit_time_dependence_enters_the_denominator() -> None:
    grad_g = np.array([1.0, 0.0])
    f = np.array([2.0, 1.0])
    row = event_time_shift_row(grad_g, f, g_t=-0.5)
    np.testing.assert_allclose(row, -grad_g / (2.0 - 0.5))


def test_non_transversal_event_is_rejected() -> None:
    with pytest.raises(ValueError, match="transversal"):
        event_time_shift_row(np.array([1.0, 0.0]), np.array([0.0, 3.0]))
    with pytest.raises(ValueError, match="shape"):
        event_projection(np.ones(3), np.ones(2))


def test_transversality_flags_grazing() -> None:
    g = np.array([1.0, 0.0])
    assert transversality(g, np.array([5.0, 0.0])) == pytest.approx(1.0)
    assert transversality(g, np.array([1.0, 40.0])) < GRAZING_TRANSVERSALITY
