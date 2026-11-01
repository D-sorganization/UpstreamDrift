"""Harness math on hand-made curves, plus a check of the existing logistic."""

from __future__ import annotations

import math

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.putting_validation.harness import (
    compare_curves,
    compare_to_series,
)
from src.shared.python.putting_validation.reference_data import Quantity, get_series
from src.shared.python.sg_optimizer.shot_model.putting import _baseline_make

pytestmark = pytest.mark.unit

DISTANCES = (3.0, 10.0, 20.0)
REFERENCE = (0.90, 0.40, 0.15)


def test_exact_residuals_and_worst_distance() -> None:
    sim = (0.92, 0.35, 0.16)
    result = compare_curves(sim, DISTANCES, REFERENCE, tolerance=0.10)
    assert result.residuals == pytest.approx((0.02, -0.05, 0.01))
    assert result.max_abs_residual == pytest.approx(0.05)
    assert result.worst_distance_ft == 10.0
    assert result.passed


def test_tolerance_edge_is_inclusive_and_just_outside_fails() -> None:
    sim, ref = (0.75, 0.25), (0.5, 0.25)  # exact in binary: residual 0.25
    assert compare_curves(sim, (3.0, 9.0), ref, tolerance=0.25).passed
    assert not compare_curves(sim, (3.0, 9.0), ref, tolerance=0.2499).passed


def test_expected_putts_quantity() -> None:
    result = compare_curves(
        (1.05, 1.60), (3.0, 10.0), (1.00, 1.50), 0.2, Quantity.EXPECTED_PUTTS
    )
    assert result.max_abs_residual == pytest.approx(0.10)
    with pytest.raises(ContractViolationError):
        compare_curves((0.5,), (3.0,), (1.0,), 0.2, Quantity.EXPECTED_PUTTS)


@pytest.mark.parametrize(
    ("sim", "dist", "ref", "tol"),
    [
        ((0.9,), (), (), 0.1),
        ((0.9, 0.8), (10.0, 3.0), (0.9, 0.8), 0.1),
        ((0.9,), (0.0,), (0.9,), 0.1),
        ((0.9,), (3.0,), (0.9, 0.8), 0.1),
        ((0.9,), (3.0,), (0.9,), 0.0),
        ((0.9,), (3.0,), (0.9,), math.nan),
        ((1.2,), (3.0,), (0.9,), 0.1),
    ],
)
def test_contract_errors(sim, dist, ref, tol) -> None:
    with pytest.raises(ContractViolationError):
        compare_curves(sim, dist, ref, tol)


def test_compare_to_series_uses_series_distances() -> None:
    series = get_series("sg_optimizer_documented_anchors")
    result = compare_to_series(series.values, series, tolerance=0.01)
    assert result.distances_ft == series.distances_ft
    assert result.max_abs_residual == 0.0


@pytest.mark.xfail(
    strict=True,
    reason="#14176: putting.py anchors (~99% at 3 ft, ~7% at 30 ft) disagree with "
    "its logistic (94.6%, 2.4%); resolving needs Broadie table A.5.",
)
def test_existing_logistic_matches_its_documented_anchors() -> None:
    """Self-consistency of the existing fit's comment; not a simulator calibration."""
    series = get_series("sg_optimizer_documented_anchors")
    result = compare_to_series(_baseline_make, series, tolerance=0.02)
    assert result.passed, result
