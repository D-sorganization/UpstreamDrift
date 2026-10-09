"""Grip-closure residual series: unavailable is never zero (OSV-2, #11728)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    GripClosureSeries,
    closure_series_from_residuals,
)

pytestmark = pytest.mark.unit


def test_series_reports_max_and_tolerance() -> None:
    series = GripClosureSeries("x", np.array([0.001, 0.004, 0.002]))
    assert series.max_m == pytest.approx(0.004)
    assert series.within(0.005) is True
    assert series.within(0.003) is False


def test_unavailable_series_is_none_not_zero() -> None:
    series = GripClosureSeries.unavailable("myosuite", "no plant")
    assert series.available is False
    assert series.max_m is None
    assert series.within(0.005) is None
    assert series.as_document()["max_m"] is None
    assert series.as_document()["reason"] == "no plant"


def test_unavailable_needs_a_reason_and_data_must_be_finite() -> None:
    with pytest.raises(ValueError):
        GripClosureSeries("x", None)
    with pytest.raises(ValueError):
        GripClosureSeries("x", np.array([0.0, np.nan]))
    with pytest.raises(ValueError):
        GripClosureSeries("x", np.array([]))


def test_tolerance_must_be_positive() -> None:
    with pytest.raises(ValueError):
        GripClosureSeries("x", np.array([0.0])).within(0.0)


def test_from_residuals_takes_translation_norm_per_frame() -> None:
    q = np.zeros((3, 2))

    def residuals(row: np.ndarray) -> np.ndarray:
        return np.array([0.003, 0.004, 0.0, 9.0, 9.0, 9.0])  # rotation ignored

    series = closure_series_from_residuals("e", residuals, q)
    np.testing.assert_allclose(series.residual_m, [0.005] * 3)


def test_engine_without_closure_is_unavailable() -> None:
    def residuals(row: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Native grip closure is not qualified")

    series = closure_series_from_residuals("opensim", residuals, np.zeros((2, 1)))
    assert series.available is False
    assert "not qualified" in str(series.reason)


def test_non_finite_residual_is_unavailable() -> None:
    series = closure_series_from_residuals(
        "e", lambda row: np.array([np.nan, 0.0, 0.0]), np.zeros((2, 1))
    )
    assert series.available is False and series.max_m is None
