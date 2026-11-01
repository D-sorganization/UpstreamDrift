"""Expected-putts curve: closed-form check, contracts and solver failure modes."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.putting_analytics.expected_putts import (
    ExpectedPuttsCurve,
    PuttAction,
    PuttOutcomes,
    solve_expected_putts,
)

from .conftest import ConstantMakeModel

HABIT = PuttAction(aim_offset_deg=0.0, pace_ft_past=1.0)

pytestmark = pytest.mark.unit


def test_matches_closed_form_for_constant_make_model() -> None:
    curve = solve_expected_putts(
        ConstantMakeModel(0.6), [0.5, 1, 5, 10], lambda _d: HABIT, n=20_000, seed=0
    )
    # E(d > 1) = 1 + (1 - p) * E(0.5) with E(0.5) = 1, so 1.4.
    assert curve.values[:2] == pytest.approx([1.0, 1.0])
    assert curve(7.5) == pytest.approx(1.4, abs=0.02)


def test_never_holing_has_no_finite_expectation() -> None:
    with pytest.raises(ValueError, match="never"):
        solve_expected_putts(ConstantMakeModel(0.0), [2, 4], lambda _d: HABIT, 100, 0)


def test_curve_and_outcome_contracts() -> None:
    with pytest.raises(ValueError):
        ExpectedPuttsCurve(np.array([1.0, 1.0]), np.array([1.0, 1.2]))
    with pytest.raises(ValueError):
        ExpectedPuttsCurve(np.array([1.0, 2.0]), np.array([0.5, 1.2]))
    with pytest.raises(ValueError):
        PuttOutcomes(np.array([True, False]), np.array([0.0, -1.0]))
    with pytest.raises(ValueError):
        solve_expected_putts(ConstantMakeModel(0.5), [3], lambda _d: HABIT, 10, 0)
