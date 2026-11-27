"""compare_scalars percent difference edge cases (#15226).

Expected values use the symmetric percent difference
``100 * (a - b) / ((|a| + |b|) / 2)`` computed by hand.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from src.shared.python.validation_pkg.comparative_analysis import (
    ComparativeSwingAnalyzer,
    RecorderInterface,
)

pytestmark = pytest.mark.unit


class _Rec(RecorderInterface):
    def __init__(self) -> None:
        self.engine = None

    def get_time_series(self, field_name: str) -> tuple:
        return np.array([]), np.array([])

    def get_induced_acceleration_series(self, source_name: str | int) -> tuple:
        return np.array([]), np.array([])

    def set_analysis_config(self, config: dict) -> None:  # noqa: ANN001
        pass


@pytest.fixture
def analyzer() -> ComparativeSwingAnalyzer:
    return ComparativeSwingAnalyzer(_Rec(), _Rec())


def test_opposite_signs_are_not_a_perfect_match(analyzer) -> None:  # noqa: ANN001
    metric = analyzer.compare_scalars("m", 5.0, -5.0)
    assert metric.difference == 10.0
    # 10 / ((5 + 5) / 2) * 100
    assert metric.percent_diff == pytest.approx(200.0)


def test_tiny_opposite_signs_are_not_a_perfect_match(analyzer) -> None:  # noqa: ANN001
    metric = analyzer.compare_scalars("m", 1e-10, -1e-10)
    assert metric.percent_diff == pytest.approx(200.0)


def test_tiny_value_against_zero_is_not_a_match(analyzer) -> None:  # noqa: ANN001
    # 1e-10 / (1e-10 / 2) * 100 = 200
    assert analyzer.compare_scalars("m", 1e-10, 0.0).percent_diff == pytest.approx(
        200.0
    )


def test_both_zero_is_a_match(analyzer) -> None:  # noqa: ANN001
    assert analyzer.compare_scalars("m", 0.0, 0.0).percent_diff == 0.0


def test_positive_values_unchanged(analyzer) -> None:  # noqa: ANN001
    assert analyzer.compare_scalars("m", 10.0, 8.0).percent_diff == pytest.approx(
        2.0 / 9.0 * 100
    )


def test_negative_values_sign_follows_difference(analyzer) -> None:  # noqa: ANN001
    # a - b = +2 so A is larger; 2 / ((4 + 6) / 2) * 100 = 40
    metric = analyzer.compare_scalars("m", -4.0, -6.0)
    assert metric.difference == 2.0
    assert metric.percent_diff == pytest.approx(40.0)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1.6e308, 1.5e308, 100.0 / 15.5),
        (1.6e308, -1.5e308, 200.0),
        (-1.6e308, 1.5e308, -200.0),
        (math.ulp(0.0), 0.0, 200.0),
        (0.0, math.ulp(0.0), -200.0),
        (math.ulp(0.0), -math.ulp(0.0), 200.0),
        (math.ulp(0.0), math.ulp(0.0), 0.0),
        (4.824193288964698e-36, -5.81075688469156e-36, 200.0),
    ],
)
def test_finite_extremes_preserve_symmetric_percent(
    analyzer, a: float, b: float, expected: float
) -> None:  # noqa: ANN001
    actual = analyzer.compare_scalars("m", a, b).percent_diff
    assert actual == pytest.approx(expected)
    assert -200.0 <= actual <= 200.0


def test_close_large_values_preserve_subtraction_accuracy(analyzer) -> None:  # noqa: ANN001
    a = 1.6e308
    b = math.nextafter(a, 0.0)
    exact_a, exact_b = Fraction(a), Fraction(b)
    expected = float(200 * (exact_a - exact_b) / (abs(exact_a) + abs(exact_b)))
    actual = analyzer.compare_scalars("m", a, b).percent_diff
    assert actual == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    "a,b", [(math.nan, 1.0), (1.0, math.nan), (math.nan, math.nan)]
)
def test_nan_is_not_a_match(analyzer, a: float, b: float) -> None:  # noqa: ANN001
    assert math.isnan(analyzer.compare_scalars("m", a, b).percent_diff)
