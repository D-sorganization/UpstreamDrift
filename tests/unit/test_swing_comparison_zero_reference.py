"""A zero reference value is not a perfect match (swing comparison scoring)."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest
from src.shared.python.biomechanics.swing_comparison import SwingComparator
from src.shared.python.validation_pkg.statistical_analysis import StatisticalAnalyzer

pytestmark = pytest.mark.unit


def _data(velocity_peak: float, club_speed: float | None = None) -> dict:
    n = 50
    vel = np.zeros((n, 2))
    vel[10, 0] = velocity_peak
    data: dict = {
        "times": np.linspace(0.0, 1.0, n),
        "joint_positions": np.zeros((n, 2)),
        "joint_velocities": vel,
        "joint_torques": np.zeros((n, 2)),
    }
    if club_speed is not None:
        data["club_head_speed"] = np.full(n, club_speed)
    return data


def test_peak_speed_against_zero_reference_is_not_a_perfect_score() -> None:
    """Reference never moves (0 rad/s), student peaks at 5 rad/s: no match."""
    comparator = SwingComparator(_data(0.0), _data(5.0))
    metric = comparator.compare_peak_speeds({"Joint1": 0})["Joint1"]
    assert metric.difference == pytest.approx(5.0)
    assert math.isnan(metric.percent_diff)
    assert metric.score == 0.0


def test_peak_speed_both_zero_is_a_perfect_score() -> None:
    comparator = SwingComparator(_data(0.0), _data(0.0))
    metric = comparator.compare_peak_speeds({"Joint1": 0})["Joint1"]
    assert metric.percent_diff == 0.0
    assert metric.score == 100.0


def test_peak_speed_nonzero_reference_keeps_closed_form_percent() -> None:
    """10 vs 12 rad/s is exactly +20 % and scores 100 - 20 = 80."""
    comparator = SwingComparator(_data(10.0), _data(12.0))
    metric = comparator.compare_peak_speeds({"Joint1": 0})["Joint1"]
    assert metric.percent_diff == pytest.approx(20.0)
    assert metric.score == pytest.approx(80.0)


def test_club_head_speed_against_zero_reference_is_not_a_perfect_score() -> None:
    comparator = SwingComparator(_data(1.0, 0.0), _data(1.0, 40.0))
    metric = comparator.compare_peak_speeds({"Joint1": 0})["Club Head"]
    assert math.isnan(metric.percent_diff)
    assert metric.score == 0.0


def test_tempo_against_zero_reference_ratio_is_not_a_perfect_score() -> None:
    with patch.object(
        StatisticalAnalyzer,
        "compute_tempo",
        side_effect=[(0.0, 0.0, 0.0), (0.0, 0.8, 3.0)],
    ):
        metric = SwingComparator(_data(1.0), _data(1.0)).compare_tempo()
    assert metric is not None
    assert metric.difference == pytest.approx(3.0)
    assert math.isnan(metric.percent_diff)
    assert metric.score == 0.0
