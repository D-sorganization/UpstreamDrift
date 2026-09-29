# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for result analysis metrics."""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.result_analysis import ResultAnalyzer
from movement_optimizer.tests.conftest import make_test_result


def test_torque_statistics_include_mean_std_and_range() -> None:
    result = make_test_result()
    result.torques = np.array(
        [
            [-2.0, 1.0, 10.0],
            [0.0, 3.0, -20.0],
            [2.0, 5.0, 30.0],
        ]
    )

    assert result.torque_joint_count() == 3
    stats = ResultAnalyzer(result).torque_statistics()

    assert stats[0].joint == "Ankle"
    assert stats[0].mean_nm == pytest.approx(0.0)
    assert stats[0].std_nm == pytest.approx(np.std([-2.0, 0.0, 2.0]))
    assert stats[0].min_nm == -2.0
    assert stats[0].max_nm == 2.0
    assert stats[2].peak_abs_nm == 30.0


def test_recommendations_include_default_message_for_clean_result() -> None:
    result = make_test_result()

    recommendations = ResultAnalyzer(result).recommendations()

    assert recommendations == ["No immediate issues detected in the exported result."]


def test_recommendations_flag_nonconvergence_joint_limits_and_com_range() -> None:
    result = make_test_result()
    result.success = False
    result.n_joint_limit_violations = 2
    result.com_horizontal_range_cm = 20.0

    recommendations = ResultAnalyzer(result).recommendations()

    assert any("did not converge" in rec for rec in recommendations)
    assert any("joint limits" in rec for rec in recommendations)
    assert any("COM travel" in rec for rec in recommendations)


def test_analyzer_rejects_non_matrix_torques() -> None:
    result = make_test_result()
    result.torques = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="torques"):
        ResultAnalyzer(result)


def test_result_reports_empty_com_samples() -> None:
    result = make_test_result()
    result.com = np.zeros((0, 2))

    assert not result.has_com_samples()
    assert ResultAnalyzer(result).com_range_cm() == 0.0
