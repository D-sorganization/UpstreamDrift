"""Tests for src.shared.python.perturbation.statistics (Issues #1949, #1744)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from src.shared.python.perturbation.statistics import (
    MetricStatistics,
    compute_metric_statistics,
)

# ---------------------------------------------------------------------------
# MetricStatistics dataclass
# ---------------------------------------------------------------------------


class TestMetricStatistics:
    def _make(self) -> MetricStatistics:
        return MetricStatistics(
            mean=2.0,
            std=1.0,
            cv=0.5,
            min_val=0.0,
            max_val=4.0,
            median=2.0,
            iqr=1.5,
            p5=0.5,
            p95=3.5,
        )

    def test_statistics_to_dict_returns_dict(self) -> None:
        assert isinstance(self._make().to_dict(), dict)

    def test_to_dict_has_all_keys(self) -> None:
        d = self._make().to_dict()
        assert "mean" in d
        assert "std" in d
        assert "cv" in d

    def test_to_dict_scalar_float(self) -> None:
        d = self._make().to_dict()
        assert isinstance(d["mean"], (int, float))

    def test_to_dict_array_becomes_list(self) -> None:
        ms = MetricStatistics(
            mean=np.array([1.0, 2.0]),
            std=np.array([0.1, 0.2]),
            cv=np.array([0.1, 0.1]),
            min_val=np.array([0.0, 1.0]),
            max_val=np.array([2.0, 3.0]),
            median=np.array([1.0, 2.0]),
            iqr=np.array([0.5, 0.5]),
            p5=np.array([0.1, 0.1]),
            p95=np.array([1.9, 2.9]),
        )
        d = ms.to_dict()
        assert isinstance(d["mean"], list)


# ---------------------------------------------------------------------------
# compute_metric_statistics
# ---------------------------------------------------------------------------


class TestComputeMetricStatistics:
    def test_basic_1d(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_metric_statistics(values)
        assert isinstance(stats, MetricStatistics)

    def test_mean_correct(self) -> None:
        values = np.array([2.0, 4.0])
        stats = compute_metric_statistics(values)
        assert abs(stats.mean - 3.0) < 1e-10

    def test_min_max_correct(self) -> None:
        values = np.array([1.0, 5.0, 3.0])
        stats = compute_metric_statistics(values)
        assert stats.min_val == 1.0
        assert stats.max_val == 5.0

    def test_median_correct(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        stats = compute_metric_statistics(values)
        assert abs(stats.median - 2.0) < 1e-10

    def test_std_correct(self) -> None:
        values = np.array([0.0, 2.0])  # std = 1.0 (population std)
        stats = compute_metric_statistics(values)
        assert abs(stats.std - 1.0) < 1e-10

    @pytest.mark.unit
    def test_cv_zero_mean_is_nan_with_reason(self) -> None:
        # NSX-82: a near-zero mean makes cv undefined (NaN), never 0
        values = np.array([0.0, 0.0, 0.0])
        stats = compute_metric_statistics(values)
        assert np.isnan(stats.cv)
        assert stats.cv_reason is not None
        assert "undefined" in stats.cv_reason

    @pytest.mark.unit
    def test_symmetric_wide_distribution_does_not_report_cv_zero(self) -> None:
        values = np.array([-90.0, 90.0, -90.0, 90.0])  # mean exactly 0, SD 90
        stats = compute_metric_statistics(values)
        assert stats.std == pytest.approx(90.0)
        assert np.isnan(stats.cv)

    @pytest.mark.unit
    def test_cv_defined_has_no_reason(self) -> None:
        stats = compute_metric_statistics(np.array([9.0, 11.0]))
        assert stats.cv == pytest.approx(0.1)
        assert stats.cv_reason is None

    @pytest.mark.unit
    def test_cv_vector_masks_only_near_zero_columns(self) -> None:
        values = np.array([[9.0, -1.0], [11.0, 1.0]])
        stats = compute_metric_statistics(values)
        assert stats.cv[0] == pytest.approx(0.1)
        assert np.isnan(stats.cv[1])
        assert stats.cv_reason is not None

    @pytest.mark.unit
    def test_to_dict_serialises_undefined_cv_as_null(self) -> None:
        d = compute_metric_statistics(np.zeros(3)).to_dict()
        assert d["cv"] is None
        assert d["cv_reason"] is not None
        json.dumps(d, allow_nan=False)

    def test_constant_array_std_zero(self) -> None:
        values = np.array([5.0, 5.0, 5.0])
        stats = compute_metric_statistics(values)
        assert stats.std == 0.0

    def test_single_value(self) -> None:
        values = np.array([7.0])
        stats = compute_metric_statistics(values)
        assert stats.mean == 7.0

    def test_statistics_empty_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            compute_metric_statistics(np.array([]))

    def test_p5_less_than_p95(self) -> None:
        values = np.linspace(0, 10, 100)
        stats = compute_metric_statistics(values)
        assert stats.p5 < stats.p95

    def test_to_dict_round_trip(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        stats = compute_metric_statistics(values)
        d = stats.to_dict()
        assert isinstance(d, dict)
        assert d["mean"] == stats.mean
