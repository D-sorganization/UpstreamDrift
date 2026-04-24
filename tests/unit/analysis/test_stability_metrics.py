"""Tests for src.shared.python.analysis.stability_metrics (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.analysis.stability_metrics import StabilityMetricsMixin


class _Concrete(StabilityMetricsMixin):
    """Minimal concrete subclass for testing StabilityMetricsMixin."""

    def __init__(self) -> None:
        self.cop_position = None
        self.com_position = None


class TestComputeStabilityMetrics:
    def setup_method(self) -> None:
        self.obj = _Concrete()

    def test_returns_none_when_no_data(self) -> None:
        result = self.obj.compute_stability_metrics()
        assert result is None

    def test_returns_none_when_cop_none(self) -> None:
        self.obj.com_position = np.zeros((10, 3))
        result = self.obj.compute_stability_metrics()
        assert result is None

    def test_returns_none_when_com_none(self) -> None:
        self.obj.cop_position = np.zeros((10, 3))
        result = self.obj.compute_stability_metrics()
        assert result is None

    def test_returns_none_when_mismatched_lengths(self) -> None:
        self.obj.cop_position = np.zeros((10, 3))
        self.obj.com_position = np.zeros((5, 3))
        result = self.obj.compute_stability_metrics()
        assert result is None

    def _make_stable_data(self, n: int = 50) -> None:
        """Set up CoP directly below CoM — zero horizontal displacement."""
        com = np.zeros((n, 3))
        com[:, 2] = 1.0  # CoM at height 1.0
        cop = np.zeros((n, 3))  # CoP at origin (directly below)
        self.obj.cop_position = cop
        self.obj.com_position = com

    def test_returns_stability_metrics_object(self) -> None:
        self._make_stable_data()
        result = self.obj.compute_stability_metrics()
        assert result is not None

    def test_distances_non_negative(self) -> None:
        self._make_stable_data()
        result = self.obj.compute_stability_metrics()
        assert result.min_com_cop_distance >= 0.0
        assert result.max_com_cop_distance >= 0.0
        assert result.mean_com_cop_distance >= 0.0

    def test_angles_in_range(self) -> None:
        self._make_stable_data()
        result = self.obj.compute_stability_metrics()
        assert 0.0 <= result.peak_inclination_angle <= 180.0
        assert 0.0 <= result.mean_inclination_angle <= 180.0

    def test_zero_horizontal_offset_zero_distance(self) -> None:
        self._make_stable_data()
        result = self.obj.compute_stability_metrics()
        assert result.min_com_cop_distance == pytest.approx(0.0, abs=1e-10)

    def test_nonzero_offset_positive_distance(self) -> None:
        n = 50
        com = np.zeros((n, 3))
        com[:, 0] = 0.1  # offset in X
        com[:, 2] = 1.0
        cop = np.zeros((n, 3))
        self.obj.cop_position = cop
        self.obj.com_position = com
        result = self.obj.compute_stability_metrics()
        assert result.min_com_cop_distance > 0.0

    def test_max_distance_gte_min_distance(self) -> None:
        n = 50
        com = np.zeros((n, 3))
        com[:, 0] = np.linspace(0, 0.2, n)  # varying offset
        com[:, 2] = 1.0
        cop = np.zeros((n, 3))
        self.obj.cop_position = cop
        self.obj.com_position = com
        result = self.obj.compute_stability_metrics()
        assert result.max_com_cop_distance >= result.min_com_cop_distance

    def test_2d_cop_handled(self) -> None:
        n = 20
        self.obj.cop_position = np.zeros((n, 2))  # 2D CoP
        self.obj.com_position = np.zeros((n, 3))
        self.obj.com_position[:, 2] = 1.0
        result = self.obj.compute_stability_metrics()
        assert result is not None
        assert result.min_com_cop_distance >= 0.0
