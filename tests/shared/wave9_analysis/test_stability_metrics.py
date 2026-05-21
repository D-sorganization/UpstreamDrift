"""Tests for src/shared/python/analysis/stability_metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.analysis.stability_metrics import StabilityMetricsMixin


class _Holder(StabilityMetricsMixin):
    def __init__(
        self,
        cop_position: np.ndarray | None = None,
        com_position: np.ndarray | None = None,
    ) -> None:
        self.cop_position = cop_position
        self.com_position = com_position


class TestComputeStabilityMetrics:
    def test_none_when_missing_cop(self) -> None:
        h = _Holder(cop_position=None, com_position=np.zeros((3, 3)))
        assert h.compute_stability_metrics() is None

    def test_none_when_missing_com(self) -> None:
        h = _Holder(cop_position=np.zeros((3, 3)), com_position=None)
        assert h.compute_stability_metrics() is None

    def test_none_when_length_mismatch(self) -> None:
        h = _Holder(
            cop_position=np.zeros((3, 3)),
            com_position=np.zeros((4, 3)),
        )
        assert h.compute_stability_metrics() is None

    def test_perfect_vertical_alignment(self) -> None:
        # CoM directly above CoP: zero distance, zero inclination
        cop = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        com = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        h = _Holder(cop_position=cop, com_position=com)
        result = h.compute_stability_metrics()
        assert result is not None
        assert result.min_com_cop_distance == 0.0
        assert result.max_com_cop_distance == 0.0
        assert result.mean_com_cop_distance == 0.0
        assert result.peak_inclination_angle == pytest.approx(0.0)
        assert result.mean_inclination_angle == pytest.approx(0.0)

    def test_45_degree_lean(self) -> None:
        # CoM offset by (1, 0, 1) above CoP at origin
        cop = np.array([[0.0, 0.0, 0.0]])
        com = np.array([[1.0, 0.0, 1.0]])
        h = _Holder(cop_position=cop, com_position=com)
        result = h.compute_stability_metrics()
        assert result is not None
        assert result.min_com_cop_distance == pytest.approx(1.0)
        assert result.peak_inclination_angle == pytest.approx(45.0, abs=0.01)

    def test_2d_cop(self) -> None:
        # 2D CoP -> z is assumed 0
        cop = np.array([[0.0, 0.0], [1.0, 0.0]])
        com = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        h = _Holder(cop_position=cop, com_position=com)
        result = h.compute_stability_metrics()
        assert result is not None
        assert result.peak_inclination_angle == pytest.approx(0.0, abs=0.01)

    def test_angles_in_valid_range(self) -> None:
        rng = np.random.default_rng(42)
        cop = rng.standard_normal((10, 3))
        com = rng.standard_normal((10, 3))
        h = _Holder(cop_position=cop, com_position=com)
        result = h.compute_stability_metrics()
        assert result is not None
        assert 0 <= result.peak_inclination_angle <= 180.0
        assert 0 <= result.mean_inclination_angle <= 180.0
        assert result.min_com_cop_distance >= 0
        assert result.max_com_cop_distance >= result.min_com_cop_distance
