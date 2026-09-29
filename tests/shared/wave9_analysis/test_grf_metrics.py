"""Tests for src/shared/python/analysis/grf_metrics.py."""

from __future__ import annotations

import numpy as np

from src.shared.python.analysis.dataclasses import GRFMetrics
from src.shared.python.analysis.grf_metrics import GRFMetricsMixin


class _Holder(GRFMetricsMixin):
    def __init__(
        self,
        cop_position: np.ndarray | None = None,
        ground_forces: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> None:
        self.cop_position = cop_position
        self.ground_forces = ground_forces
        self.dt = dt


class TestComputeGRFMetrics:
    def test_none_when_no_cop(self) -> None:
        assert _Holder(cop_position=None).compute_grf_metrics() is None

    def test_none_when_empty_cop(self) -> None:
        assert _Holder(cop_position=np.zeros((0, 2))).compute_grf_metrics() is None

    def test_2d_cop_path_length(self) -> None:
        # 3 points forming an L: (0,0)->(1,0)->(1,1). Path = 2.0
        cop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        h = _Holder(cop_position=cop, dt=1.0)
        result = h.compute_grf_metrics()
        assert isinstance(result, GRFMetrics)
        assert result.cop_path_length == 2.0
        assert result.cop_max_velocity == 1.0  # max step / dt
        assert result.cop_x_range == 1.0
        assert result.cop_y_range == 1.0
        assert result.peak_vertical_force is None
        assert result.peak_shear_force is None

    def test_3d_cop_and_forces(self) -> None:
        cop = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        # Forces: [Fx, Fy, Fz]; Z is vertical
        forces = np.array([[0.0, 0.0, 100.0], [3.0, 4.0, 200.0], [0.0, 0.0, 150.0]])
        h = _Holder(cop_position=cop, ground_forces=forces, dt=0.5)
        result = h.compute_grf_metrics()
        assert result is not None
        assert result.cop_path_length == 2.0
        assert result.cop_max_velocity == 2.0  # 1.0 / 0.5
        assert result.peak_vertical_force == 200.0
        # Max shear = sqrt(3^2 + 4^2) = 5.0
        assert result.peak_shear_force == 5.0

    def test_zero_dt(self) -> None:
        cop = np.array([[0.0, 0.0], [1.0, 0.0]])
        h = _Holder(cop_position=cop, dt=0.0)
        result = h.compute_grf_metrics()
        assert result is not None
        assert result.cop_max_velocity == 0.0

    def test_postconditions_non_negative(self) -> None:
        cop = np.array([[5.0, 5.0], [3.0, 2.0], [-1.0, 4.0]])
        h = _Holder(cop_position=cop, dt=0.1)
        result = h.compute_grf_metrics()
        assert result is not None
        assert result.cop_path_length >= 0
        assert result.cop_x_range >= 0
        assert result.cop_y_range >= 0
