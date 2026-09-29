"""Tests for src.shared.python.analysis.angular_momentum (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.analysis.angular_momentum import AngularMomentumMetricsMixin


class _Concrete(AngularMomentumMetricsMixin):
    def __init__(self) -> None:
        self.angular_momentum = None
        self.times = None


class TestAngularMomentumMetrics:
    def setup_method(self) -> None:
        self.obj = _Concrete()

    def test_returns_none_when_no_data(self) -> None:
        assert self.obj.compute_angular_momentum_metrics() is None

    def test_returns_none_when_empty(self) -> None:
        self.obj.angular_momentum = np.zeros((0, 3))
        assert self.obj.compute_angular_momentum_metrics() is None

    def _set_data(self, n: int = 50) -> None:
        t = np.linspace(0.0, 1.0, n)
        L = np.zeros((n, 3))
        L[:, 0] = np.sin(2 * np.pi * t) * 2.0
        L[:, 1] = np.cos(2 * np.pi * t) * 1.0
        L[:, 2] = 0.5
        self.obj.times = t
        self.obj.angular_momentum = L

    def test_returns_metrics_object(self) -> None:
        self._set_data()
        result = self.obj.compute_angular_momentum_metrics()
        assert result is not None

    def test_peak_magnitude_positive(self) -> None:
        self._set_data()
        result = self.obj.compute_angular_momentum_metrics()
        assert result.peak_magnitude > 0.0

    def test_mean_magnitude_non_negative(self) -> None:
        self._set_data()
        result = self.obj.compute_angular_momentum_metrics()
        assert result.mean_magnitude >= 0.0

    def test_peak_magnitude_gte_mean(self) -> None:
        self._set_data()
        result = self.obj.compute_angular_momentum_metrics()
        assert result.peak_magnitude >= result.mean_magnitude

    def test_angular_momentum_variability_non_negative(self) -> None:
        self._set_data()
        result = self.obj.compute_angular_momentum_metrics()
        assert result.variability >= 0.0

    def test_component_peaks_non_negative(self) -> None:
        self._set_data()
        result = self.obj.compute_angular_momentum_metrics()
        assert result.peak_lx >= 0.0
        assert result.peak_ly >= 0.0
        assert result.peak_lz >= 0.0

    def test_constant_momentum_zero_variability(self) -> None:
        n = 50
        self.obj.times = np.linspace(0.0, 1.0, n)
        L = np.ones((n, 3))
        L[:, 0] = 1.0
        L[:, 1] = 0.0
        L[:, 2] = 0.0
        self.obj.angular_momentum = L
        result = self.obj.compute_angular_momentum_metrics()
        assert result.variability == pytest.approx(0.0, abs=1e-12)

    def test_peak_time_in_range(self) -> None:
        self._set_data()
        result = self.obj.compute_angular_momentum_metrics()
        assert self.obj.times[0] <= result.peak_time <= self.obj.times[-1]


class TestGRFMetrics:
    """Test GRFMetricsMixin from grf_metrics module."""

    def setup_method(self) -> None:
        from src.shared.python.analysis.grf_metrics import GRFMetricsMixin

        class _ConcreteGRF(GRFMetricsMixin):
            def __init__(self) -> None:
                self.cop_position = None
                self.ground_forces = None
                self.dt = 0.01

        self.obj = _ConcreteGRF()

    def test_returns_none_when_no_data(self) -> None:
        result = self.obj.compute_grf_metrics()
        assert result is None

    def _set_cop_data(self, n: int = 50) -> None:
        t = np.linspace(0.0, 1.0, n)
        cop = np.zeros((n, 2))
        cop[:, 0] = 0.05 * np.sin(2 * np.pi * t)
        cop[:, 1] = 0.03 * np.cos(2 * np.pi * t)
        self.obj.cop_position = cop

    def test_returns_grf_metrics(self) -> None:
        self._set_cop_data()
        result = self.obj.compute_grf_metrics()
        assert result is not None

    def test_path_length_non_negative(self) -> None:
        self._set_cop_data()
        result = self.obj.compute_grf_metrics()
        assert result.cop_path_length >= 0.0

    def test_velocity_non_negative(self) -> None:
        self._set_cop_data()
        result = self.obj.compute_grf_metrics()
        assert result.cop_max_velocity >= 0.0

    def test_ranges_non_negative(self) -> None:
        self._set_cop_data()
        result = self.obj.compute_grf_metrics()
        assert result.cop_x_range >= 0.0
        assert result.cop_y_range >= 0.0

    def test_stationary_cop_zero_path(self) -> None:
        n = 50
        cop = np.zeros((n, 2))  # static CoP
        self.obj.cop_position = cop
        result = self.obj.compute_grf_metrics()
        assert result.cop_path_length == pytest.approx(0.0, abs=1e-12)

    def test_force_metrics_with_ground_forces(self) -> None:
        n = 50
        cop = np.random.default_rng(0).standard_normal((n, 3)) * 0.05
        forces = np.zeros((n, 3))
        forces[:, 2] = 700.0 + 50.0 * np.sin(np.linspace(0, np.pi, n))
        self.obj.cop_position = cop
        self.obj.ground_forces = forces
        result = self.obj.compute_grf_metrics()
        assert result.peak_vertical_force is not None
        assert result.peak_vertical_force > 0.0
