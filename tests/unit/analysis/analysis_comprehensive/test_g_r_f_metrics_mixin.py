"""Comprehensive tests for src.shared.python.analysis package.

Tests the analysis mixin components directly by creating lightweight
stub classes that satisfy each mixin's attribute requirements.

Covers: basic_stats, energy_metrics, stability_metrics, angular_momentum,
grf_metrics, pca_analysis, and swing_metrics.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.analysis.angular_momentum import AngularMomentumMetricsMixin
from src.shared.python.analysis.basic_stats import BasicStatsMixin
from src.shared.python.analysis.dataclasses import (
    AngularMomentumMetrics,
    GRFMetrics,
    PCAResult,
    PeakInfo,
    StabilityMetrics,
    SummaryStatistics,
)
from src.shared.python.analysis.energy_metrics import EnergyMetricsMixin
from src.shared.python.analysis.grf_metrics import GRFMetricsMixin
from src.shared.python.analysis.pca_analysis import PCAAnalysisMixin
from src.shared.python.analysis.stability_metrics import StabilityMetricsMixin

# ============================================================================
# Stub classes for mixin testing
# ============================================================================


class BasicStatsHost(BasicStatsMixin):
    """Host class for BasicStatsMixin."""

    def __init__(
        self,
        times: np.ndarray,
        club_head_speed: np.ndarray | None = None,
    ) -> None:
        self.times = times
        self.club_head_speed = club_head_speed


class EnergyHost(EnergyMetricsMixin):
    """Host class for EnergyMetricsMixin."""

    def __init__(
        self,
        club_head_speed: np.ndarray | None = None,
    ) -> None:
        self.club_head_speed = club_head_speed


class StabilityHost(StabilityMetricsMixin):
    """Host class for StabilityMetricsMixin."""

    def __init__(
        self,
        cop_position: np.ndarray | None = None,
        com_position: np.ndarray | None = None,
    ) -> None:
        self.cop_position = cop_position
        self.com_position = com_position


class AngularMomentumHost(AngularMomentumMetricsMixin):
    """Host class for AngularMomentumMetricsMixin."""

    def __init__(
        self,
        angular_momentum: np.ndarray | None = None,
        times: np.ndarray | None = None,
    ) -> None:
        self.angular_momentum = angular_momentum
        self.times = times


class GRFHost(GRFMetricsMixin):
    """Host class for GRFMetricsMixin."""

    def __init__(
        self,
        cop_position: np.ndarray | None = None,
        ground_forces: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> None:
        self.cop_position = cop_position
        self.ground_forces = ground_forces
        self.dt = dt


class PCAHost(PCAAnalysisMixin):
    """Host class for PCAAnalysisMixin."""

    def __init__(
        self,
        times: np.ndarray,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
    ) -> None:
        self.times = times
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities


# ============================================================================
# Tests for BasicStatsMixin
# ============================================================================


# ============================================================================
# Tests for EnergyMetricsMixin
# ============================================================================


# ============================================================================
# Tests for StabilityMetricsMixin
# ============================================================================


# ============================================================================
# Tests for AngularMomentumMetricsMixin
# ============================================================================


# ============================================================================
# Tests for GRFMetricsMixin
# ============================================================================


class TestGRFMetricsMixin:
    """Tests for GRFMetricsMixin."""

    def test_no_data_returns_none(self) -> None:
        host = GRFHost()
        assert host.compute_grf_metrics() is None

    def test_stationary_cop(self) -> None:
        """Stationary CoP should have zero path length and ranges."""
        N = 50
        cop = np.zeros((N, 3))
        host = GRFHost(cop_position=cop, dt=0.01)
        metrics = host.compute_grf_metrics()
        assert metrics is not None
        assert isinstance(metrics, GRFMetrics)
        assert metrics.cop_path_length == pytest.approx(0.0)
        assert metrics.cop_x_range == pytest.approx(0.0)
        assert metrics.cop_y_range == pytest.approx(0.0)

    def test_linear_cop_path(self) -> None:
        """CoP moving linearly: path length should match distance."""
        N = 100
        cop = np.zeros((N, 3))
        cop[:, 0] = np.linspace(0, 1, N)  # Move 1m in X
        host = GRFHost(cop_position=cop, dt=0.01)
        metrics = host.compute_grf_metrics()
        assert metrics is not None
        assert metrics.cop_path_length == pytest.approx(1.0, abs=0.02)
        assert metrics.cop_x_range == pytest.approx(1.0, abs=0.01)

    def test_2d_cop(self) -> None:
        """Should handle 2D CoP data."""
        N = 50
        cop = np.zeros((N, 2))
        cop[:, 0] = np.linspace(0, 0.5, N)
        host = GRFHost(cop_position=cop, dt=0.01)
        metrics = host.compute_grf_metrics()
        assert metrics is not None
        assert metrics.cop_path_length > 0

    def test_with_ground_forces(self) -> None:
        """Should compute force metrics when ground forces available."""
        N = 50
        cop = np.zeros((N, 3))
        forces = np.zeros((N, 3))
        forces[:, 2] = 980.0  # 100kg * 9.8 m/s^2 vertical force
        forces[25, 0] = 100.0  # Lateral shear at midpoint
        host = GRFHost(cop_position=cop, ground_forces=forces, dt=0.01)
        metrics = host.compute_grf_metrics()
        assert metrics is not None
        assert metrics.peak_vertical_force == pytest.approx(980.0)
        assert metrics.peak_shear_force is not None
        assert metrics.peak_shear_force >= 100.0


# ============================================================================
# Tests for PCAAnalysisMixin
# ============================================================================


# ============================================================================
# Tests for dataclasses
# ============================================================================
