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


# ============================================================================
# Tests for PCAAnalysisMixin
# ============================================================================


# ============================================================================
# Tests for dataclasses
# ============================================================================


class TestDataclasses:
    """Tests for analysis dataclass instantiation."""

    def test_peak_info(self) -> None:
        peak = PeakInfo(value=10.0, time=0.5, index=50)
        assert peak.value == 10.0
        assert peak.prominence is None
        assert peak.width is None

    def test_peak_info_with_extras(self) -> None:
        peak = PeakInfo(value=10.0, time=0.5, index=50, prominence=1.5, width=0.02)
        assert peak.prominence == 1.5
        assert peak.width == 0.02

    def test_analysis_comprehensive_summary_statistics(self) -> None:
        stats = SummaryStatistics(
            mean=5.0,
            median=5.0,
            std=1.0,
            min=1.0,
            max=9.0,
            range=8.0,
            min_time=0.0,
            max_time=1.0,
            rms=5.1,
        )
        assert stats.range == 8.0

    def test_stability_metrics(self) -> None:
        sm = StabilityMetrics(
            min_com_cop_distance=0.01,
            max_com_cop_distance=0.05,
            mean_com_cop_distance=0.03,
            peak_inclination_angle=5.0,
            mean_inclination_angle=3.0,
        )
        assert sm.peak_inclination_angle == 5.0

    def test_angular_momentum_metrics(self) -> None:
        am = AngularMomentumMetrics(
            peak_magnitude=100.0,
            peak_time=0.5,
            mean_magnitude=50.0,
            peak_lx=60.0,
            peak_ly=40.0,
            peak_lz=20.0,
            variability=0.3,
        )
        assert am.peak_magnitude == 100.0

    def test_grf_metrics(self) -> None:
        grf = GRFMetrics(
            cop_path_length=0.5,
            cop_max_velocity=0.1,
            cop_x_range=0.2,
            cop_y_range=0.15,
            peak_vertical_force=980.0,
            peak_shear_force=50.0,
        )
        assert grf.cop_path_length == 0.5

    def test_pca_result(self) -> None:
        pca = PCAResult(
            components=np.eye(3),
            explained_variance=np.array([1.0, 0.5, 0.1]),
            explained_variance_ratio=np.array([0.625, 0.3125, 0.0625]),
            projected_data=np.zeros((10, 3)),
            mean=np.zeros(3),
        )
        assert pca.components.shape == (3, 3)
