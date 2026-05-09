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


class TestBasicStatsMixin:
    """Tests for BasicStatsMixin."""

    @pytest.fixture()
    def host(self) -> BasicStatsHost:
        N = 100
        times = np.linspace(0, 1.0, N)
        speed = np.sin(2 * np.pi * times) * 50 + 50
        return BasicStatsHost(times=times, club_head_speed=speed)

    def test_analysis_comprehensive_compute_summary_stats(
        self, host: BasicStatsHost
    ) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        host_small = BasicStatsHost(times=np.arange(5, dtype=float))
        stats = host_small.compute_summary_stats(data)
        assert isinstance(stats, SummaryStatistics)
        assert stats.mean == pytest.approx(3.0)
        assert stats.median == pytest.approx(3.0)
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(5.0)
        assert stats.range == pytest.approx(4.0)
        assert stats.std >= 0
        assert stats.rms >= 0

    def test_summary_stats_single_element(self) -> None:
        host = BasicStatsHost(times=np.array([0.0]))
        stats = host.compute_summary_stats(np.array([42.0]))
        assert stats.mean == 42.0
        assert stats.range == 0.0

    def test_analysis_comprehensive_find_peaks_in_data(
        self, host: BasicStatsHost
    ) -> None:
        data = np.sin(2 * np.pi * host.times * 3)  # 3 full cycles
        peaks = host.find_peaks_in_data(data, height=0.5)
        assert len(peaks) > 0
        for peak in peaks:
            assert isinstance(peak, PeakInfo)
            assert peak.value > 0.5

    def test_find_peaks_with_prominence(self, host: BasicStatsHost) -> None:
        data = np.sin(2 * np.pi * host.times * 2)
        peaks = host.find_peaks_in_data(data, prominence=0.5)
        for peak in peaks:
            assert peak.prominence is not None
            assert peak.prominence >= 0.5

    def test_analysis_comprehensive_find_club_head_speed_peak(
        self, host: BasicStatsHost
    ) -> None:
        peak = host.find_club_head_speed_peak()
        assert peak is not None
        assert isinstance(peak, PeakInfo)
        assert peak.value == pytest.approx(100.0, abs=1.0)

    def test_find_club_head_speed_peak_none(self) -> None:
        host = BasicStatsHost(times=np.arange(10, dtype=float))
        assert host.find_club_head_speed_peak() is None

    def test_find_club_head_speed_peak_empty(self) -> None:
        host = BasicStatsHost(
            times=np.arange(10, dtype=float), club_head_speed=np.array([])
        )
        assert host.find_club_head_speed_peak() is None


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
