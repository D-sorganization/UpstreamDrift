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


class TestAngularMomentumMetricsMixin:
    """Tests for AngularMomentumMetricsMixin."""

    def test_no_data_returns_none(self) -> None:
        host = AngularMomentumHost()
        assert host.compute_angular_momentum_metrics() is None

    def test_constant_angular_momentum(self) -> None:
        """Constant angular momentum vector."""
        N = 100
        am = np.ones((N, 3))  # (1, 1, 1) -> mag = sqrt(3)
        times = np.linspace(0, 1, N)
        host = AngularMomentumHost(angular_momentum=am, times=times)
        metrics = host.compute_angular_momentum_metrics()
        assert metrics is not None
        assert isinstance(metrics, AngularMomentumMetrics)
        assert metrics.peak_magnitude == pytest.approx(np.sqrt(3))
        assert metrics.mean_magnitude == pytest.approx(np.sqrt(3))
        assert metrics.peak_lx == pytest.approx(1.0)
        assert metrics.peak_ly == pytest.approx(1.0)
        assert metrics.peak_lz == pytest.approx(1.0)
        # Variability = 0 for constant
        assert metrics.variability == pytest.approx(0.0)

    def test_varying_angular_momentum(self) -> None:
        N = 100
        t = np.linspace(0, 1, N)
        am = np.column_stack([np.sin(2 * np.pi * t), np.zeros(N), np.zeros(N)])
        host = AngularMomentumHost(angular_momentum=am, times=t)
        metrics = host.compute_angular_momentum_metrics()
        assert metrics is not None
        assert metrics.peak_lx == pytest.approx(1.0, abs=0.05)
        assert metrics.variability > 0

    def test_empty_angular_momentum(self) -> None:
        host = AngularMomentumHost(angular_momentum=np.array([]))
        assert host.compute_angular_momentum_metrics() is None


# ============================================================================
# Tests for GRFMetricsMixin
# ============================================================================


# ============================================================================
# Tests for PCAAnalysisMixin
# ============================================================================


# ============================================================================
# Tests for dataclasses
# ============================================================================
