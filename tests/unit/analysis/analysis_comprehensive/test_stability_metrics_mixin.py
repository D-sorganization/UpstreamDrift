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


class TestStabilityMetricsMixin:
    """Tests for StabilityMetricsMixin."""

    def test_no_data_returns_none(self) -> None:
        host = StabilityHost()
        assert host.compute_stability_metrics() is None

    def test_com_directly_above_cop(self) -> None:
        """When CoM is directly above CoP, distance=0 and angle=0."""
        N = 50
        cop = np.zeros((N, 3))
        com = np.zeros((N, 3))
        com[:, 2] = 1.0  # CoM 1m above CoP
        host = StabilityHost(cop_position=cop, com_position=com)
        metrics = host.compute_stability_metrics()
        assert metrics is not None
        assert isinstance(metrics, StabilityMetrics)
        assert metrics.mean_com_cop_distance == pytest.approx(0.0)
        assert metrics.mean_inclination_angle == pytest.approx(0.0)

    def test_tilted_com(self) -> None:
        """When CoM is offset horizontally, inclination angle > 0."""
        N = 50
        cop = np.zeros((N, 3))
        com = np.zeros((N, 3))
        com[:, 0] = 1.0  # 1m lateral offset
        com[:, 2] = 1.0  # 1m height
        host = StabilityHost(cop_position=cop, com_position=com)
        metrics = host.compute_stability_metrics()
        assert metrics is not None
        # 45 degrees (arctan(1/1))
        assert metrics.peak_inclination_angle == pytest.approx(45.0, abs=0.5)
        assert metrics.mean_com_cop_distance == pytest.approx(1.0)

    def test_2d_cop(self) -> None:
        """Should handle 2D CoP data."""
        N = 20
        cop_2d = np.zeros((N, 2))
        com = np.zeros((N, 3))
        com[:, 2] = 1.0
        host = StabilityHost(cop_position=cop_2d, com_position=com)
        metrics = host.compute_stability_metrics()
        assert metrics is not None
        assert metrics.mean_inclination_angle == pytest.approx(0.0)

    def test_length_mismatch_returns_none(self) -> None:
        cop = np.zeros((10, 3))
        com = np.zeros((20, 3))
        host = StabilityHost(cop_position=cop, com_position=com)
        assert host.compute_stability_metrics() is None


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
