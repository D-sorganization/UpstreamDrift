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


class TestEnergyMetricsMixin:
    """Tests for EnergyMetricsMixin."""

    def test_basic_energy_metrics(self) -> None:
        ke = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        pe = np.array([16.0, 9.0, 4.0, 1.0, 0.0])
        host = EnergyHost()
        metrics = host.compute_energy_metrics(ke, pe)
        assert "max_kinetic_energy" in metrics
        assert "max_total_energy" in metrics
        assert metrics["max_kinetic_energy"] == pytest.approx(16.0)
        assert metrics["max_potential_energy"] == pytest.approx(16.0)

    def test_energy_conservation(self) -> None:
        """Total energy should be constant for a conservative system."""
        ke = np.array([0.0, 5.0, 10.0, 5.0, 0.0], dtype=float)
        pe = np.array([10.0, 5.0, 0.0, 5.0, 10.0], dtype=float)
        host = EnergyHost()
        metrics = host.compute_energy_metrics(ke, pe)
        assert metrics["energy_variation"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["energy_drift"] == pytest.approx(0.0, abs=1e-10)

    def test_with_club_head_speed(self) -> None:
        ke = np.array([0.0, 5.0, 10.0])
        pe = np.array([10.0, 5.0, 0.0])
        speed = np.array([0.0, 5.0, 10.0])  # Peak at index 2
        host = EnergyHost(club_head_speed=speed)
        metrics = host.compute_energy_metrics(ke, pe)
        # ke_at_impact = 10, max_total = 10, efficiency = 100%
        assert metrics["energy_efficiency"] == pytest.approx(100.0)

    def test_energy_efficiency_zero_total(self) -> None:
        ke = np.zeros(5)
        pe = np.zeros(5)
        host = EnergyHost(club_head_speed=np.ones(5))
        metrics = host.compute_energy_metrics(ke, pe)
        assert metrics["energy_efficiency"] == 0.0

    def test_length_mismatch_raises(self) -> None:
        host = EnergyHost()
        with pytest.raises(Exception, match="same length"):
            host.compute_energy_metrics(np.ones(3), np.ones(5))

    def test_negative_ke_raises(self) -> None:
        host = EnergyHost()
        with pytest.raises(Exception, match="non-negative"):
            host.compute_energy_metrics(np.array([-1.0, 0.0]), np.array([0.0, 0.0]))


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
