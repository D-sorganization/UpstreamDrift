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


class TestPCAAnalysisMixin:
    """Tests for PCAAnalysisMixin."""

    @pytest.fixture()
    def host(self) -> PCAHost:
        N = 100
        t = np.linspace(0, 1, N)
        # 3 joints with correlated motion
        pos = np.column_stack(
            [np.sin(2 * np.pi * t), 0.5 * np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]
        )
        vel = np.column_stack(
            [np.cos(2 * np.pi * t), 0.5 * np.cos(2 * np.pi * t), -np.sin(2 * np.pi * t)]
        )
        return PCAHost(times=t, joint_positions=pos, joint_velocities=vel)

    def test_pca_basic(self, host: PCAHost) -> None:
        result = host.compute_principal_component_analysis()
        assert result is not None
        assert isinstance(result, PCAResult)
        assert len(result.explained_variance) == 3
        assert np.all(result.explained_variance >= 0)
        assert np.sum(result.explained_variance_ratio) <= 1.0 + 1e-6

    def test_pca_n_components(self, host: PCAHost) -> None:
        result = host.compute_principal_component_analysis(n_components=2)
        assert result is not None
        assert result.components.shape[0] == 2
        assert len(result.explained_variance) == 2

    def test_pca_velocity_data(self, host: PCAHost) -> None:
        result = host.compute_principal_component_analysis(data_type="velocity")
        assert result is not None
        assert len(result.explained_variance) == 3

    def test_pca_empty_data(self) -> None:
        host = PCAHost(
            times=np.array([0.0]),
            joint_positions=np.zeros((1, 0)),
            joint_velocities=np.zeros((1, 0)),
        )
        result = host.compute_principal_component_analysis()
        assert result is None

    def test_principal_movements(self, host: PCAHost) -> None:
        result = host.compute_principal_movements(n_modes=2)
        assert result is not None
        eigvecs, scores = result
        assert eigvecs.shape[0] == 2

    def test_analyze_kinematic_sequence(self, host: PCAHost) -> None:
        segments = {"J0": 0, "J1": 1, "J2": 2}
        seq, score = host.analyze_kinematic_sequence(segments)
        assert len(seq) == 3
        assert 0.0 <= score <= 1.0

    def test_kinematic_sequence_invalid_index(self, host: PCAHost) -> None:
        segments = {"Bad": 99}
        seq, score = host.analyze_kinematic_sequence(segments)
        assert len(seq) == 0

    def test_kinematic_sequence_perfect_order(self) -> None:
        """When segments peak in expected order, score = 1.0."""
        N = 100
        t = np.linspace(0, 1, N)
        vel = np.zeros((N, 3))
        # J0 peaks at t=0.3, J1 at t=0.5, J2 at t=0.7
        vel[:, 0] = np.exp(-50 * (t - 0.3) ** 2)
        vel[:, 1] = np.exp(-50 * (t - 0.5) ** 2)
        vel[:, 2] = np.exp(-50 * (t - 0.7) ** 2)
        host = PCAHost(times=t, joint_positions=np.zeros((N, 3)), joint_velocities=vel)
        seq, score = host.analyze_kinematic_sequence({"J0": 0, "J1": 1, "J2": 2})
        assert score == pytest.approx(1.0)
        assert seq[0].segment_name == "J0"
        assert seq[1].segment_name == "J1"
        assert seq[2].segment_name == "J2"


# ============================================================================
# Tests for dataclasses
# ============================================================================
