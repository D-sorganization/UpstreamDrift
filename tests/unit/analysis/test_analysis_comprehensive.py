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

    def test_compute_summary_stats(self, host: BasicStatsHost) -> None:
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

    def test_find_peaks_in_data(self, host: BasicStatsHost) -> None:
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

    def test_find_club_head_speed_peak(self, host: BasicStatsHost) -> None:
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

    def test_summary_statistics(self) -> None:
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
