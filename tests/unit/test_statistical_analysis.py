"""
Unit tests for shared.python.statistical_analysis module.
"""

import numpy as np
import pytest

from shared.python.statistical_analysis import StatisticalAnalyzer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    times = np.linspace(0, 1.0, 101)  # 101 points, dt=0.01

    # Create synthetic club head speed:
    # Rise to peak at 0.5s, then fall
    # Gaussian-ish
    speed = 50.0 * np.exp(-((times - 0.5) ** 2) / (2 * 0.1**2))

    # Positions (2 joints)
    # Joint 0: Linear increase (hips)
    # Joint 1: Quadratic (shoulders)
    positions = np.zeros((101, 2))
    positions[:, 0] = np.deg2rad(45 * times)
    positions[:, 1] = np.deg2rad(90 * times**2)

    velocities = np.zeros((101, 2))
    torques = np.zeros((101, 2))

    # CoP (Circle)
    cop = np.stack([np.cos(2*np.pi*times), np.sin(2*np.pi*times)], axis=1)

    # Ground Forces
    forces = np.zeros((101, 3))
    forces[:, 2] = 1000.0 # Vertical

    return {
        "times": times,
        "positions": positions,
        "velocities": velocities,
        "torques": torques,
        "speed": speed,
        "cop": cop,
        "forces": forces
    }

@pytest.fixture
def analyzer(sample_data):
    return StatisticalAnalyzer(
        times=sample_data["times"],
        joint_positions=sample_data["positions"],
        joint_velocities=sample_data["velocities"],
        joint_torques=sample_data["torques"],
        club_head_speed=sample_data["speed"],
        cop_position=sample_data["cop"],
        ground_forces=sample_data["forces"]
    )

class TestStatisticalAnalyzer:
    def test_initialization(self, analyzer):
        assert analyzer.dt == pytest.approx(0.01)
        assert analyzer.duration == pytest.approx(1.0)

    def test_compute_summary_stats(self, analyzer):
        data = np.array([1, 2, 3, 4, 5])
        # Manually create analyzer for simple data or just use method
        stats = analyzer.compute_summary_stats(data)
        assert stats.min == 1
        assert stats.max == 5
        assert stats.mean == 3

    def test_find_club_head_speed_peak(self, analyzer):
        peak = analyzer.find_club_head_speed_peak()
        assert peak is not None
        assert peak.time == pytest.approx(0.5, abs=0.01)
        assert peak.value > 49.0

    def test_compute_x_factor(self, analyzer):
        # Joint 0 (hips), Joint 1 (shoulders)
        # Difference
        xf = analyzer.compute_x_factor(1, 0)
        assert xf is not None
        assert len(xf) == 101

    def test_compute_x_factor_stretch(self, analyzer):
        vel, peak = analyzer.compute_x_factor_stretch(1, 0)
        assert vel is not None
        assert len(vel) == 101
        assert peak >= 0

    def test_compute_grf_metrics(self, analyzer):
        metrics = analyzer.compute_grf_metrics()
        assert metrics is not None
        # CoP is circle radius 1. Path length ~ 2*pi
        assert metrics.cop_path_length == pytest.approx(2 * np.pi, abs=0.1)
        assert metrics.peak_vertical_force == 1000.0

    def test_detect_swing_phases(self, analyzer):
        phases = analyzer.detect_swing_phases()
        assert len(phases) > 0
        names = [p.name for p in phases]
        assert "Impact" in names
        assert "Downswing" in names

    def test_generate_report(self, analyzer):
        report = analyzer.generate_comprehensive_report()
        assert "club_head_speed" in report
        assert "grf_metrics" in report
        assert "joints" in report
