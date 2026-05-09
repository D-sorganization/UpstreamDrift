"""Tests for src.shared.python.analysis.reporting (Issues #1949, #1744)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.shared.python.analysis.angular_momentum import AngularMomentumMetricsMixin
from src.shared.python.analysis.basic_stats import BasicStatsMixin
from src.shared.python.analysis.grf_metrics import GRFMetricsMixin
from src.shared.python.analysis.phase_detection import PhaseDetectionMixin
from src.shared.python.analysis.reporting import ReportingMixin
from src.shared.python.analysis.stability_metrics import StabilityMetricsMixin
from src.shared.python.analysis.swing_metrics import SwingMetricsMixin


class _Concrete(
    ReportingMixin,
    BasicStatsMixin,
    PhaseDetectionMixin,
    SwingMetricsMixin,
    GRFMetricsMixin,
    AngularMomentumMetricsMixin,
    StabilityMetricsMixin,
):
    """Concrete test class combining all required mixins."""

    def __init__(self, n: int = 100, n_joints: int = 3) -> None:
        t = np.linspace(0.0, 2.0, n)
        self.times = t
        self.dt = float(t[1] - t[0])
        self.duration = float(t[-1] - t[0])

        pos = np.zeros((n, n_joints))
        vel = np.zeros((n, n_joints))
        torques = np.zeros((n, n_joints))
        for j in range(n_joints):
            pos[:, j] = np.sin(2 * np.pi * (j + 1) * t)
            vel[:, j] = np.gradient(pos[:, j], self.dt)
            torques[:, j] = vel[:, j] * 0.5

        self.joint_positions = pos
        self.joint_velocities = vel
        self.joint_torques = torques
        self.joint_accelerations = np.gradient(vel, self.dt, axis=0)

        # Simulated club head speed (increases then decreases)
        self.club_head_speed = np.abs(np.sin(np.pi * t / 2.0)) * 50.0

        # No CoP/CoM for stability (None → stability returns None)
        self.cop_position = None
        self.com_position = None
        self.grf_vertical = None
        self.angular_momentum = None


def _make() -> _Concrete:
    return _Concrete(n=80, n_joints=3)


class TestGenerateComprehensiveReport:
    def test_reporting_returns_dict(self) -> None:
        result = _make().generate_comprehensive_report()
        assert isinstance(result, dict)

    def test_has_duration(self) -> None:
        result = _make().generate_comprehensive_report()
        assert "duration" in result
        assert result["duration"] > 0.0

    def test_has_num_samples(self) -> None:
        result = _make().generate_comprehensive_report()
        assert "num_samples" in result
        assert result["num_samples"] == 80

    def test_has_sample_rate(self) -> None:
        result = _make().generate_comprehensive_report()
        assert "sample_rate" in result
        assert result["sample_rate"] > 0.0

    def test_has_phases(self) -> None:
        result = _make().generate_comprehensive_report()
        assert "phases" in result
        assert isinstance(result["phases"], list)

    def test_has_joints(self) -> None:
        result = _make().generate_comprehensive_report()
        assert "joints" in result
        assert "joint_0" in result["joints"]
        assert "joint_1" in result["joints"]

    def test_joint_data_has_range_of_motion(self) -> None:
        result = _make().generate_comprehensive_report()
        j0 = result["joints"]["joint_0"]
        assert "range_of_motion" in j0
        assert "rom_deg" in j0["range_of_motion"]


class TestComputeJerkMetrics:
    def test_returns_jerk_metrics_or_none(self) -> None:
        obj = _make()
        result = obj.compute_jerk_metrics(joint_idx=0)
        # Returns JerkMetrics or None depending on data
        assert result is not None

    def test_negative_joint_idx_raises(self) -> None:
        obj = _make()
        with pytest.raises((ValueError, TypeError, AssertionError)):
            obj.compute_jerk_metrics(joint_idx=-1)

    def test_out_of_range_joint_idx_returns_none(self) -> None:
        obj = _make()
        result = obj.compute_jerk_metrics(joint_idx=99)
        assert result is None

    def test_jerk_metrics_peak_non_negative(self) -> None:
        obj = _make()
        result = obj.compute_jerk_metrics(joint_idx=0)
        if result is not None:
            assert result.peak_jerk >= 0.0

    def test_jerk_metrics_rms_non_negative(self) -> None:
        obj = _make()
        result = obj.compute_jerk_metrics(joint_idx=0)
        if result is not None:
            assert result.rms_jerk >= 0.0


class TestExportStatisticsCsv:
    def test_creates_file(self, tmp_path: Path) -> None:
        obj = _make()
        filename = str(tmp_path / "output.csv")
        obj.export_statistics_csv(filename)
        assert Path(filename).exists()

    def test_empty_filename_raises(self) -> None:
        obj = _make()
        with pytest.raises((ValueError, TypeError, AssertionError)):
            obj.export_statistics_csv("")

    def test_file_has_content(self, tmp_path: Path) -> None:
        obj = _make()
        filename = str(tmp_path / "output.csv")
        obj.export_statistics_csv(filename)
        content = Path(filename).read_text()
        assert len(content) > 0

    def test_accepts_existing_report(self, tmp_path: Path) -> None:
        obj = _make()
        report = obj.generate_comprehensive_report()
        filename = str(tmp_path / "output.csv")
        obj.export_statistics_csv(filename, report=report)
        assert Path(filename).exists()


class TestComputeSwingProfile:
    def test_returns_swing_profile_metrics(self) -> None:
        from src.shared.python.analysis.dataclasses import SwingProfileMetrics

        obj = _make()
        result = obj.compute_swing_profile()
        assert isinstance(result, SwingProfileMetrics)

    def test_speed_score_in_range(self) -> None:
        obj = _make()
        result = obj.compute_swing_profile()
        assert 0.0 <= result.speed_score <= 100.0

    def test_efficiency_score_in_range(self) -> None:
        obj = _make()
        result = obj.compute_swing_profile()
        assert 0.0 <= result.efficiency_score <= 100.0

    def test_power_score_in_range(self) -> None:
        obj = _make()
        result = obj.compute_swing_profile()
        assert 0.0 <= result.power_score <= 100.0
