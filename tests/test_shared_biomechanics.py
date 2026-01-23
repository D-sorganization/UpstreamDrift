"""Tests for shared biomechanics analysis and plotting modules."""

import numpy as np
import pytest

from src.shared.python.biomechanics_data import BiomechanicalData
from src.shared.python.plotting import GolfSwingPlotter, RecorderInterface
from src.shared.python.statistical_analysis import (
    StatisticalAnalyzer,
    SummaryStatistics,
    SwingPhase,
)


class MockRecorder(RecorderInterface):
    """Mock recorder for testing."""

    def __init__(self, num_samples=100, num_joints=3) -> None:
        self.times = np.linspace(0, 1, num_samples)
        self.num_joints = num_joints
        self.joint_positions = np.zeros((num_samples, num_joints))
        self.joint_velocities = np.zeros((num_samples, num_joints))
        self.joint_torques = np.zeros((num_samples, num_joints))
        self.club_head_speed = np.abs(np.sin(self.times * 10))  # Fake speed profile
        self.kinetic_energy = np.linspace(0, 10, num_samples)
        self.potential_energy = np.linspace(10, 0, num_samples)
        self.total_energy = self.kinetic_energy + self.potential_energy
        self.actuator_powers = np.zeros((num_samples, num_joints))
        self.club_head_position = np.zeros((num_samples, 3))

        # Fill with some data
        for i in range(num_joints):
            self.joint_positions[:, i] = np.sin(self.times * (i + 1))
            self.joint_velocities[:, i] = np.cos(self.times * (i + 1))
            self.actuator_powers[:, i] = np.sin(self.times)

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self, field_name):
            return self.times, getattr(self, field_name)
        return self.times, np.array([])

    def get_counterfactual_series(
        self, field_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return counterfactual data series."""
        return self.times, np.zeros_like(self.times)

    def get_induced_acceleration_series(
        self, field_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return induced acceleration data series."""
        return self.times, np.zeros_like(self.times)


def test_statistical_analyzer() -> None:
    """Test StatisticalAnalyzer metrics calculation."""
    times = np.linspace(0, 1, 100)
    # Sin wave
    data = np.sin(times * 2 * np.pi)

    analyzer = StatisticalAnalyzer(
        times=times,
        joint_positions=np.zeros((100, 1)),
        joint_velocities=np.zeros((100, 1)),
        joint_torques=np.zeros((100, 1)),
        club_head_speed=np.abs(data) * 10,
    )

    # Basic stats
    stats = analyzer.compute_summary_stats(data)
    assert isinstance(stats, SummaryStatistics)
    assert stats.min == pytest.approx(-1.0, abs=0.01)
    assert stats.max == pytest.approx(1.0, abs=0.01)
    assert stats.mean == pytest.approx(0.0, abs=0.1)

    # Peaks
    peaks = analyzer.find_peaks_in_data(data, height=0.5)
    assert len(peaks) > 0
    assert peaks[0].value > 0.5

    # Club head speed peak
    peak_speed = analyzer.find_club_head_speed_peak()
    assert peak_speed is not None
    assert peak_speed.value > 0

    # Tempo
    tempo = analyzer.compute_tempo()
    # It might be None if data isn't shaped like a swing, but function shouldn't crash
    if tempo:
        assert len(tempo) == 3

    # Phases
    phases = analyzer.detect_swing_phases()
    assert len(phases) > 0
    assert isinstance(phases[0], SwingPhase)

    # Report
    report = analyzer.generate_comprehensive_report()
    assert "duration" in report
    assert "joints" in report


def test_biomechanics_data() -> None:
    """Test BiomechanicalData container."""
    data = BiomechanicalData(time=0.5)
    assert data.time == 0.5
    assert isinstance(data.joint_positions, np.ndarray)
    assert data.club_head_speed == 0.0


def test_plotter_alignment() -> None:
    """Test joint name alignment logic in GolfSwingPlotter."""
    recorder = MockRecorder(num_joints=3)

    # Case 1: Perfect match
    joint_names = ["J1", "J2", "J3"]
    plotter = GolfSwingPlotter(recorder, joint_names)
    assert plotter._get_aligned_label(0, 3) == "J1"
    assert plotter._get_aligned_label(2, 3) == "J3"

    # Case 2: Mismatch (e.g. FreeFlyer base at start)
    # Data has 4 columns (1 base + 3 joints), Names has 3 entries
    assert plotter._get_aligned_label(1, 4) == "J1"  # Index 1 aligned to Name 0
    assert plotter._get_aligned_label(3, 4) == "J3"  # Index 3 aligned to Name 2
    assert plotter._get_aligned_label(0, 4) == "DoF 0"  # Base unmatched

    # Case 3: Empty names
    plotter_empty = GolfSwingPlotter(recorder, [])
    assert plotter_empty._get_aligned_label(0, 3) == "DoF 0"


def test_plotter_methods() -> None:
    """Test that plotter methods run without error."""
    recorder = MockRecorder()
    plotter = GolfSwingPlotter(recorder, ["J1", "J2", "J3"])

    # We rely on matplotlib Figure, not checking render output, just execution
    from matplotlib.figure import Figure

    fig = Figure()

    plotter.plot_joint_angles(fig)
    assert len(fig.axes) > 0
    fig.clear()

    plotter.plot_joint_velocities(fig)
    assert len(fig.axes) > 0
    fig.clear()

    plotter.plot_phase_diagram(fig, joint_idx=0)
    assert len(fig.axes) > 0
    fig.clear()

    plotter.plot_kinematic_sequence(fig, {"J1": 0, "J2": 1})
    assert len(fig.axes) > 0
    fig.clear()

    plotter.plot_energy_analysis(fig)
    assert len(fig.axes) > 0
    fig.clear()

    plotter.plot_torque_comparison(fig)
    assert len(fig.axes) > 0
    fig.clear()

    plotter.plot_actuator_powers(fig)
    assert len(fig.axes) > 0
    fig.clear()

    plotter.plot_club_head_speed(fig)
    assert len(fig.axes) > 0
    fig.clear()

    plotter.plot_summary_dashboard(fig)
    assert len(fig.axes) > 0
