"""Tests for comparative analysis module."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from shared.python.comparative_analysis import (
    ComparativeSwingAnalyzer,
    RecorderInterface,
)
from shared.python.comparative_plotting import ComparativePlotter


class MockRecorder(RecorderInterface):
    """Mock recorder for testing."""

    def __init__(self, duration=1.0, amplitude=1.0, offset=0.0) -> None:
        self.times = np.linspace(0, duration, 100)
        norm_time = np.linspace(0, 1, 100)

        # Simple sine wave (one full cycle per swing)
        self.signal = amplitude * np.sin(2 * np.pi * norm_time) + offset

        # Multidimensional data
        self.joint_positions = np.column_stack([self.signal, self.signal * 0.5])
        self.joint_velocities = np.column_stack(
            [np.cos(2 * np.pi * norm_time), np.cos(2 * np.pi * norm_time) * 0.5]
        )

        # Club head speed (bell curve centered at 50% of swing)
        self.club_head_speed = amplitude * np.exp(-((norm_time - 0.5) ** 2) / 0.05)

        # Energy
        self.kinetic_energy = self.club_head_speed**2

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self, field_name):
            return self.times, getattr(self, field_name)
        return self.times, np.array([])


def test_alignment() -> None:
    """Test signal alignment."""
    # Create two recorders with different durations but same shape
    rec_a = MockRecorder(duration=1.0, amplitude=1.0)
    rec_b = MockRecorder(duration=2.0, amplitude=1.0)  # Slower swing

    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)

    # Test alignment of 1D signal
    aligned = analyzer.align_signals("club_head_speed")
    assert aligned is not None
    assert len(aligned.times) == 100
    assert aligned.times[-1] == 1.0

    # Check that peaks align roughly (both centered at 50%)
    peak_idx_a = np.argmax(aligned.signal_a)
    peak_idx_b = np.argmax(aligned.signal_b)
    assert abs(peak_idx_a - peak_idx_b) < 5  # Allow small interpolation error

    # Test alignment of 2D signal (joint 0)
    aligned_joint = analyzer.align_signals("joint_positions", joint_idx=0)
    assert aligned_joint is not None
    assert aligned_joint.correlation > 0.95  # Should be highly correlated


def test_metrics() -> None:
    """Test metric computation."""
    rec_a = MockRecorder(amplitude=10.0)
    rec_b = MockRecorder(amplitude=8.0)

    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)

    # Peak speed
    speed_metric = analyzer.compare_peak_speeds()
    assert speed_metric is not None
    assert speed_metric.value_a > speed_metric.value_b
    assert speed_metric.difference == pytest.approx(2.0, abs=0.1)

    # Duration (same duration in mock)
    dur_metric = analyzer.compare_durations()
    assert dur_metric is not None
    assert dur_metric.difference == pytest.approx(0.0)

    # Report
    report = analyzer.generate_comparison_report()
    assert "metrics" in report
    assert len(report["metrics"]) >= 2


def test_plotter() -> None:
    """Test plotting functions."""
    rec_a = MockRecorder(amplitude=1.0)
    rec_b = MockRecorder(amplitude=0.8)
    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)
    plotter = ComparativePlotter(analyzer)

    fig = Figure()

    # Test simple plot
    plotter.plot_comparison(fig, "club_head_speed")
    assert len(fig.axes) > 0
    fig.clear()

    # Test phase comparison
    plotter.plot_phase_comparison(fig, joint_idx=0)
    assert len(fig.axes) > 0
    fig.clear()

    # Test dashboard (should not crash)
    # Note: Using subfigures requires newer matplotlib, if this fails we'll need to update dep or change impl
    plotter.plot_dashboard(fig)
    assert len(fig.axes) > 0
