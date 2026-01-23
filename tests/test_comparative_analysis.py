"""Tests for comparative analysis module.

TEST-001: Enhanced test coverage for comparative_analysis.py.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.shared.python.comparative_analysis import (
    AlignedSignals,
    ComparativeSwingAnalyzer,
    ComparisonMetric,
    RecorderInterface,
)
from src.shared.python.comparative_plotting import ComparativePlotter


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

        # Angular momentum (3D vector)
        self.angular_momentum = np.column_stack(
            [self.signal, self.signal * 0.5, self.signal * 0.3]
        )

        # Center of pressure (3D position)
        self.cop_position = np.column_stack(
            [
                norm_time,  # x moves forward
                np.sin(4 * np.pi * norm_time) * 0.1,  # y oscillates
                np.zeros(100),  # z stays at ground level
            ]
        )

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


# TEST-001: Additional edge case and coverage tests
def test_dataclasses() -> None:
    """Test dataclass structures."""
    # ComparisonMetric
    metric = ComparisonMetric(
        name="test_metric", value_a=10.0, value_b=8.0, difference=2.0, percent_diff=22.2
    )
    assert metric.name == "test_metric"
    assert metric.value_a == 10.0
    assert metric.difference == 2.0

    # AlignedSignals
    times = np.linspace(0, 1, 50)
    signal_a = np.sin(times)
    signal_b = np.cos(times)
    aligned = AlignedSignals(
        times=times,
        signal_a=signal_a,
        signal_b=signal_b,
        error_curve=signal_a - signal_b,
        rms_error=0.5,
        correlation=0.8,
    )
    assert len(aligned.times) == 50
    assert aligned.rms_error == 0.5
    assert aligned.correlation == 0.8


def test_edge_cases_insufficient_data() -> None:
    """Test handling of insufficient data."""
    rec_a = MockRecorder()
    rec_b = MockRecorder()
    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)

    # Single point data - should return None
    rec_a.times = np.array([0.0])
    rec_a.club_head_speed = np.array([1.0])

    aligned = analyzer.align_signals("club_head_speed")
    assert aligned is None


def test_edge_cases_invalid_joint_index() -> None:
    """Test handling of out-of-bounds joint index."""
    rec_a = MockRecorder()
    rec_b = MockRecorder()
    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)

    # Request joint index that doesn't exist
    aligned = analyzer.align_signals("joint_positions", joint_idx=999)
    assert aligned is None


def test_edge_cases_zero_correlation() -> None:
    """Test handling of constant signals (zero correlation)."""
    rec_a = MockRecorder()
    rec_b = MockRecorder()
    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)

    # Create constant signals (zero std dev)
    rec_a.club_head_speed = np.ones(100) * 5.0
    rec_b.club_head_speed = np.ones(100) * 5.0

    aligned = analyzer.align_signals("club_head_speed")
    assert aligned is not None
    assert aligned.correlation == 0.0  # Constant signals have no correlation


def test_compare_scalars_zero_mean() -> None:
    """Test compare_scalars with zero mean case."""
    rec_a = MockRecorder()
    rec_b = MockRecorder()
    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)

    # Values that average to nearly zero
    metric = analyzer.compare_scalars("test_metric", 1e-10, -1e-10)
    assert metric.percent_diff == 0.0  # Should handle division by near-zero


def test_missing_fields() -> None:
    """Test handling of missing recorder fields."""
    rec_a = MockRecorder()
    rec_b = MockRecorder()
    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)

    # Compare peak speeds when field is missing
    delattr(rec_a, "club_head_speed")
    speed_metric = analyzer.compare_peak_speeds()
    assert speed_metric is None

    # Compare durations when field is missing
    # When missing, compare_scalars handles 0.0 vs 0.0, so we expect a valid metric with 0 diff
    dur_metric = analyzer.compare_durations()
    assert dur_metric is not None
    assert dur_metric.difference == 0.0


def test_generate_full_report() -> None:
    """Test comprehensive report generation with all metrics."""
    rec_a = MockRecorder(amplitude=10.0)
    rec_b = MockRecorder(amplitude=8.0)
    analyzer = ComparativeSwingAnalyzer(
        rec_a, rec_b, name_a="Pro Swing", name_b="Amateur Swing"
    )

    report = analyzer.generate_comparison_report()

    # Check structure
    assert "swing_a" in report
    assert "swing_b" in report
    assert "metrics" in report
    assert report["swing_a"] == "Pro Swing"
    assert report["swing_b"] == "Amateur Swing"

    # Check metrics were collected
    assert len(report["metrics"]) >= 4  # speed, duration, energy, angular momentum, cop

    # Check each metric type appears
    metric_names = [m.name for m in report["metrics"]]
    assert any("Speed" in name for name in metric_names)
    assert any("Duration" in name for name in metric_names)
    assert any("Energy" in name for name in metric_names)


def test_custom_num_points() -> None:
    """Test alignment with custom number of interpolation points."""
    rec_a = MockRecorder()
    rec_b = MockRecorder()
    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)

    aligned = analyzer.align_signals("club_head_speed", num_points=200)
    assert aligned is not None
    assert len(aligned.times) == 200
    assert aligned.times[0] == 0.0
    assert aligned.times[-1] == 1.0


def test_multidimensional_alignment() -> None:
    """Test alignment of multidimensional signals."""
    rec_a = MockRecorder()
    rec_b = MockRecorder()
    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)

    # Test each joint dimension
    for joint_idx in [0, 1]:
        aligned = analyzer.align_signals("joint_positions", joint_idx=joint_idx)
        assert aligned is not None
        assert len(aligned.signal_a) == 100
        assert len(aligned.signal_b) == 100
        assert len(aligned.error_curve) == 100


def test_angular_momentum_comparison() -> None:
    """Test angular momentum magnitude comparison in report."""
    rec_a = MockRecorder(amplitude=10.0)
    rec_b = MockRecorder(amplitude=5.0)
    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)

    report = analyzer.generate_comparison_report()

    # Find angular momentum metric
    am_metric = next(
        (m for m in report["metrics"] if "Angular Momentum" in m.name), None
    )
    assert am_metric is not None
    assert (
        am_metric.value_a > am_metric.value_b
    )  # Higher amplitude = higher angular momentum


def test_cop_path_length_comparison() -> None:
    """Test center of pressure path length comparison in report."""
    rec_a = MockRecorder(amplitude=1.0)
    rec_b = MockRecorder(amplitude=0.5)
    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)

    report = analyzer.generate_comparison_report()

    # Find CoP path length metric
    cop_metric = next((m for m in report["metrics"] if "CoP Path" in m.name), None)
    assert cop_metric is not None
    # Both should have some path length > 0
    assert cop_metric.value_a > 0
    assert cop_metric.value_b > 0
