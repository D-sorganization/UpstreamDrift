import numpy as np
import pytest

from src.shared.python.comparative_analysis import (
    AlignedSignals,
    ComparativeSwingAnalyzer,
    RecorderInterface,
)


class MockRecorder(RecorderInterface):
    """Mock recorder for testing."""

    def __init__(self, data_dict):
        self.data = data_dict

    def get_time_series(self, field_name: str):
        return self.data.get(field_name, (np.array([]), np.array([])))


@pytest.fixture
def sample_data():
    t = np.linspace(0, 1, 10)

    # Swing A: linear
    data_a = {
        "joint_positions": (t, np.column_stack((t, t * 2))),
        "club_head_speed": (t, np.ones_like(t) * 10),
        "kinetic_energy": (t, np.ones_like(t) * 50),
        "angular_momentum": (t, np.ones((10, 3))),
        "cop_position": (t, np.column_stack((t, t))),
    }

    # Swing B: slightly different linear
    data_b = {
        "joint_positions": (t, np.column_stack((t * 1.1, t * 2.2))),
        "club_head_speed": (t, np.ones_like(t) * 8),
        "kinetic_energy": (t, np.ones_like(t) * 40),
        "angular_momentum": (t, np.ones((10, 3)) * 0.8),
        "cop_position": (t, np.column_stack((t * 0.9, t * 0.9))),
    }

    return data_a, data_b


def test_align_signals(sample_data):
    data_a, data_b = sample_data
    rec_a = MockRecorder(data_a)
    rec_b = MockRecorder(data_b)

    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)

    # Test 1D signal (implicitly via multi-dim slicing or just club speed)
    aligned = analyzer.align_signals("club_head_speed")
    assert aligned is not None
    assert isinstance(aligned, AlignedSignals)
    assert len(aligned.times) == 100
    assert np.allclose(aligned.signal_a, 10.0)
    assert np.allclose(aligned.signal_b, 8.0)
    assert aligned.rms_error > 0

    # Test 2D signal with joint index
    aligned_joint = analyzer.align_signals("joint_positions", joint_idx=0)
    assert aligned_joint is not None
    assert np.allclose(aligned_joint.signal_a, np.linspace(0, 1, 100))  # roughly

    # Test missing data
    aligned_missing = analyzer.align_signals("nonexistent")
    assert aligned_missing is None

    # Test joint index out of bounds
    aligned_oob = analyzer.align_signals("joint_positions", joint_idx=99)
    assert aligned_oob is None


def test_compare_scalars():
    rec = MockRecorder({})
    analyzer = ComparativeSwingAnalyzer(rec, rec)

    metric = analyzer.compare_scalars("Test", 10.0, 8.0)
    assert metric.value_a == 10.0
    assert metric.value_b == 8.0
    assert metric.difference == 2.0
    assert np.isclose(metric.percent_diff, (2.0 / 9.0) * 100)


def test_generate_comparison_report(sample_data):
    data_a, data_b = sample_data
    rec_a = MockRecorder(data_a)
    rec_b = MockRecorder(data_b)

    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)

    report = analyzer.generate_comparison_report()

    assert report["swing_a"] == "Swing A"
    assert report["swing_b"] == "Swing B"
    assert len(report["metrics"]) > 0

    metric_names = [m.name for m in report["metrics"]]
    assert "Peak Club Speed" in metric_names
    assert "Swing Duration" in metric_names
    assert "Max Kinetic Energy" in metric_names
    assert "Max Angular Momentum" in metric_names
    assert "CoP Path Length" in metric_names


def test_missing_data_report():
    rec_empty = MockRecorder({})
    analyzer = ComparativeSwingAnalyzer(rec_empty, rec_empty)

    report = analyzer.generate_comparison_report()
    assert len(report["metrics"]) == 0


def test_compute_dtw_distance(sample_data):
    data_a, data_b = sample_data
    rec_a = MockRecorder(data_a)
    rec_b = MockRecorder(data_b)

    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)

    # Test DTW on club_head_speed
    dist, path = analyzer.compute_dtw_distance("club_head_speed", radius=5)

    assert isinstance(dist, float)
    assert dist >= 0.0
    assert isinstance(path, list)
    assert len(path) > 0
    assert isinstance(path[0], tuple)
    assert len(path[0]) == 2
    # Ensure path starts at end (backtracked) or matches expectation of N, M
    # signal_processing.compute_dtw_path returns path from end (backtracked) BUT reversed before return?
    # Let's check signal_processing code:
    # "pi, pj are in reverse order ... Loop backwards to reverse" -> It returns forward path!
    # "path.append((int(pi[k]), int(pj[k])))"

    # So path[0] should be (0, 0)
    assert path[0] == (0, 0)

    # Path should end at (N-1, M-1)
    N = len(data_a["club_head_speed"][0])
    M = len(data_b["club_head_speed"][0])
    assert path[-1] == (N - 1, M - 1)
