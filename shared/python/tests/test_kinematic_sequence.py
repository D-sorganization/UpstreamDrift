"""Tests for the kinematic sequence analyzer."""

import numpy as np
import pytest

from shared.python.kinematic_sequence import (
    KinematicSequenceAnalyzer,
)


@pytest.fixture
def analyzer():
    """Default analyzer with standard order."""
    return KinematicSequenceAnalyzer(expected_order=["Pelvis", "Torso", "Arm", "Club"])


def test_perfect_sequence(analyzer):
    """Test a perfectly timed kinematic sequence."""
    times = np.linspace(0, 1.0, 100)

    # Create Gaussian peaks at different times
    def gaussian(t, center):
        return np.exp(-((t - center) ** 2) / 0.01)

    velocities = {
        "Pelvis": gaussian(times, 0.2) * 5.0,
        "Torso": gaussian(times, 0.3) * 8.0,
        "Arm": gaussian(times, 0.4) * 12.0,
        "Club": gaussian(times, 0.5) * 45.0,
    }

    result = analyzer.analyze(velocities, times)

    assert result.is_valid_sequence
    assert result.efficiency_score == 1.0
    assert result.sequence_order == ["Pelvis", "Torso", "Arm", "Club"]

    # Check normalized velocities
    # Club is max (45.0).
    club_peak = next(p for p in result.peaks if p.name == "Club")
    assert np.isclose(club_peak.normalized_velocity, 1.0)

    pelvis_peak = next(p for p in result.peaks if p.name == "Pelvis")
    # Use larger tolerance because discrete sampling of Gaussian might miss exact peak
    assert np.isclose(pelvis_peak.normalized_velocity, 5.0 / 45.0, atol=0.01)


def test_out_of_sequence(analyzer):
    """Test a sequence where peaks are out of order."""
    times = np.linspace(0, 1.0, 100)

    def gaussian(t, center):
        return np.exp(-((t - center) ** 2) / 0.01)

    # Arm fires before Torso
    velocities = {
        "Pelvis": gaussian(times, 0.2),
        "Arm": gaussian(times, 0.3),  # Early arm
        "Torso": gaussian(times, 0.4),  # Late torso
        "Club": gaussian(times, 0.5),
    }

    result = analyzer.analyze(velocities, times)

    assert not result.is_valid_sequence
    assert result.efficiency_score < 1.0
    assert result.sequence_order == ["Pelvis", "Arm", "Torso", "Club"]


def test_missing_segments(analyzer):
    """Test analysis with subset of segments."""
    times = np.linspace(0, 1.0, 100)

    def gaussian(t, center):
        return np.exp(-((t - center) ** 2) / 0.01)

    # Only Pelvis and Club provided
    velocities = {"Pelvis": gaussian(times, 0.2), "Club": gaussian(times, 0.5)}

    result = analyzer.analyze(velocities, times)

    # Should be valid relative to each other (Pelvis before Club)
    assert result.is_valid_sequence
    assert result.efficiency_score == 1.0
    assert result.sequence_order == ["Pelvis", "Club"]


def test_empty_data(analyzer):
    """Test with empty input."""
    result = analyzer.analyze({}, np.array([]))

    assert not result.is_valid_sequence
    assert result.efficiency_score == 0.0
    assert len(result.peaks) == 0


def test_custom_order():
    """Test with custom expected order."""
    custom_analyzer = KinematicSequenceAnalyzer(expected_order=["A", "B", "C"])
    times = np.linspace(0, 1, 10)
    velocities = {
        "C": np.array([0] * 9 + [1]),  # Peak at end
        "B": np.array([0] * 5 + [1] + [0] * 4),  # Peak in middle
        "A": np.array([1] + [0] * 9),  # Peak at start
    }

    result = custom_analyzer.analyze(velocities, times)

    assert result.is_valid_sequence
    assert result.sequence_order == ["A", "B", "C"]


def test_timing_gaps(analyzer):
    """Test calculation of timing gaps."""
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

    # Peaks at indices 1 and 3 (times 0.1 and 0.3)
    velocities = {
        "Pelvis": np.array([0, 10, 0, 0, 0]),
        "Torso": np.array([0, 0, 0, 10, 0]),
    }

    result = analyzer.analyze(velocities, times)

    assert "Pelvis->Torso" in result.timing_gaps
    gap = result.timing_gaps["Pelvis->Torso"]
    assert np.isclose(gap, 0.2)


def test_extract_velocities_from_recorder(analyzer):
    """Test helper extraction method."""

    class MockRecorder:
        def get_time_series(self, key):
            if key == "joint_velocities":
                times = [0.0, 1.0]
                # 2 frames, 3 joints
                vels = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
                return times, vels
            return [], []

    recorder = MockRecorder()
    indices = {"JointA": 0, "JointC": 2}

    vels, times = analyzer.extract_velocities_from_recorder(recorder, indices)

    assert np.array_equal(times, [0.0, 1.0])
    assert "JointA" in vels
    assert np.array_equal(vels["JointA"], [1.0, 4.0])
    assert "JointC" in vels
    assert np.array_equal(vels["JointC"], [3.0, 6.0])
