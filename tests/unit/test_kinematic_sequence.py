"""Unit tests for kinematic sequence analysis."""

import numpy as np

from shared.python.kinematic_sequence import KinematicSequenceAnalyzer


class MockRecorder:
    def __init__(self, times, velocities):
        self.times = times
        self.velocities = velocities

    def get_time_series(self, name):
        if name == "joint_velocities":
            return self.times, self.velocities
        return [], []


def test_kinematic_sequence_ideal():
    """Test analysis of an ideal kinematic sequence."""
    times = np.linspace(0, 1.0, 100)

    # Create synthetic peaks
    # Pelvis peak at t=0.2, Torso at 0.3, Arm at 0.4, Club at 0.5
    pelvis_vel = np.exp(-((times - 0.2) ** 2) / 0.01) * 10
    torso_vel = np.exp(-((times - 0.3) ** 2) / 0.01) * 15
    arm_vel = np.exp(-((times - 0.4) ** 2) / 0.01) * 20
    club_vel = np.exp(-((times - 0.5) ** 2) / 0.01) * 30

    data = {
        "Pelvis": pelvis_vel,
        "Torso": torso_vel,
        "Arm": arm_vel,
        "Club": club_vel,
    }

    analyzer = KinematicSequenceAnalyzer(
        expected_order=["Pelvis", "Torso", "Arm", "Club"]
    )
    result = analyzer.analyze(data, times)

    assert result.is_valid_sequence
    assert result.efficiency_score == 1.0
    assert result.sequence_order == ["Pelvis", "Torso", "Arm", "Club"]

    # Check peaks
    assert len(result.peaks) == 4
    assert result.peaks[0].name == "Pelvis"
    assert np.isclose(result.peaks[0].time, 0.2, atol=0.02)
    assert result.peaks[3].name == "Club"
    assert np.isclose(result.peaks[3].time, 0.5, atol=0.02)


def test_kinematic_sequence_out_of_order():
    """Test analysis of an incorrect sequence."""
    times = np.linspace(0, 1.0, 100)

    # Club peaks too early (casting)
    pelvis_vel = np.exp(-((times - 0.2) ** 2) / 0.01) * 10
    club_vel = np.exp(-((times - 0.25) ** 2) / 0.01) * 30  # Early club
    torso_vel = np.exp(-((times - 0.3) ** 2) / 0.01) * 15
    arm_vel = np.exp(-((times - 0.4) ** 2) / 0.01) * 20

    data = {
        "Pelvis": pelvis_vel,
        "Torso": torso_vel,
        "Arm": arm_vel,
        "Club": club_vel,
    }

    analyzer = KinematicSequenceAnalyzer(
        expected_order=["Pelvis", "Torso", "Arm", "Club"]
    )
    result = analyzer.analyze(data, times)

    assert not result.is_valid_sequence
    assert result.efficiency_score < 1.0
    # Expected pairs: P<T, P<A, P<C, T<A, T<C, A<C (6 pairs)
    # Actual: P(0.2), C(0.25), T(0.3), A(0.4)
    # Pairs:
    # P<T (ok), P<A (ok), P<C (ok)
    # T<A (ok), T<C (FAIL: T=0.3 > C=0.25)
    # A<C (FAIL: A=0.4 > C=0.25)
    # Correct pairs: 4/6 = 0.666
    assert np.isclose(result.efficiency_score, 4 / 6, atol=0.01)
    assert result.sequence_order == ["Pelvis", "Club", "Torso", "Arm"]


def test_extract_velocities():
    """Test extraction helper."""
    times = np.array([0, 1, 2])
    # 2 joints
    vels = np.array([[1, 10], [2, 20], [3, 30]])

    recorder = MockRecorder(times, vels)
    indices = {"JointA": 0, "JointB": 1}

    analyzer = KinematicSequenceAnalyzer()
    data, t_out = analyzer.extract_velocities_from_recorder(recorder, indices)

    assert np.array_equal(t_out, times)
    assert "JointA" in data
    assert np.array_equal(data["JointA"], np.array([1, 2, 3]))
    assert "JointB" in data
    assert np.array_equal(data["JointB"], np.array([10, 20, 30]))


def test_empty_data():
    """Test handling empty data."""
    analyzer = KinematicSequenceAnalyzer()
    result = analyzer.analyze({}, np.array([]))

    assert len(result.peaks) == 0
    assert result.efficiency_score == 0.0
