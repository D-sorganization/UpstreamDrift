"""Unit tests for kinematic sequence analysis."""

import numpy as np

from src.shared.python.kinematic_sequence import KinematicSequenceAnalyzer


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
    assert result.sequence_consistency == 1.0
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
    assert result.sequence_consistency < 1.0
    # Expected pairs: P<T, P<A, P<C, T<A, T<C, A<C (6 pairs)
    # Actual: P(0.2), C(0.25), T(0.3), A(0.4)
    # Pairs:
    # P<T (ok), P<A (ok), P<C (ok)
    # T<A (ok), T<C (FAIL: T=0.3 > C=0.25)
    # A<C (FAIL: A=0.4 > C=0.25)
    # Correct pairs: 4/6 = 0.666
    assert np.isclose(result.sequence_consistency, 4 / 6, atol=0.01)
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
    assert result.sequence_consistency == 0.0


# =============================================================================
# Speed Gain Tests (Issue #1083)
# =============================================================================


class TestSpeedGain:
    """Tests for speed gain metric: distal_peak / proximal_peak."""

    def test_speed_gain_computed(self) -> None:
        """Speed gain should be calculated for distal segments."""
        times = np.linspace(0, 1.0, 200)
        pelvis_vel = np.exp(-((times - 0.2) ** 2) / 0.005) * 10
        torso_vel = np.exp(-((times - 0.3) ** 2) / 0.005) * 15
        arm_vel = np.exp(-((times - 0.4) ** 2) / 0.005) * 20
        club_vel = np.exp(-((times - 0.5) ** 2) / 0.005) * 30

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

        peak_map = {p.name: p for p in result.peaks}

        # Pelvis has no proximal → speed_gain should be None
        assert peak_map["Pelvis"].speed_gain is None

        # Torso / Pelvis = 15 / 10 = 1.5
        assert peak_map["Torso"].speed_gain is not None
        assert peak_map["Torso"].speed_gain == np.testing.assert_approx_equal(
            peak_map["Torso"].speed_gain, 1.5, significant=2
        ) or np.isclose(peak_map["Torso"].speed_gain, 1.5, rtol=0.05)

        # Arm / Torso = 20 / 15 ≈ 1.33
        assert peak_map["Arm"].speed_gain is not None
        assert np.isclose(peak_map["Arm"].speed_gain, 20 / 15, rtol=0.05)

        # Club / Arm = 30 / 20 = 1.5
        assert peak_map["Club"].speed_gain is not None
        assert np.isclose(peak_map["Club"].speed_gain, 30 / 20, rtol=0.05)

    def test_speed_gain_increases_distally(self) -> None:
        """In a good sequence, speed gain should be > 1 for distal segments."""
        times = np.linspace(0, 1.0, 200)
        pelvis_vel = np.exp(-((times - 0.2) ** 2) / 0.005) * 8
        torso_vel = np.exp(-((times - 0.3) ** 2) / 0.005) * 14
        arm_vel = np.exp(-((times - 0.4) ** 2) / 0.005) * 22
        club_vel = np.exp(-((times - 0.5) ** 2) / 0.005) * 38

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

        peak_map = {p.name: p for p in result.peaks}

        for name in ["Torso", "Arm", "Club"]:
            assert peak_map[name].speed_gain is not None
            assert (
                peak_map[name].speed_gain > 1.0
            ), f"{name} speed gain should be > 1.0, got {peak_map[name].speed_gain}"

    def test_speed_gain_missing_segment(self) -> None:
        """Speed gain should handle missing segments gracefully."""
        times = np.linspace(0, 1.0, 200)
        pelvis_vel = np.exp(-((times - 0.2) ** 2) / 0.005) * 10
        club_vel = np.exp(-((times - 0.5) ** 2) / 0.005) * 30

        # Missing Torso and Arm
        data = {"Pelvis": pelvis_vel, "Club": club_vel}
        analyzer = KinematicSequenceAnalyzer(
            expected_order=["Pelvis", "Torso", "Arm", "Club"]
        )
        result = analyzer.analyze(data, times)

        peak_map = {p.name: p for p in result.peaks}
        # Club's proximal in expected_order is "Arm", which is missing
        # So speed_gain should be None
        assert peak_map["Club"].speed_gain is None


# =============================================================================
# Deceleration (Braking) Rate Tests (Issue #1083)
# =============================================================================


class TestDecelerationRate:
    """Tests for deceleration rate metric: slope post-peak over ~30ms window."""

    def test_deceleration_computed(self) -> None:
        """Deceleration rate should be computed for all segments."""
        times = np.linspace(0, 1.0, 1000)  # 1ms resolution
        pelvis_vel = np.exp(-((times - 0.2) ** 2) / 0.005) * 10
        torso_vel = np.exp(-((times - 0.3) ** 2) / 0.005) * 15
        arm_vel = np.exp(-((times - 0.4) ** 2) / 0.005) * 20
        club_vel = np.exp(-((times - 0.5) ** 2) / 0.005) * 30

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

        peak_map = {p.name: p for p in result.peaks}

        # All segments should have a positive deceleration rate
        # (velocity decreases after peak → deceleration is positive)
        for name in ["Pelvis", "Torso", "Arm", "Club"]:
            assert (
                peak_map[name].deceleration_rate is not None
            ), f"{name} should have deceleration_rate computed"
            assert (
                peak_map[name].deceleration_rate > 0
            ), f"{name} deceleration_rate should be positive, got {peak_map[name].deceleration_rate}"

    def test_proximal_decelerates_faster(self) -> None:
        """Proximal segments should decelerate faster (braking effect).

        In a proper kinematic chain, proximal segments 'brake' to
        transfer energy to the next segment. This means their
        deceleration rate should be relatively high.
        """
        times = np.linspace(0, 1.0, 1000)

        # Narrow Gaussian = faster deceleration post-peak
        pelvis_vel = np.exp(-((times - 0.2) ** 2) / 0.002) * 10  # Narrow peak
        torso_vel = np.exp(-((times - 0.3) ** 2) / 0.003) * 15
        arm_vel = np.exp(-((times - 0.4) ** 2) / 0.004) * 20
        club_vel = np.exp(-((times - 0.5) ** 2) / 0.010) * 30  # Broad peak

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

        peak_map = {p.name: p for p in result.peaks}

        # Pelvis should decelerate faster than Club
        assert peak_map["Pelvis"].deceleration_rate is not None
        assert peak_map["Club"].deceleration_rate is not None
        assert peak_map["Pelvis"].deceleration_rate > peak_map["Club"].deceleration_rate

    def test_deceleration_units(self) -> None:
        """Deceleration rate should have correct magnitude (velocity/time units)."""
        times = np.linspace(0, 1.0, 1000)
        # Peak velocity ~10 at t=0.2, using Gaussian
        pelvis_vel = np.exp(-((times - 0.2) ** 2) / 0.005) * 10

        data = {"Pelvis": pelvis_vel}
        analyzer = KinematicSequenceAnalyzer(expected_order=["Pelvis"])
        result = analyzer.analyze(data, times)

        peak = result.peaks[0]
        assert peak.deceleration_rate is not None
        # Should be a reasonable magnitude (not 0, not infinity)
        assert 0 < peak.deceleration_rate < 1e6
