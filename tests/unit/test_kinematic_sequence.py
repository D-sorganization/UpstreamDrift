"""Unit tests for kinematic sequence analysis."""

import numpy as np

from src.shared.python.kinematic_sequence import (
    KinematicSequenceAnalyzer,
    SegmentTimingAnalyzer,
)


class MockRecorder:
    def __init__(self, times, velocities):
        self.times = times
        self.velocities = velocities

    def get_time_series(self, name):
        if name == "joint_velocities":
            return self.times, self.velocities
        return [], []


def test_kinematic_sequence_ideal():
    """Test analysis of an ideal segment timing sequence."""
    times = np.linspace(0, 1.0, 100)

    proximal_vel = np.exp(-((times - 0.2) ** 2) / 0.01) * 10
    mid_proximal_vel = np.exp(-((times - 0.3) ** 2) / 0.01) * 15
    mid_distal_vel = np.exp(-((times - 0.4) ** 2) / 0.01) * 20
    distal_vel = np.exp(-((times - 0.5) ** 2) / 0.01) * 30

    data = {
        "proximal": proximal_vel,
        "mid_proximal": mid_proximal_vel,
        "mid_distal": mid_distal_vel,
        "distal": distal_vel,
    }

    analyzer = KinematicSequenceAnalyzer(
        expected_order=["proximal", "mid_proximal", "mid_distal", "distal"]
    )
    result = analyzer.analyze(data, times)

    assert result.is_valid_sequence
    assert result.sequence_consistency == 1.0
    assert result.sequence_order == ["proximal", "mid_proximal", "mid_distal", "distal"]

    assert len(result.peaks) == 4
    assert result.peaks[0].name == "proximal"
    assert np.isclose(result.peaks[0].time, 0.2, atol=0.02)
    assert result.peaks[3].name == "distal"
    assert np.isclose(result.peaks[3].time, 0.5, atol=0.02)


def test_kinematic_sequence_out_of_order():
    """Test analysis of an incorrect sequence."""
    times = np.linspace(0, 1.0, 100)

    proximal_vel = np.exp(-((times - 0.2) ** 2) / 0.01) * 10
    distal_vel = np.exp(-((times - 0.25) ** 2) / 0.01) * 30
    mid_proximal_vel = np.exp(-((times - 0.3) ** 2) / 0.01) * 15
    mid_distal_vel = np.exp(-((times - 0.4) ** 2) / 0.01) * 20

    data = {
        "proximal": proximal_vel,
        "mid_proximal": mid_proximal_vel,
        "mid_distal": mid_distal_vel,
        "distal": distal_vel,
    }

    analyzer = KinematicSequenceAnalyzer(
        expected_order=["proximal", "mid_proximal", "mid_distal", "distal"]
    )
    result = analyzer.analyze(data, times)

    assert not result.is_valid_sequence
    assert result.sequence_consistency < 1.0
    assert np.isclose(result.sequence_consistency, 4 / 6, atol=0.01)
    assert result.sequence_order == [
        "proximal",
        "distal",
        "mid_proximal",
        "mid_distal",
    ]


def test_extract_velocities():
    """Test extraction helper."""
    times = np.array([0, 1, 2])
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
    analyzer = SegmentTimingAnalyzer()
    result = analyzer.analyze({}, np.array([]))

    assert len(result.peaks) == 0
    assert result.sequence_consistency == 0.0


def test_backward_compat_alias():
    """KinematicSequenceAnalyzer should be an alias for SegmentTimingAnalyzer."""
    assert KinematicSequenceAnalyzer is SegmentTimingAnalyzer


def test_no_expected_order_peaks_only():
    """Without expected_order, only peak detection should be performed."""
    times = np.linspace(0, 1.0, 200)
    seg_a = np.exp(-((times - 0.2) ** 2) / 0.005) * 10
    seg_b = np.exp(-((times - 0.4) ** 2) / 0.005) * 20

    data = {"SegA": seg_a, "SegB": seg_b}
    analyzer = SegmentTimingAnalyzer(expected_order=None)
    result = analyzer.analyze(data, times)

    assert len(result.peaks) == 2
    assert result.sequence_consistency == 0.0
    assert result.expected_order is None

    for peak in result.peaks:
        assert peak.speed_gain is None

    for peak in result.peaks:
        assert peak.deceleration_rate is not None


class TestSpeedGain:
    """Tests for speed gain metric."""

    def test_speed_gain_computed(self) -> None:
        """Speed gain should be calculated for distal segments."""
        times = np.linspace(0, 1.0, 200)
        proximal_vel = np.exp(-((times - 0.2) ** 2) / 0.005) * 10
        mid_proximal_vel = np.exp(-((times - 0.3) ** 2) / 0.005) * 15
        mid_distal_vel = np.exp(-((times - 0.4) ** 2) / 0.005) * 20
        distal_vel = np.exp(-((times - 0.5) ** 2) / 0.005) * 30

        data = {
            "proximal": proximal_vel,
            "mid_proximal": mid_proximal_vel,
            "mid_distal": mid_distal_vel,
            "distal": distal_vel,
        }
        analyzer = KinematicSequenceAnalyzer(
            expected_order=["proximal", "mid_proximal", "mid_distal", "distal"]
        )
        result = analyzer.analyze(data, times)

        peak_map = {p.name: p for p in result.peaks}

        assert peak_map["proximal"].speed_gain is None

        assert peak_map["mid_proximal"].speed_gain is not None
        assert peak_map["mid_proximal"].speed_gain == np.testing.assert_approx_equal(
            peak_map["mid_proximal"].speed_gain, 1.5, significant=2
        ) or np.isclose(peak_map["mid_proximal"].speed_gain, 1.5, rtol=0.05)

        assert peak_map["mid_distal"].speed_gain is not None
        assert np.isclose(peak_map["mid_distal"].speed_gain, 20 / 15, rtol=0.05)

        assert peak_map["distal"].speed_gain is not None
        assert np.isclose(peak_map["distal"].speed_gain, 30 / 20, rtol=0.05)

    def test_speed_gain_increases_distally(self) -> None:
        """In a good sequence, speed gain should be > 1 for distal segments."""
        times = np.linspace(0, 1.0, 200)
        proximal_vel = np.exp(-((times - 0.2) ** 2) / 0.005) * 8
        mid_proximal_vel = np.exp(-((times - 0.3) ** 2) / 0.005) * 14
        mid_distal_vel = np.exp(-((times - 0.4) ** 2) / 0.005) * 22
        distal_vel = np.exp(-((times - 0.5) ** 2) / 0.005) * 38

        data = {
            "proximal": proximal_vel,
            "mid_proximal": mid_proximal_vel,
            "mid_distal": mid_distal_vel,
            "distal": distal_vel,
        }
        analyzer = KinematicSequenceAnalyzer(
            expected_order=["proximal", "mid_proximal", "mid_distal", "distal"]
        )
        result = analyzer.analyze(data, times)

        peak_map = {p.name: p for p in result.peaks}

        for name in ["mid_proximal", "mid_distal", "distal"]:
            assert peak_map[name].speed_gain is not None
            assert peak_map[name].speed_gain > 1.0, (
                f"{name} speed gain should be > 1.0, got {peak_map[name].speed_gain}"
            )

    def test_speed_gain_missing_segment(self) -> None:
        """Speed gain should handle missing segments gracefully."""
        times = np.linspace(0, 1.0, 200)
        proximal_vel = np.exp(-((times - 0.2) ** 2) / 0.005) * 10
        distal_vel = np.exp(-((times - 0.5) ** 2) / 0.005) * 30

        data = {"proximal": proximal_vel, "distal": distal_vel}
        analyzer = KinematicSequenceAnalyzer(
            expected_order=["proximal", "mid_proximal", "mid_distal", "distal"]
        )
        result = analyzer.analyze(data, times)

        peak_map = {p.name: p for p in result.peaks}
        assert peak_map["distal"].speed_gain is None


class TestDecelerationRate:
    """Tests for deceleration rate metric."""

    def test_deceleration_computed(self) -> None:
        """Deceleration rate should be computed for all segments."""
        times = np.linspace(0, 1.0, 1000)
        proximal_vel = np.exp(-((times - 0.2) ** 2) / 0.005) * 10
        mid_proximal_vel = np.exp(-((times - 0.3) ** 2) / 0.005) * 15
        mid_distal_vel = np.exp(-((times - 0.4) ** 2) / 0.005) * 20
        distal_vel = np.exp(-((times - 0.5) ** 2) / 0.005) * 30

        data = {
            "proximal": proximal_vel,
            "mid_proximal": mid_proximal_vel,
            "mid_distal": mid_distal_vel,
            "distal": distal_vel,
        }
        analyzer = KinematicSequenceAnalyzer(
            expected_order=["proximal", "mid_proximal", "mid_distal", "distal"]
        )
        result = analyzer.analyze(data, times)

        peak_map = {p.name: p for p in result.peaks}

        for name in ["proximal", "mid_proximal", "mid_distal", "distal"]:
            assert peak_map[name].deceleration_rate is not None, (
                f"{name} should have deceleration_rate computed"
            )
            assert peak_map[name].deceleration_rate > 0, (
                f"{name} deceleration_rate should be positive"
            )

    def test_proximal_decelerates_faster(self) -> None:
        """Proximal segments should decelerate faster (braking effect)."""
        times = np.linspace(0, 1.0, 1000)

        proximal_vel = np.exp(-((times - 0.2) ** 2) / 0.002) * 10
        mid_proximal_vel = np.exp(-((times - 0.3) ** 2) / 0.003) * 15
        mid_distal_vel = np.exp(-((times - 0.4) ** 2) / 0.004) * 20
        distal_vel = np.exp(-((times - 0.5) ** 2) / 0.010) * 30

        data = {
            "proximal": proximal_vel,
            "mid_proximal": mid_proximal_vel,
            "mid_distal": mid_distal_vel,
            "distal": distal_vel,
        }
        analyzer = KinematicSequenceAnalyzer(
            expected_order=["proximal", "mid_proximal", "mid_distal", "distal"]
        )
        result = analyzer.analyze(data, times)

        peak_map = {p.name: p for p in result.peaks}

        assert peak_map["proximal"].deceleration_rate is not None
        assert peak_map["distal"].deceleration_rate is not None
        assert (
            peak_map["proximal"].deceleration_rate
            > peak_map["distal"].deceleration_rate
        )

    def test_deceleration_units(self) -> None:
        """Deceleration rate should have correct magnitude."""
        times = np.linspace(0, 1.0, 1000)
        proximal_vel = np.exp(-((times - 0.2) ** 2) / 0.005) * 10

        data = {"proximal": proximal_vel}
        analyzer = KinematicSequenceAnalyzer(expected_order=["proximal"])
        result = analyzer.analyze(data, times)

        peak = result.peaks[0]
        assert peak.deceleration_rate is not None
        assert 0 < peak.deceleration_rate < 1e6
