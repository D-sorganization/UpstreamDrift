import numpy as np

from src.shared.python.kinematic_sequence import KinematicSequenceAnalyzer


class TestKinematicSequence:
    def test_perfect_sequence(self) -> None:
        """Test a perfectly ordered segment timing sequence."""
        # proximal -> mid_proximal -> mid_distal -> distal
        times = np.linspace(0, 1, 100)

        from typing import cast

        def gaussian(t: np.ndarray, center: float, width: float = 0.1) -> np.ndarray:
            return cast(np.ndarray, np.exp(-((t - center) ** 2) / (2 * width**2)))

        vel_proximal = gaussian(times, 0.2)
        vel_mid_proximal = gaussian(times, 0.4)
        vel_mid_distal = gaussian(times, 0.6)
        vel_distal = gaussian(times, 0.8)

        data = {
            "proximal": vel_proximal,
            "mid_proximal": vel_mid_proximal,
            "mid_distal": vel_mid_distal,
            "distal": vel_distal,
        }

        analyzer = KinematicSequenceAnalyzer(
            expected_order=["proximal", "mid_proximal", "mid_distal", "distal"]
        )
        result = analyzer.analyze(data, times)

        assert result.is_valid_sequence
        assert result.sequence_consistency == 1.0
        assert result.sequence_order == [
            "proximal",
            "mid_proximal",
            "mid_distal",
            "distal",
        ]
        assert len(result.peaks) == 4
        assert result.peaks[3].name == "distal"  # Last peak

        # Check timing gaps
        assert "proximal->mid_proximal" in result.timing_gaps
        gap = result.timing_gaps["proximal->mid_proximal"]
        assert abs(gap - 0.2) < 0.05

    def test_out_of_order_sequence(self) -> None:
        """Test a sequence that is out of order."""
        times = np.linspace(0, 1, 100)

        vel_proximal = np.zeros(100)
        vel_proximal[20] = 10.0

        vel_mid_distal = np.zeros(100)
        vel_mid_distal[40] = 10.0

        vel_mid_proximal = np.zeros(100)
        vel_mid_proximal[60] = 10.0

        vel_distal = np.zeros(100)
        vel_distal[80] = 10.0

        data = {
            "proximal": vel_proximal,
            "mid_proximal": vel_mid_proximal,
            "mid_distal": vel_mid_distal,
            "distal": vel_distal,
        }

        analyzer = KinematicSequenceAnalyzer(
            expected_order=["proximal", "mid_proximal", "mid_distal", "distal"]
        )
        result = analyzer.analyze(data, times)

        assert not result.is_valid_sequence
        assert result.sequence_consistency < 1.0
        assert result.sequence_order == [
            "proximal",
            "mid_distal",
            "mid_proximal",
            "distal",
        ]

    def test_missing_data(self) -> None:
        """Test with empty data."""
        times = np.linspace(0, 1, 100)
        data = {"proximal": np.array([]), "mid_proximal": np.array([])}

        analyzer = KinematicSequenceAnalyzer()
        result = analyzer.analyze(data, times)

        assert len(result.peaks) == 0
        assert result.sequence_consistency == 0.0

    def test_subset_of_segments(self) -> None:
        """Test with only a subset of expected segments."""
        times = np.linspace(0, 1, 100)
        vel_proximal = np.zeros(100)
        vel_proximal[20] = 1.0
        vel_distal = np.zeros(100)
        vel_distal[80] = 1.0

        data = {"proximal": vel_proximal, "distal": vel_distal}

        analyzer = KinematicSequenceAnalyzer(
            expected_order=["proximal", "mid_proximal", "mid_distal", "distal"]
        )
        result = analyzer.analyze(data, times)

        assert result.is_valid_sequence
        assert result.sequence_consistency == 1.0
        assert result.sequence_order == ["proximal", "distal"]

    def test_extract_velocities_helper(self) -> None:
        """Test extraction helper."""

        class MockRecorder:
            def get_time_series(self, key: str) -> tuple[np.ndarray, np.ndarray]:
                if key == "joint_velocities":
                    return np.linspace(0, 1, 100), np.ones((100, 4))
                return np.array([]), np.array([])

        recorder = MockRecorder()
        indices = {"proximal": 0, "mid_proximal": 1}

        analyzer = KinematicSequenceAnalyzer()
        data, times = analyzer.extract_velocities_from_recorder(recorder, indices)

        assert len(times) == 100
        assert "proximal" in data
        assert "mid_proximal" in data
        assert len(data["proximal"]) == 100
