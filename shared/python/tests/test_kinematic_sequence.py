import numpy as np

from shared.python.kinematic_sequence import KinematicSequenceAnalyzer


class TestKinematicSequence:
    def test_perfect_sequence(self) -> None:
        """Test a perfectly ordered kinematic sequence."""
        # Pelvis -> Torso -> Arm -> Club
        times = np.linspace(0, 1, 100)

        # Create peaks at different times
        # 0.2, 0.4, 0.6, 0.8

        from typing import cast
        def gaussian(t: np.ndarray, center: float, width: float = 0.1) -> np.ndarray:
            return cast(np.ndarray, np.exp(-((t - center) ** 2) / (2 * width**2)))

        vel_pelvis = gaussian(times, 0.2)
        vel_torso = gaussian(times, 0.4)
        vel_arm = gaussian(times, 0.6)
        vel_club = gaussian(times, 0.8)

        data = {
            "Pelvis": vel_pelvis,
            "Torso": vel_torso,
            "Arm": vel_arm,
            "Club": vel_club,
        }

        analyzer = KinematicSequenceAnalyzer(
            expected_order=["Pelvis", "Torso", "Arm", "Club"]
        )
        result = analyzer.analyze(data, times)

        assert result.is_valid_sequence
        assert result.efficiency_score == 1.0
        assert result.sequence_order == ["Pelvis", "Torso", "Arm", "Club"]
        assert len(result.peaks) == 4
        assert result.peaks[3].name == "Club"  # Last peak

        # Check timing gaps
        assert "Pelvis->Torso" in result.timing_gaps
        gap = result.timing_gaps["Pelvis->Torso"]
        assert abs(gap - 0.2) < 0.05

    def test_out_of_order_sequence(self) -> None:
        """Test a sequence that is out of order (e.g., casting)."""
        # Pelvis -> Arm -> Torso -> Club
        # Arm fires before Torso
        times = np.linspace(0, 1, 100)

        vel_pelvis = np.zeros(100)
        vel_pelvis[20] = 10.0  # t=0.2

        vel_arm = np.zeros(100)
        vel_arm[40] = 10.0  # t=0.4

        vel_torso = np.zeros(100)
        vel_torso[60] = 10.0  # t=0.6

        vel_club = np.zeros(100)
        vel_club[80] = 10.0  # t=0.8

        data = {
            "Pelvis": vel_pelvis,
            "Torso": vel_torso,
            "Arm": vel_arm,
            "Club": vel_club,
        }

        analyzer = KinematicSequenceAnalyzer(
            expected_order=["Pelvis", "Torso", "Arm", "Club"]
        )
        result = analyzer.analyze(data, times)

        assert not result.is_valid_sequence
        assert result.efficiency_score < 1.0
        assert result.sequence_order == ["Pelvis", "Arm", "Torso", "Club"]

    def test_missing_data(self) -> None:
        """Test with empty data."""
        times = np.linspace(0, 1, 100)
        data = {"Pelvis": np.array([]), "Torso": np.array([])}

        analyzer = KinematicSequenceAnalyzer()
        result = analyzer.analyze(data, times)

        assert len(result.peaks) == 0
        assert result.efficiency_score == 0.0

    def test_subset_of_segments(self) -> None:
        """Test with only a subset of expected segments."""
        times = np.linspace(0, 1, 100)
        vel_pelvis = np.zeros(100)
        vel_pelvis[20] = 1.0
        vel_club = np.zeros(100)
        vel_club[80] = 1.0

        data = {"Pelvis": vel_pelvis, "Club": vel_club}

        analyzer = KinematicSequenceAnalyzer(
            expected_order=["Pelvis", "Torso", "Arm", "Club"]
        )
        result = analyzer.analyze(data, times)

        assert result.is_valid_sequence  # Subsets that respect order are valid
        assert result.efficiency_score == 1.0
        assert result.sequence_order == ["Pelvis", "Club"]

    def test_extract_velocities_helper(self) -> None:
        """Test extraction helper."""

        class MockRecorder:
            def get_time_series(self, key: str) -> tuple[np.ndarray, np.ndarray]:
                if key == "joint_velocities":
                    # 100 samples, 4 joints
                    return np.linspace(0, 1, 100), np.ones((100, 4))
                return np.array([]), np.array([])

        recorder = MockRecorder()
        indices = {"Pelvis": 0, "Torso": 1}

        analyzer = KinematicSequenceAnalyzer()
        data, times = analyzer.extract_velocities_from_recorder(recorder, indices)

        assert len(times) == 100
        assert "Pelvis" in data
        assert "Torso" in data
        assert len(data["Pelvis"]) == 100
