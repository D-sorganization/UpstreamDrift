"""Tests for motion training module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import test subjects
from motion_training.club_trajectory_parser import (
    ClubFrame,
    ClubTrajectory,
    ClubTrajectoryParser,
    SwingEventMarkers,
    compute_hand_positions,
)


class TestClubFrame:
    """Tests for ClubFrame dataclass."""

    def test_create_frame(self):
        """Test creating a club frame."""
        frame = ClubFrame(
            time=0.0,
            sample_index=1,
            grip_position=np.array([0.0, 0.5, 0.8]),
            grip_rotation=np.eye(3),
            club_face_position=np.array([0.0, 0.5, -0.2]),
            club_face_rotation=np.eye(3),
        )
        assert frame.time == 0.0
        assert frame.sample_index == 1
        np.testing.assert_array_equal(frame.grip_position, [0.0, 0.5, 0.8])


class TestClubTrajectory:
    """Tests for ClubTrajectory."""

    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory for testing."""
        frames = []
        for i in range(10):
            t = i * 0.01
            frames.append(
                ClubFrame(
                    time=t,
                    sample_index=i,
                    grip_position=np.array([0.0, 0.5 + 0.01 * i, 0.8]),
                    grip_rotation=np.eye(3),
                    club_face_position=np.array([0.0, 0.5 + 0.01 * i, -0.2]),
                    club_face_rotation=np.eye(3),
                )
            )
        return ClubTrajectory(
            frames=frames,
            events=SwingEventMarkers(address=0, top=3, impact=6, finish=9),
        )

    def test_num_frames(self, sample_trajectory):
        """Test frame count."""
        assert sample_trajectory.num_frames == 10

    def test_duration(self, sample_trajectory):
        """Test duration calculation."""
        assert sample_trajectory.duration == pytest.approx(0.09)

    def test_times_property(self, sample_trajectory):
        """Test times array."""
        times = sample_trajectory.times
        assert len(times) == 10
        assert times[0] == 0.0
        assert times[-1] == pytest.approx(0.09)

    def test_grip_positions_property(self, sample_trajectory):
        """Test grip positions array."""
        positions = sample_trajectory.grip_positions
        assert positions.shape == (10, 3)

    def test_get_event_frame(self, sample_trajectory):
        """Test getting event frames."""
        impact = sample_trajectory.get_event_frame("impact")
        assert impact is not None
        assert impact.sample_index == 6

    def test_get_frame_at_time(self, sample_trajectory):
        """Test interpolation."""
        # Get frame at middle time
        frame = sample_trajectory.get_frame_at_time(0.045)
        assert frame.time == pytest.approx(0.045)

        # Before start
        frame = sample_trajectory.get_frame_at_time(-1.0)
        assert frame.time == 0.0

        # After end
        frame = sample_trajectory.get_frame_at_time(1.0)
        assert frame.time == pytest.approx(0.09)


class TestComputeHandPositions:
    """Tests for hand position computation."""

    def test_default_offsets(self):
        """Test hand positions with default offsets."""
        frame = ClubFrame(
            time=0.0,
            sample_index=1,
            grip_position=np.array([0.0, 0.0, 1.0]),
            grip_rotation=np.eye(3),  # Identity - Z points up
            club_face_position=np.array([0.0, 0.0, 0.0]),
            club_face_rotation=np.eye(3),
        )

        left, right = compute_hand_positions(frame)

        # With identity rotation, Z is up
        # Left hand should be 4cm higher (default offset)
        np.testing.assert_array_almost_equal(left, [0.0, 0.0, 1.04])
        np.testing.assert_array_almost_equal(right, [0.0, 0.0, 0.96])

    def test_custom_offsets(self):
        """Test hand positions with custom offsets."""
        frame = ClubFrame(
            time=0.0,
            sample_index=1,
            grip_position=np.array([0.0, 0.0, 1.0]),
            grip_rotation=np.eye(3),
            club_face_position=np.array([0.0, 0.0, 0.0]),
            club_face_rotation=np.eye(3),
        )

        left, right = compute_hand_positions(frame, left_offset=0.1, right_offset=-0.1)

        np.testing.assert_array_almost_equal(left, [0.0, 0.0, 1.1])
        np.testing.assert_array_almost_equal(right, [0.0, 0.0, 0.9])


class TestClubTrajectoryParser:
    """Tests for trajectory parser."""

    def test_init_with_invalid_path(self):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            ClubTrajectoryParser("/nonexistent/path.xlsx")

    @pytest.fixture
    def mock_excel_data(self):
        """Create mock Excel data for testing."""
        return np.array([
            ["Wiffle ball", None, "A=", 1, "T=", 3, "I=", 5, "F=", 8, "CHS", 100.0, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            [None, None, "Mid-hands", None, None, None, None, None, None, None, None, None, None, None, "Center of club face", None, None, None, None, None, None, None, None, None, None, None],
            ["Sample #", "Time", "X", "Y", "Z", "Xx", "Xy", "Xz", "Yx", "Yy", "Yz", "Zx", "Zy", "Zz", "X", "Y", "Z", "Xx", "Xy", "Xz", "Yx", "Yy", "Yz", "Zx", "Zy", "Zz"],
            [1, 0.0, 0.0, 50.0, 100.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, None, None, None, 0.0, 50.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, None, None, None],
        ])


class TestSwingEventMarkers:
    """Tests for SwingEventMarkers."""

    def test_default_values(self):
        """Test default marker values."""
        markers = SwingEventMarkers()
        assert markers.address == 0
        assert markers.top == 0
        assert markers.impact == 0
        assert markers.finish == 0
        assert markers.club_head_speed == 0.0

    def test_custom_values(self):
        """Test custom marker values."""
        markers = SwingEventMarkers(
            address=10,
            top=50,
            impact=100,
            finish=150,
            club_head_speed=115.5,
        )
        assert markers.address == 10
        assert markers.top == 50
        assert markers.impact == 100
        assert markers.finish == 150
        assert markers.club_head_speed == 115.5


# IK Solver tests
class TestDualHandIKSolver:
    """Tests for dual-hand IK solver."""

    @pytest.fixture
    def mock_model(self):
        """Create mock Pinocchio model."""
        pin = pytest.importorskip("pinocchio")

        # Use sample manipulator for testing
        model = pin.buildSampleModelManipulator()
        return model

    def test_ik_solver_import(self):
        """Test that IK solver can be imported."""
        try:
            from motion_training.dual_hand_ik_solver import (
                DualHandIKSolver,
                IKSolverSettings,
                create_ik_solver,
            )
        except ImportError as e:
            pytest.skip(f"IK solver dependencies not available: {e}")

    def test_ik_settings_defaults(self):
        """Test IK settings have reasonable defaults."""
        from motion_training.dual_hand_ik_solver import IKSolverSettings

        settings = IKSolverSettings()
        assert settings.solver == "quadprog"
        assert settings.damping > 0
        assert settings.dt > 0
        assert settings.max_iterations > 0
        assert settings.left_hand_offset != 0  # Should have some offset
        assert settings.right_hand_offset != 0


class TestTrajectoryExporter:
    """Tests for trajectory exporter."""

    @pytest.fixture
    def sample_ik_result(self):
        """Create sample IK result for testing."""
        from motion_training.dual_hand_ik_solver import TrajectoryIKResult

        result = TrajectoryIKResult()
        for i in range(10):
            result.configurations.append(np.random.randn(12))
            result.times.append(i * 0.01)
            result.left_hand_errors.append(0.001 * np.random.rand())
            result.right_hand_errors.append(0.001 * np.random.rand())
        result.convergence_rate = 0.9
        return result

    def test_exporter_init(self, sample_ik_result):
        """Test exporter initialization."""
        from motion_training.trajectory_exporter import TrajectoryExporter

        exporter = TrajectoryExporter(sample_ik_result)
        assert exporter.num_frames == 10
        assert exporter.num_dof == 12

    def test_export_csv(self, sample_ik_result, tmp_path):
        """Test CSV export."""
        from motion_training.trajectory_exporter import TrajectoryExporter

        exporter = TrajectoryExporter(sample_ik_result)
        output_path = exporter.export(tmp_path / "test", format="csv")

        assert output_path.exists()
        assert output_path.suffix == ".csv"

    def test_export_npz(self, sample_ik_result, tmp_path):
        """Test NPZ export."""
        from motion_training.trajectory_exporter import TrajectoryExporter

        exporter = TrajectoryExporter(sample_ik_result)
        output_path = exporter.export(tmp_path / "test", format="npz")

        assert output_path.exists()
        assert output_path.suffix == ".npz"

        # Verify contents
        data = np.load(output_path)
        assert "q" in data
        assert "times" in data

    def test_export_json(self, sample_ik_result, tmp_path):
        """Test JSON export."""
        from motion_training.trajectory_exporter import TrajectoryExporter
        import json

        exporter = TrajectoryExporter(sample_ik_result)
        output_path = exporter.export(tmp_path / "test", format="json")

        assert output_path.exists()
        assert output_path.suffix == ".json"

        # Verify JSON is valid
        with open(output_path) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "trajectory" in data

    def test_export_unsupported_format(self, sample_ik_result, tmp_path):
        """Test that unsupported format raises error."""
        from motion_training.trajectory_exporter import TrajectoryExporter

        exporter = TrajectoryExporter(sample_ik_result)
        with pytest.raises(ValueError):
            exporter.export(tmp_path / "test", format="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
