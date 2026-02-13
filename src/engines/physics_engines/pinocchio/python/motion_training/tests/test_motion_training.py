"""Tests for motion training module."""

from __future__ import annotations

import typing

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

    def test_create_frame(self) -> None:
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
    def sample_trajectory(self) -> ClubTrajectory:
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

    def test_num_frames(self, sample_trajectory) -> None:
        """Test frame count."""
        assert sample_trajectory.num_frames == 10

    def test_duration(self, sample_trajectory) -> None:
        """Test duration calculation."""
        assert sample_trajectory.duration == pytest.approx(0.09)

    def test_times_property(self, sample_trajectory) -> None:
        """Test times array."""
        times = sample_trajectory.times
        assert len(times) == 10
        assert times[0] == 0.0
        assert times[-1] == pytest.approx(0.09)

    def test_grip_positions_property(self, sample_trajectory) -> None:
        """Test grip positions array."""
        positions = sample_trajectory.grip_positions
        assert positions.shape == (10, 3)

    def test_get_event_frame(self, sample_trajectory) -> None:
        """Test getting event frames."""
        impact = sample_trajectory.get_event_frame("impact")
        assert impact is not None
        assert impact.sample_index == 6

    def test_get_frame_at_time(self, sample_trajectory) -> None:
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

    def test_default_offsets(self) -> None:
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

    def test_custom_offsets(self) -> None:
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


_NUM_COLUMNS = 26  # Total columns in the mock Excel data


def _pad_row(values: list[object]) -> list[object]:
    """Pad a row with None values to reach the expected column count.

    Args:
        values: The non-None prefix values for the row.

    Returns:
        A list of exactly ``_NUM_COLUMNS`` elements.
    """
    return values + [None] * (_NUM_COLUMNS - len(values))


def _make_metadata_header_row() -> list[object]:
    """Build the first row: shot label, swing event indices, and club head speed.

    Returns:
        A list representing the metadata header row of the Excel sheet.
    """
    return _pad_row(
        ["Wiffle ball", None, "A=", 1, "T=", 3, "I=", 5, "F=", 8, "CHS", 100.0]
    )


def _make_sensor_group_row() -> list[object]:
    """Build the second row: sensor-group labels (Mid-hands, Center of club face).

    Returns:
        A list representing the sensor-group sub-header row.
    """
    row: list[object] = [None] * _NUM_COLUMNS
    row[2] = "Mid-hands"
    row[14] = "Center of club face"
    return row


def _make_column_names_row() -> list[object]:
    """Build the third row: column names for sample data.

    Returns:
        A list of column header strings (Sample #, Time, X/Y/Z, rotation components).
    """
    prefix = ["Sample #", "Time"]
    position_and_rotation = [
        "X",
        "Y",
        "Z",
        "Xx",
        "Xy",
        "Xz",
        "Yx",
        "Yy",
        "Yz",
        "Zx",
        "Zy",
        "Zz",
    ]
    return prefix + position_and_rotation + position_and_rotation


def _make_sample_data_row() -> list[object]:
    """Build a single data row with sample index 1 at time 0.

    The grip (Mid-hands) is at (0, 50, 100) mm with partial identity rotation.
    The club face is at (0, 50, 0) mm with partial identity rotation.

    Returns:
        A list representing one frame of motion capture data.
    """
    sample_index = 1
    time = 0.0

    grip_position = [0.0, 50.0, 100.0]
    grip_rotation_partial = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, None, None, None]

    face_position = [0.0, 50.0, 0.0]
    face_rotation_partial = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, None, None, None]

    return (
        [sample_index, time]
        + grip_position
        + grip_rotation_partial
        + face_position
        + face_rotation_partial
    )


class TestClubTrajectoryParser:
    """Tests for trajectory parser."""

    def test_init_with_invalid_path(self) -> None:
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            ClubTrajectoryParser("/nonexistent/path.xlsx")

    @pytest.fixture
    def mock_excel_data(self) -> np.ndarray:
        """Create mock Excel data for testing.

        Returns a numpy array with 4 rows: metadata header, sensor-group
        sub-header, column names, and one data sample.
        """
        return np.array(
            [
                _make_metadata_header_row(),
                _make_sensor_group_row(),
                _make_column_names_row(),
                _make_sample_data_row(),
            ]
        )


class TestSwingEventMarkers:
    """Tests for SwingEventMarkers."""

    def test_default_values(self) -> None:
        """Test default marker values."""
        markers = SwingEventMarkers()
        assert markers.address == 0
        assert markers.top == 0
        assert markers.impact == 0
        assert markers.finish == 0
        assert markers.club_head_speed == 0.0

    def test_custom_values(self) -> None:
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
    def mock_model(self) -> typing.Any:
        """Create mock Pinocchio model."""
        pin = pytest.importorskip("pinocchio")

        # Use sample manipulator for testing
        model = pin.buildSampleModelManipulator()
        return model

    def test_ik_solver_import(self) -> None:
        """Test that IK solver can be imported."""
        try:
            pass
        except ImportError as e:
            pytest.skip(f"IK solver dependencies not available: {e}")

    def test_ik_settings_defaults(self) -> None:
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
    def sample_ik_result(self) -> typing.Any:
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

    def test_exporter_init(self, sample_ik_result) -> None:
        """Test exporter initialization."""
        from motion_training.trajectory_exporter import TrajectoryExporter

        exporter = TrajectoryExporter(sample_ik_result)
        assert exporter.num_frames == 10
        assert exporter.num_dof == 12

    def test_export_csv(self, sample_ik_result, tmp_path) -> None:
        """Test CSV export."""
        from motion_training.trajectory_exporter import TrajectoryExporter

        exporter = TrajectoryExporter(sample_ik_result)
        output_path = exporter.export(tmp_path / "test", format="csv")

        assert output_path.exists()
        assert output_path.suffix == ".csv"

    def test_export_npz(self, sample_ik_result, tmp_path) -> None:
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

    def test_export_json(self, sample_ik_result, tmp_path) -> None:
        """Test JSON export."""
        import json

        from motion_training.trajectory_exporter import TrajectoryExporter

        exporter = TrajectoryExporter(sample_ik_result)
        output_path = exporter.export(tmp_path / "test", format="json")

        assert output_path.exists()
        assert output_path.suffix == ".json"

        # Verify JSON is valid
        with open(output_path) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "trajectory" in data

    def test_export_unsupported_format(self, sample_ik_result, tmp_path) -> None:
        """Test that unsupported format raises error."""
        from motion_training.trajectory_exporter import TrajectoryExporter

        exporter = TrajectoryExporter(sample_ik_result)
        with pytest.raises(ValueError):
            exporter.export(tmp_path / "test", format="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
