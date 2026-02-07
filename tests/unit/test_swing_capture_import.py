"""Tests for the Golf Swing Capture Import module.

Tests import from CSV, JSON, and C3D formats, marker-to-joint conversion,
swing phase detection, and RL trajectory export.

Follows TDD and Design by Contract principles.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.swing_capture_import import (
    JointTrajectory,
    SwingCaptureImporter,
    SwingPhaseLabels,
)

# ---- Fixtures ----


@pytest.fixture
def importer() -> SwingCaptureImporter:
    """Create a default swing capture importer."""
    return SwingCaptureImporter(target_frame_rate=100.0)


@pytest.fixture
def sample_csv_file(tmp_path: Path) -> Path:
    """Create a sample CSV file with joint trajectory data."""
    filepath = tmp_path / "swing_capture.csv"
    n_frames = 200
    times = np.linspace(0, 2.0, n_frames)

    # Simulate a swing-like pattern
    shoulder = 0.5 * np.sin(2 * np.pi * 0.5 * times)
    elbow = 0.3 * np.sin(2 * np.pi * 0.5 * times + 0.5)
    wrist = 0.2 * np.sin(2 * np.pi * 0.5 * times + 1.0)

    header = "time,shoulder,elbow,wrist"
    data = np.column_stack([times, shoulder, elbow, wrist])
    np.savetxt(str(filepath), data, delimiter=",", header=header, comments="")
    return filepath


@pytest.fixture
def sample_json_file(tmp_path: Path) -> Path:
    """Create a sample JSON file with joint trajectory data."""
    filepath = tmp_path / "swing_capture.json"
    n_frames = 100
    times = np.linspace(0, 1.0, n_frames).tolist()
    positions = np.random.randn(n_frames, 4).tolist()

    data = {
        "times": times,
        "joint_names": ["shoulder", "elbow", "wrist", "hip"],
        "positions": positions,
    }

    with open(filepath, "w") as f:
        json.dump(data, f)
    return filepath


@pytest.fixture
def sample_trajectory() -> JointTrajectory:
    """Create a sample joint trajectory for testing."""
    n_frames = 200
    times = np.linspace(0, 2.0, n_frames)
    positions = np.column_stack(
        [
            0.5 * np.sin(2 * np.pi * 0.5 * times),  # shoulder
            0.3 * np.sin(2 * np.pi * 0.5 * times + 0.5),  # elbow
            0.2 * np.sin(2 * np.pi * 0.5 * times + 1.0),  # wrist
        ]
    )
    velocities = np.gradient(positions, times, axis=0)

    return JointTrajectory(
        joint_names=["shoulder", "elbow", "wrist"],
        positions=positions,
        velocities=velocities,
        times=times,
        frame_rate=100.0,
        source_file="test_swing.csv",
    )


# ---- JointTrajectory Tests ----


class TestJointTrajectory:
    """Tests for JointTrajectory dataclass."""

    def test_properties(self, sample_trajectory: JointTrajectory) -> None:
        """Test trajectory properties."""
        assert sample_trajectory.n_frames == 200
        assert sample_trajectory.n_joints == 3

    def test_joint_names(self, sample_trajectory: JointTrajectory) -> None:
        """Test joint names."""
        assert sample_trajectory.joint_names == ["shoulder", "elbow", "wrist"]

    def test_data_dimensions(self, sample_trajectory: JointTrajectory) -> None:
        """Postcondition: positions and velocities have consistent shapes."""
        assert sample_trajectory.positions.shape == (200, 3)
        assert sample_trajectory.velocities.shape == (200, 3)
        assert len(sample_trajectory.times) == 200


# ---- CSV Import Tests ----


class TestCSVImport:
    """Tests for CSV file import."""

    def test_import_csv_basic(
        self, importer: SwingCaptureImporter, sample_csv_file: Path
    ) -> None:
        """Test basic CSV import."""
        trajectory = importer.import_csv(sample_csv_file)

        assert isinstance(trajectory, JointTrajectory)
        assert trajectory.n_frames == 200
        assert trajectory.n_joints == 3
        assert trajectory.joint_names == ["shoulder", "elbow", "wrist"]

    def test_import_csv_values(
        self, importer: SwingCaptureImporter, sample_csv_file: Path
    ) -> None:
        """Postcondition: imported values are finite."""
        trajectory = importer.import_csv(sample_csv_file)
        assert np.all(np.isfinite(trajectory.positions))
        assert np.all(np.isfinite(trajectory.velocities))

    def test_import_csv_velocities_computed(
        self, importer: SwingCaptureImporter, sample_csv_file: Path
    ) -> None:
        """Postcondition: velocities are computed from positions."""
        trajectory = importer.import_csv(sample_csv_file)
        assert trajectory.velocities.shape == trajectory.positions.shape

    def test_import_csv_file_not_found(self, importer: SwingCaptureImporter) -> None:
        """Precondition: file must exist."""
        with pytest.raises(FileNotFoundError):
            importer.import_csv("/nonexistent/file.csv")

    def test_import_csv_auto_detect(
        self, importer: SwingCaptureImporter, sample_csv_file: Path
    ) -> None:
        """Test auto-detect format for CSV."""
        trajectory = importer.import_file(sample_csv_file)
        assert isinstance(trajectory, JointTrajectory)


# ---- JSON Import Tests ----


class TestJSONImport:
    """Tests for JSON file import."""

    def test_import_json_basic(
        self, importer: SwingCaptureImporter, sample_json_file: Path
    ) -> None:
        """Test basic JSON import."""
        trajectory = importer.import_json(sample_json_file)

        assert isinstance(trajectory, JointTrajectory)
        assert trajectory.n_frames == 100
        assert trajectory.n_joints == 4
        assert trajectory.joint_names == ["shoulder", "elbow", "wrist", "hip"]

    def test_import_json_velocities_computed(
        self, importer: SwingCaptureImporter, sample_json_file: Path
    ) -> None:
        """Test velocities are computed when not provided."""
        trajectory = importer.import_json(sample_json_file)
        assert trajectory.velocities.shape == trajectory.positions.shape

    def test_import_json_with_velocities(self, tmp_path: Path) -> None:
        """Test JSON import with explicit velocities."""
        filepath = tmp_path / "with_vel.json"
        data = {
            "times": [0.0, 0.01, 0.02],
            "positions": [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
            "velocities": [[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]],
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

        importer = SwingCaptureImporter()
        trajectory = importer.import_json(filepath)
        np.testing.assert_array_equal(
            trajectory.velocities, [[10, 10], [10, 10], [10, 10]]
        )

    def test_import_json_missing_keys(self, tmp_path: Path) -> None:
        """Precondition: JSON must have required keys."""
        filepath = tmp_path / "bad.json"
        with open(filepath, "w") as f:
            json.dump({"times": [0, 1]}, f)

        importer = SwingCaptureImporter()
        with pytest.raises(ValueError, match="missing required keys"):
            importer.import_json(filepath)

    def test_import_json_not_object(self, tmp_path: Path) -> None:
        """Precondition: JSON root must be object."""
        filepath = tmp_path / "array.json"
        with open(filepath, "w") as f:
            json.dump([1, 2, 3], f)

        importer = SwingCaptureImporter()
        with pytest.raises(ValueError, match="root must be an object"):
            importer.import_json(filepath)

    def test_import_json_auto_detect(
        self, importer: SwingCaptureImporter, sample_json_file: Path
    ) -> None:
        """Test auto-detect format for JSON."""
        trajectory = importer.import_file(sample_json_file)
        assert isinstance(trajectory, JointTrajectory)


# ---- Unsupported Format Tests ----


class TestUnsupportedFormats:
    """Tests for unsupported file formats."""

    def test_unsupported_format(
        self, importer: SwingCaptureImporter, tmp_path: Path
    ) -> None:
        """Precondition: format must be supported."""
        filepath = tmp_path / "data.xyz"
        filepath.touch()
        with pytest.raises(ValueError, match="Unsupported format"):
            importer.import_file(filepath)

    def test_file_not_found(self, importer: SwingCaptureImporter) -> None:
        """Precondition: file must exist."""
        with pytest.raises(FileNotFoundError):
            importer.import_file("/nonexistent/file.csv")


# ---- Swing Phase Detection Tests ----


class TestSwingPhaseDetection:
    """Tests for automatic swing phase detection."""

    def test_detect_phases(
        self, importer: SwingCaptureImporter, sample_trajectory: JointTrajectory
    ) -> None:
        """Test phase detection returns valid labels."""
        phases = importer.detect_swing_phases(sample_trajectory)

        assert isinstance(phases, SwingPhaseLabels)
        assert 0 <= phases.address < sample_trajectory.n_frames
        assert 0 <= phases.impact < sample_trajectory.n_frames
        assert phases.follow_through_end == sample_trajectory.n_frames - 1

    def test_phase_ordering(
        self, importer: SwingCaptureImporter, sample_trajectory: JointTrajectory
    ) -> None:
        """Test that phase indices are in chronological order."""
        phases = importer.detect_swing_phases(sample_trajectory)

        assert phases.address <= phases.backswing_start
        assert phases.top_of_backswing <= phases.impact
        assert phases.impact <= phases.follow_through_end


# ---- Demonstration Dataset Tests ----


class TestDemonstrationDataset:
    """Tests for building demonstration datasets."""

    def test_build_dataset(
        self, importer: SwingCaptureImporter, sample_trajectory: JointTrajectory
    ) -> None:
        """Test building demonstration dataset."""
        dataset = importer.build_demonstration_dataset([sample_trajectory])

        assert dataset["num_demonstrations"] == 1
        assert len(dataset["demonstrations"]) == 1
        assert dataset["format_version"] == "1.0"

    def test_dataset_contents(
        self, importer: SwingCaptureImporter, sample_trajectory: JointTrajectory
    ) -> None:
        """Test demonstration dataset content structure."""
        dataset = importer.build_demonstration_dataset([sample_trajectory])
        demo = dataset["demonstrations"][0]

        assert "observations" in demo
        assert "actions" in demo
        assert "times" in demo
        assert "joint_names" in demo
        assert demo["n_frames"] == 200
        assert demo["n_joints"] == 3

    def test_dataset_multiple_trajectories(
        self, importer: SwingCaptureImporter, sample_trajectory: JointTrajectory
    ) -> None:
        """Test dataset with multiple trajectories."""
        dataset = importer.build_demonstration_dataset(
            [sample_trajectory, sample_trajectory]
        )
        assert dataset["num_demonstrations"] == 2


# ---- RL Export Tests ----


class TestRLExport:
    """Tests for RL trajectory export."""

    def test_export_for_rl(
        self, importer: SwingCaptureImporter, sample_trajectory: JointTrajectory
    ) -> None:
        """Test RL export creates valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rl_trajectory.json"
            result_path = importer.export_for_rl(sample_trajectory, output_path)

            assert result_path.exists()

            with open(result_path) as f:
                data = json.load(f)

            assert "joint_names" in data
            assert "positions" in data
            assert "velocities" in data
            assert "times" in data
            assert data["n_frames"] == 200
            assert data["n_joints"] == 3

    def test_export_creates_parent_dirs(
        self, importer: SwingCaptureImporter, sample_trajectory: JointTrajectory
    ) -> None:
        """Test that export creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "deep" / "trajectory.json"
            result_path = importer.export_for_rl(sample_trajectory, output_path)
            assert result_path.exists()
