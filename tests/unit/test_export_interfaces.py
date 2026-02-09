"""Tests for common export interfaces (Issue #1176).

Validates the VideoConfig, VideoExportProtocol, DatasetExporter,
and DatasetRecord classes in src/engines/common/export.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.common.export import (
    DatasetExporter,
    DatasetRecord,
    VideoConfig,
    VideoExportProtocol,
)


class TestVideoConfig:
    """Tests for VideoConfig dataclass."""

    def test_default_values(self) -> None:
        config = VideoConfig()
        assert config.width == 1920
        assert config.height == 1080
        assert config.fps == 60
        assert config.format == "mp4"
        assert config.codec is None
        assert config.show_overlays is True

    def test_custom_values(self) -> None:
        config = VideoConfig(width=640, height=480, fps=30, format="gif")
        assert config.width == 640
        assert config.fps == 30
        assert config.format == "gif"


class TestVideoExportProtocol:
    """Tests for VideoExportProtocol abstract class."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            VideoExportProtocol()  # type: ignore[abstract]

    def test_concrete_implementation(self) -> None:
        """A concrete subclass should work."""

        class DummyVideoExporter(VideoExportProtocol):
            def __init__(self) -> None:
                self._recording = False
                self._frames = 0

            def start_recording(self, output_path: Path, config: VideoConfig) -> bool:
                self._recording = True
                return True

            def capture_frame(self, overlay_callback=None) -> bool:
                self._frames += 1
                return True

            def finish_recording(self) -> Path | None:
                self._recording = False
                return Path("output.mp4")

            @property
            def is_recording(self) -> bool:
                return self._recording

            @property
            def frame_count(self) -> int:
                return self._frames

        exporter = DummyVideoExporter()
        assert not exporter.is_recording
        assert exporter.frame_count == 0

        exporter.start_recording(Path("test.mp4"), VideoConfig())
        assert exporter.is_recording

        exporter.capture_frame()
        exporter.capture_frame()
        assert exporter.frame_count == 2

        result = exporter.finish_recording()
        assert result == Path("output.mp4")
        assert not exporter.is_recording


class TestDatasetRecord:
    """Tests for DatasetRecord dataclass."""

    def test_minimal_record(self) -> None:
        rec = DatasetRecord(
            time=0.0,
            positions=np.array([1.0, 2.0]),
            velocities=np.array([0.1, 0.2]),
        )
        assert rec.time == 0.0
        assert rec.accelerations is None
        assert rec.forces is None
        assert rec.metadata == {}

    def test_full_record(self) -> None:
        rec = DatasetRecord(
            time=1.5,
            positions=np.array([1.0]),
            velocities=np.array([0.5]),
            accelerations=np.array([-9.81]),
            forces=np.array([10.0]),
            metadata={"engine": "MuJoCo"},
        )
        assert rec.time == 1.5
        np.testing.assert_array_equal(rec.accelerations, np.array([-9.81]))
        assert rec.metadata["engine"] == "MuJoCo"


class TestDatasetExporter:
    """Tests for DatasetExporter."""

    def _make_exporter(self, n_records: int = 5) -> DatasetExporter:
        """Create an exporter with sample records."""
        exporter = DatasetExporter(joint_names=["shoulder", "elbow"])
        exporter.set_metadata("engine", "PendulumEngine")
        exporter.set_metadata("model", "double_pendulum")

        for i in range(n_records):
            t = i * 0.01
            exporter.add_from_state(
                time=t,
                q=np.array([0.1 * i, -0.05 * i]),
                v=np.array([1.0 - 0.1 * i, 0.5 + 0.05 * i]),
                qacc=np.array([-9.81 * np.sin(0.1 * i), -4.9 * np.sin(0.05 * i)]),
            )
        return exporter

    def test_record_count(self) -> None:
        exporter = self._make_exporter(10)
        assert exporter.record_count == 10

    def test_chronological_order_enforced(self) -> None:
        exporter = DatasetExporter()
        exporter.add_from_state(0.01, np.array([0.0]), np.array([0.0]))
        exporter.add_from_state(0.02, np.array([0.1]), np.array([0.1]))

        with pytest.raises(ValueError, match="chronological"):
            exporter.add_from_state(0.005, np.array([0.0]), np.array([0.0]))

    def test_export_csv(self, tmp_path: Path) -> None:
        exporter = self._make_exporter(5)
        output = tmp_path / "test.csv"

        result = exporter.export_csv(output)

        assert result == output
        assert output.exists()

        # Read and validate CSV
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 6  # header + 5 records

        header = lines[0].split(",")
        assert header[0] == "time"
        assert "q_shoulder" in header
        assert "q_elbow" in header
        assert "v_shoulder" in header
        assert "v_elbow" in header

    def test_export_json(self, tmp_path: Path) -> None:
        import json

        exporter = self._make_exporter(3)
        output = tmp_path / "test.json"

        result = exporter.export_json(output)

        assert result == output
        assert output.exists()

        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["record_count"] == 3
        assert data["metadata"]["engine"] == "PendulumEngine"
        assert len(data["records"]) == 3
        assert "positions" in data["records"][0]
        assert "velocities" in data["records"][0]

    def test_export_empty_raises(self, tmp_path: Path) -> None:
        exporter = DatasetExporter()

        with pytest.raises(ValueError, match="No records"):
            exporter.export_csv(tmp_path / "empty.csv")

        with pytest.raises(ValueError, match="No records"):
            exporter.export_json(tmp_path / "empty.json")

    def test_clear(self) -> None:
        exporter = self._make_exporter(5)
        assert exporter.record_count == 5

        exporter.clear()
        assert exporter.record_count == 0

    def test_export_csv_without_joint_names(self, tmp_path: Path) -> None:
        """CSV export without explicit joint names uses q_0, q_1 etc."""
        exporter = DatasetExporter()
        exporter.add_from_state(0.0, np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        output = tmp_path / "no_names.csv"

        exporter.export_csv(output)
        header = output.read_text(encoding="utf-8").split("\n")[0]
        assert "q_0" in header
        assert "v_0" in header

    def test_export_csv_with_forces(self, tmp_path: Path) -> None:
        """CSV export includes force columns when present."""
        exporter = DatasetExporter(joint_names=["j1"])
        exporter.add_from_state(
            0.0,
            np.array([1.0]),
            np.array([0.1]),
            forces=np.array([5.0]),
        )
        output = tmp_path / "forces.csv"

        exporter.export_csv(output)
        header = output.read_text(encoding="utf-8").split("\n")[0]
        assert "f_0" in header


class TestDatasetExporterHDF5:
    """Tests for HDF5 export (requires h5py)."""

    def test_export_hdf5(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")

        exporter = DatasetExporter(joint_names=["shoulder", "elbow"])
        exporter.set_metadata("engine", "test")
        for i in range(10):
            exporter.add_from_state(
                time=i * 0.01,
                q=np.array([0.1 * i, -0.05 * i]),
                v=np.array([1.0, 0.5]),
            )

        output = tmp_path / "test.h5"
        result = exporter.export_hdf5(output)

        assert result == output
        assert output.exists()

        import h5py

        with h5py.File(output, "r") as f:
            assert "time" in f
            assert "positions" in f
            assert "velocities" in f
            assert len(f["time"]) == 10
