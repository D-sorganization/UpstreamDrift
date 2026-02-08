"""Tests for Engine Capabilities, Export, and Simulation Control modules.

TDD tests covering:
    - EngineCapabilities creation and serialization
    - CapabilityLevel enum
    - DatasetExporter CSV/JSON export
    - SimulationController state machine
    - ForceOverlay and MeasurementResult serialization
    - PhysicsEngine.get_capabilities() default
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.engines.common.capabilities import CapabilityLevel, EngineCapabilities
from src.engines.common.export import DatasetExporter, DatasetRecord, VideoConfig
from src.engines.common.simulation_control import (
    ForceOverlay,
    MeasurementResult,
    SimulationController,
    SimulationMode,
)

# =============================================================================
# EngineCapabilities Tests
# =============================================================================


class TestCapabilityLevel:
    """Tests for CapabilityLevel enum."""

    def test_levels_exist(self) -> None:
        """All three levels must exist."""
        assert CapabilityLevel.FULL is not None
        assert CapabilityLevel.PARTIAL is not None
        assert CapabilityLevel.NONE is not None

    def test_levels_are_distinct(self) -> None:
        """Levels must be distinct values."""
        assert CapabilityLevel.FULL != CapabilityLevel.PARTIAL
        assert CapabilityLevel.FULL != CapabilityLevel.NONE
        assert CapabilityLevel.PARTIAL != CapabilityLevel.NONE


class TestEngineCapabilities:
    """Tests for EngineCapabilities dataclass."""

    def test_default_creation(self) -> None:
        """Default capabilities should be NONE for everything."""
        caps = EngineCapabilities()
        assert caps.engine_name == ""
        assert caps.mass_matrix == CapabilityLevel.NONE
        assert caps.video_export == CapabilityLevel.NONE
        assert caps.dataset_export == CapabilityLevel.NONE

    def test_frozen_immutability(self) -> None:
        """Capabilities must be immutable after creation."""
        caps = EngineCapabilities(engine_name="test")
        with pytest.raises(AttributeError):
            caps.engine_name = "mutated"  # type: ignore[misc]

    def test_convenience_properties_none(self) -> None:
        """Convenience properties should be False when NONE."""
        caps = EngineCapabilities()
        assert not caps.has_video_export
        assert not caps.has_dataset_export
        assert not caps.has_force_visualization
        assert not caps.has_contact_forces
        assert not caps.has_measurements

    def test_convenience_properties_full(self) -> None:
        """Convenience properties should be True when FULL."""
        caps = EngineCapabilities(
            video_export=CapabilityLevel.FULL,
            dataset_export=CapabilityLevel.FULL,
            force_visualization=CapabilityLevel.FULL,
            contact_forces=CapabilityLevel.FULL,
            measurements=CapabilityLevel.FULL,
        )
        assert caps.has_video_export
        assert caps.has_dataset_export
        assert caps.has_force_visualization
        assert caps.has_contact_forces
        assert caps.has_measurements

    def test_convenience_properties_partial(self) -> None:
        """Convenience properties should be True when PARTIAL."""
        caps = EngineCapabilities(
            video_export=CapabilityLevel.PARTIAL,
            contact_forces=CapabilityLevel.PARTIAL,
        )
        assert caps.has_video_export
        assert caps.has_contact_forces

    def test_to_dict(self) -> None:
        """Serialization to dict for API responses."""
        caps = EngineCapabilities(
            engine_name="MuJoCo",
            mass_matrix=CapabilityLevel.FULL,
            jacobian=CapabilityLevel.FULL,
            contact_forces=CapabilityLevel.FULL,
            video_export=CapabilityLevel.FULL,
        )
        d = caps.to_dict()
        assert d["engine_name"] == "MuJoCo"
        assert d["mass_matrix"] == "full"
        assert d["video_export"] == "full"
        assert d["dataset_export"] == "none"

    def test_roundtrip_serialization(self) -> None:
        """Serialize and deserialize should produce equivalent object."""
        original = EngineCapabilities(
            engine_name="Drake",
            mass_matrix=CapabilityLevel.FULL,
            jacobian=CapabilityLevel.FULL,
            contact_forces=CapabilityLevel.PARTIAL,
            video_export=CapabilityLevel.NONE,
        )
        d = original.to_dict()
        restored = EngineCapabilities.from_dict(d)
        assert restored.engine_name == "Drake"
        assert restored.mass_matrix == CapabilityLevel.FULL
        assert restored.contact_forces == CapabilityLevel.PARTIAL
        assert restored.video_export == CapabilityLevel.NONE

    def test_from_dict_invalid_level(self) -> None:
        """Unknown capability levels should default to NONE."""
        d = {"engine_name": "test", "mass_matrix": "invalid_value"}
        caps = EngineCapabilities.from_dict(d)
        assert caps.mass_matrix == CapabilityLevel.NONE


# =============================================================================
# DatasetExporter Tests
# =============================================================================


class TestDatasetExporter:
    """Tests for DatasetExporter."""

    @pytest.fixture()
    def exporter(self) -> DatasetExporter:
        """Create a DatasetExporter with sample joint names."""
        return DatasetExporter(joint_names=["shoulder", "elbow", "wrist"])

    @pytest.fixture()
    def sample_records(self) -> list[DatasetRecord]:
        """Create sample dataset records."""
        return [
            DatasetRecord(
                time=0.0,
                positions=np.array([0.0, 0.1, 0.2]),
                velocities=np.array([0.0, 0.0, 0.0]),
            ),
            DatasetRecord(
                time=0.01,
                positions=np.array([0.01, 0.11, 0.21]),
                velocities=np.array([1.0, 1.0, 1.0]),
            ),
            DatasetRecord(
                time=0.02,
                positions=np.array([0.03, 0.13, 0.22]),
                velocities=np.array([2.0, 2.0, 1.0]),
            ),
        ]

    def test_empty_export_raises(self, exporter: DatasetExporter) -> None:
        """Exporting with no records should raise."""
        with pytest.raises(ValueError, match="No records"):
            exporter.export_csv(Path("/tmp/test.csv"))

    def test_add_record(
        self, exporter: DatasetExporter, sample_records: list[DatasetRecord]
    ) -> None:
        """Adding records should increment count."""
        for rec in sample_records:
            exporter.add_record(rec)
        assert exporter.record_count == 3

    def test_chronological_order_enforced(self, exporter: DatasetExporter) -> None:
        """Records must be in chronological order."""
        exporter.add_record(
            DatasetRecord(time=1.0, positions=np.zeros(3), velocities=np.zeros(3))
        )
        with pytest.raises(ValueError, match="chronological"):
            exporter.add_record(
                DatasetRecord(time=0.5, positions=np.zeros(3), velocities=np.zeros(3))
            )

    def test_export_csv(
        self, exporter: DatasetExporter, sample_records: list[DatasetRecord]
    ) -> None:
        """CSV export should create valid file with correct columns."""
        for rec in sample_records:
            exporter.add_record(rec)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.csv"
            result = exporter.export_csv(path)
            assert result == path
            assert path.exists()

            lines = path.read_text().strip().split("\n")
            assert len(lines) == 4  # header + 3 records
            header = lines[0]
            assert "time" in header
            assert "q_shoulder" in header
            assert "v_shoulder" in header

    def test_export_json(
        self, exporter: DatasetExporter, sample_records: list[DatasetRecord]
    ) -> None:
        """JSON export should create valid file with records."""
        for rec in sample_records:
            exporter.add_record(rec)
        exporter.set_metadata("engine", "MuJoCo")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.json"
            result = exporter.export_json(path)
            assert result == path
            assert path.exists()

            data = json.loads(path.read_text())
            assert data["record_count"] == 3
            assert data["metadata"]["engine"] == "MuJoCo"
            assert len(data["records"]) == 3
            assert data["records"][0]["time"] == 0.0

    def test_add_from_state_convenience(self, exporter: DatasetExporter) -> None:
        """add_from_state convenience method should work."""
        exporter.add_from_state(
            time=0.0,
            q=np.array([1.0, 2.0, 3.0]),
            v=np.array([0.0, 0.0, 0.0]),
        )
        assert exporter.record_count == 1

    def test_clear_records(
        self, exporter: DatasetExporter, sample_records: list[DatasetRecord]
    ) -> None:
        """Clearing should reset record count."""
        for rec in sample_records:
            exporter.add_record(rec)
        assert exporter.record_count == 3
        exporter.clear()
        assert exporter.record_count == 0

    def test_csv_with_accelerations(self) -> None:
        """CSV export should include acceleration columns when present."""
        exporter = DatasetExporter()
        exporter.add_record(
            DatasetRecord(
                time=0.0,
                positions=np.array([1.0]),
                velocities=np.array([0.0]),
                accelerations=np.array([9.81]),
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "accel.csv"
            exporter.export_csv(path)
            header = path.read_text().strip().split("\n")[0]
            assert "a_0" in header


# =============================================================================
# SimulationController Tests
# =============================================================================


class _MockController(SimulationController):
    """Concrete implementation for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.step_count = 0

    def _do_step(self) -> None:
        self.step_count += 1

    def translate_body(self, body_name: str, delta: np.ndarray) -> bool:
        return True

    def rotate_body(self, body_name: str, axis: np.ndarray, angle: float) -> bool:
        return True

    def measure_distance(self, body_a: str, body_b: str) -> MeasurementResult:
        return MeasurementResult(
            type="distance", value=1.5, unit="m", point_a=body_a, point_b=body_b
        )

    def measure_angle(self, body_a: str, body_b: str, body_c: str) -> MeasurementResult:
        return MeasurementResult(
            type="angle",
            value=1.57,
            unit="rad",
            point_a=body_a,
            point_b=body_b,
        )


class TestSimulationController:
    """Tests for SimulationController state machine."""

    @pytest.fixture()
    def ctrl(self) -> _MockController:
        return _MockController()

    def test_initial_state(self, ctrl: _MockController) -> None:
        """Controller starts in IDLE mode."""
        assert ctrl.mode == SimulationMode.IDLE
        assert not ctrl.is_running
        assert not ctrl.is_paused

    def test_start(self, ctrl: _MockController) -> None:
        """Starting from IDLE transitions to RUNNING."""
        assert ctrl.start()
        assert ctrl.mode == SimulationMode.RUNNING
        assert ctrl.is_running

    def test_pause(self, ctrl: _MockController) -> None:
        """Pausing from RUNNING transitions to PAUSED."""
        ctrl.start()
        assert ctrl.pause()
        assert ctrl.mode == SimulationMode.PAUSED
        assert ctrl.is_paused

    def test_resume(self, ctrl: _MockController) -> None:
        """Starting from PAUSED transitions to RUNNING."""
        ctrl.start()
        ctrl.pause()
        assert ctrl.start()
        assert ctrl.is_running

    def test_stop(self, ctrl: _MockController) -> None:
        """Stopping transitions to IDLE."""
        ctrl.start()
        assert ctrl.stop()
        assert ctrl.mode == SimulationMode.IDLE

    def test_invalid_start_from_running(self, ctrl: _MockController) -> None:
        """Cannot start when already running."""
        ctrl.start()
        assert not ctrl.start()

    def test_invalid_pause_from_idle(self, ctrl: _MockController) -> None:
        """Cannot pause when idle."""
        assert not ctrl.pause()

    def test_single_step(self, ctrl: _MockController) -> None:
        """Single step should execute one step and pause."""
        assert ctrl.single_step()
        assert ctrl.step_count == 1
        assert ctrl.mode == SimulationMode.PAUSED

    def test_force_overlay_add_clear(self, ctrl: _MockController) -> None:
        """Force overlays can be added and cleared."""
        overlay = ForceOverlay(
            body_name="wrist",
            force=np.array([10.0, 0.0, 0.0]),
        )
        ctrl.add_force_overlay(overlay)
        assert len(ctrl.overlays) == 1
        ctrl.clear_overlays()
        assert len(ctrl.overlays) == 0

    def test_measurement_distance(self, ctrl: _MockController) -> None:
        """Distance measurement should return valid result."""
        result = ctrl.measure_distance("body_a", "body_b")
        assert result.type == "distance"
        assert result.value == 1.5
        assert result.unit == "m"

    def test_measurement_angle(self, ctrl: _MockController) -> None:
        """Angle measurement should return valid result."""
        result = ctrl.measure_angle("body_a", "body_b", "body_c")
        assert result.type == "angle"
        assert result.value == pytest.approx(1.57)


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for API serialization."""

    def test_force_overlay_to_dict(self) -> None:
        """ForceOverlay should serialize correctly."""
        overlay = ForceOverlay(
            body_name="elbow",
            force=np.array([1.0, 2.0, 3.0]),
            torque=np.array([0.1, 0.2, 0.3]),
            scale=2.0,
            label="joint torque",
        )
        d = overlay.to_dict()
        assert d["body_name"] == "elbow"
        assert d["force"] == [1.0, 2.0, 3.0]
        assert d["label"] == "joint torque"

    def test_measurement_result_to_dict(self) -> None:
        """MeasurementResult should serialize correctly."""
        result = MeasurementResult(
            type="distance",
            value=2.5,
            unit="m",
            point_a="hip",
            point_b="knee",
            vector=np.array([0.0, -1.0, 0.0]),
        )
        d = result.to_dict()
        assert d["type"] == "distance"
        assert d["value"] == 2.5
        assert d["vector"] == [0.0, -1.0, 0.0]


# =============================================================================
# VideoConfig Tests
# =============================================================================


class TestVideoConfig:
    """Tests for VideoConfig dataclass."""

    def test_defaults(self) -> None:
        """Default config should use HD 1080p."""
        config = VideoConfig()
        assert config.width == 1920
        assert config.height == 1080
        assert config.fps == 60
        assert config.format == "mp4"
        assert config.show_overlays

    def test_custom(self) -> None:
        """Custom config should override defaults."""
        config = VideoConfig(width=1280, height=720, fps=30, format="avi")
        assert config.width == 1280
        assert config.fps == 30
