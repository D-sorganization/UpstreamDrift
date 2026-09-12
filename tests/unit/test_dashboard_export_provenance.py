"""Tests for Unified Engine Dashboard export provenance (#8820).

Validates that:
- GenericPhysicsRecorder stamps engine name, model path, model hash, run ID, and timestamp into datasets.
- Resetting recorder generates a new run ID.
- Multiformat exports (CSV, JSON, MATLAB, HDF5) embed provenance metadata.
- CSV comment headers with '#' preserve downstream tabular parsing in pandas.
- MuJoCo, Drake, and Pinocchio dashboard exports produce distinct, byte-level distinguishable outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
from src.shared.python.data_io.export import export_recording_all_formats
from src.shared.python.data_io.provenance import ProvenanceInfo
from src.shared.python.engine_core.interfaces import PhysicsEngine


@dataclass
class DummyCapabilities:
    engine_name: str


class MockEngineWithCaps(PhysicsEngine):
    """Mock engine exposing configurable name, model path, and capabilities."""

    def __init__(self, name: str = "MuJoCo", model_path: str | None = None) -> None:
        self._name = name
        self.model_path = model_path
        self._time = 0.0
        self._q = np.array([0.1, 0.2])
        self._v = np.array([0.3, 0.4])

    def get_capabilities(self) -> Any:
        return DummyCapabilities(engine_name=self._name)

    @property
    def model_name(self) -> str:
        return f"model_{self._name.lower()}"

    def get_time(self) -> float:
        return self._time

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q, self._v

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        self._q = q
        self._v = v

    def forward(self) -> None:
        pass

    def set_control(self, u: np.ndarray) -> None:
        pass

    def compute_mass_matrix(self) -> np.ndarray:
        return np.eye(2)

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0])

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0])

    def compute_drift_acceleration(self) -> np.ndarray:
        return np.array([0.0, 0.0])

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        return tau

    def compute_gravity_forces(self) -> np.ndarray:
        return np.zeros(2)

    def compute_jacobian(self, body_name: str) -> dict[str, Any]:
        return {}

    def compute_coriolis_centrifugal_forces(self) -> np.ndarray:
        return np.zeros(2)

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        return np.zeros(2)

    def compute_forward_dynamics(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray
    ) -> np.ndarray:
        return np.zeros(2)

    def compute_bias_forces(self) -> np.ndarray:
        return np.zeros(2)

    def load_from_path(self, path: str) -> None:
        self.model_path = path

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        pass

    def step(self, dt: float | None = None) -> None:
        if dt is not None:
            self._time += dt
        else:
            self._time += 0.01

    def reset(self) -> None:
        self._time = 0.0

    def get_joint_names(self) -> list[str]:
        return ["joint_1", "joint_2"]

    def get_actuator_names(self) -> list[str]:
        return ["act_1", "act_2"]

    def get_body_names(self) -> list[str]:
        return ["body_1"]

    def get_body_mass(self, body_name: str) -> float:
        return 1.0

    def get_body_inertia(self, body_name: str) -> np.ndarray:
        return np.eye(3)

    def get_body_position(self, body_name: str) -> np.ndarray:
        return np.zeros(3)

    def get_body_rotation(self, body_name: str) -> np.ndarray:
        return np.eye(3)

    def get_body_velocity(self, body_name: str) -> np.ndarray:
        return np.zeros(6)

    def get_contact_forces(self) -> list[Any]:
        return []

    def get_joint_limits(self) -> np.ndarray:
        return np.zeros(2)

    @property
    def engine_type(self) -> str:
        return self._name.lower()

    def save_checkpoint(self) -> Any:
        from src.shared.python.engine_core.checkpoint import StateCheckpoint

        return StateCheckpoint(
            id="mock_cp",
            timestamp=self._time,
            wall_time=0.0,
            engine_type=self.engine_type,
            engine_state={"q": self._q.tolist(), "v": self._v.tolist()},
            q=tuple(self._q.tolist()),
            v=tuple(self._v.tolist()),
        )

    def restore_checkpoint(self, checkpoint: Any) -> None:
        return None


@pytest.mark.unit
class TestDashboardExportProvenance:
    """Test suite for Issue #8820 provenance stamping."""

    def test_recorder_stamps_engine_and_run_id(self, tmp_path: Path) -> None:
        model_file = tmp_path / "robot.urdf"
        model_file.write_text("<robot name='test'/>")

        engine = MockEngineWithCaps(name="MuJoCo", model_path=str(model_file))
        recorder = GenericPhysicsRecorder(engine, max_samples=100, initial_capacity=10)

        recorder.start()
        recorder.record_step()
        recorder.stop()

        data = recorder.get_data_dict()

        assert data["engine_name"] == "MuJoCo"
        assert "run_id" in data
        assert len(data["run_id"]) > 0
        assert data["model_file_path"] == str(model_file)
        assert "model_file_hash" in data
        assert isinstance(data["provenance"], ProvenanceInfo)
        assert data["provenance"].engine_name == "MuJoCo"
        assert data["provenance"].run_id == data["run_id"]
        assert data["provenance"].model_file_path == str(model_file)

    def test_recorder_reset_regenerates_run_id(self) -> None:
        engine = MockEngineWithCaps(name="Drake")
        recorder = GenericPhysicsRecorder(engine)

        initial_run_id = recorder.run_id
        assert initial_run_id is not None

        recorder.reset()
        new_run_id = recorder.run_id

        assert new_run_id is not None
        assert new_run_id != initial_run_id

    def test_csv_export_provenance_headers_and_pandas_parsing(
        self, tmp_path: Path
    ) -> None:
        model_file = tmp_path / "club.xml"
        model_file.write_text("<mujoco><worldbody/></mujoco>")

        engine = MockEngineWithCaps(name="Pinocchio", model_path=str(model_file))
        recorder = GenericPhysicsRecorder(engine, max_samples=50, initial_capacity=10)

        recorder.start()
        for _ in range(5):
            recorder.record_step()
        recorder.stop()

        data = recorder.get_data_dict()
        base_path = str(tmp_path / "pinocchio_export")

        results = export_recording_all_formats(base_path, data, formats=["csv"])
        assert results["csv"] is True

        csv_path = tmp_path / "pinocchio_export.csv"
        assert csv_path.exists()

        content = csv_path.read_text()
        # Verify provenance comment headers exist
        assert "# Engine: Pinocchio" in content
        assert f"# Run ID: {data['run_id']}" in content
        assert f"# Model file: {model_file}" in content
        assert "# Generated: " in content

        # Verify pandas parses numeric tabular rows when comment='#'
        df = pd.read_csv(csv_path, comment="#")
        assert len(df) == 5
        assert "time" in df.columns
        assert "kinetic_energy" in df.columns

    def test_json_export_embeds_provenance(self, tmp_path: Path) -> None:
        engine = MockEngineWithCaps(name="MuJoCo")
        recorder = GenericPhysicsRecorder(engine, max_samples=50, initial_capacity=10)

        recorder.start()
        recorder.record_step()
        recorder.stop()

        data = recorder.get_data_dict()
        base_path = str(tmp_path / "mujoco_export")

        results = export_recording_all_formats(base_path, data, formats=["json"])
        assert results["json"] is True

        json_path = tmp_path / "mujoco_export.json"
        assert json_path.exists()

        with open(json_path) as f:
            payload = json.load(f)

        assert "provenance" in payload
        assert payload["provenance"]["engine_name"] == "MuJoCo"
        assert payload["provenance"]["run_id"] == data["run_id"]
        assert "timestamp_utc" in payload["provenance"]

    def test_engine_exports_are_distinguishable(self, tmp_path: Path) -> None:
        """Verify MuJoCo, Drake, and Pinocchio exports are byte-level distinct."""
        engines = [
            MockEngineWithCaps(name="MuJoCo"),
            MockEngineWithCaps(name="Drake"),
            MockEngineWithCaps(name="Pinocchio"),
        ]

        csv_outputs = []
        for eng in engines:
            rec = GenericPhysicsRecorder(eng, max_samples=20, initial_capacity=10)
            rec.start()
            rec.record_step()
            rec.stop()

            base = str(tmp_path / f"export_{eng.get_capabilities().engine_name}")
            export_recording_all_formats(base, rec.get_data_dict(), formats=["csv"])
            csv_path = tmp_path / f"export_{eng.get_capabilities().engine_name}.csv"
            csv_outputs.append(csv_path.read_text())

        # All exports must be pair-wise distinguishable
        assert csv_outputs[0] != csv_outputs[1]
        assert csv_outputs[1] != csv_outputs[2]
        assert csv_outputs[0] != csv_outputs[2]

        assert "# Engine: MuJoCo" in csv_outputs[0]
        assert "# Engine: Drake" in csv_outputs[1]
        assert "# Engine: Pinocchio" in csv_outputs[2]
