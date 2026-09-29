"""Path-traversal / LFI regression tests for dataset routes.

The endpoint accepts a ``file_path`` (read) and an optional ``output_path``
(write). Before the fix both flowed unvalidated into the importer, letting any
authenticated tenant read arbitrary files (``/etc/passwd``, other tenants'
captures) and write JSON to arbitrary writable paths. These tests assert both
paths are validated against allow-listed roots and that escapes return 400.

``/dataset/generate`` also accepts an ``output_path``. Issue #7329 covers the
same arbitrary-write sink on that endpoint.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_engine_manager
from src.api.routes import dataset as dataset_module
from src.api.routes.dataset import router

pytestmark = pytest.mark.unit


class _MockEngine:
    engine_type = "mock"


class _MockEngineManager:
    def get_active_physics_engine(self) -> _MockEngine:
        return _MockEngine()


class _GeneratedDataset:
    num_samples = 1
    total_frames = 2


class _RecordingDatasetGenerator:
    exported_paths: list[Path] = []

    def __init__(self, engine: _MockEngine) -> None:
        self.engine = engine

    def generate(self, config: object) -> _GeneratedDataset:
        return _GeneratedDataset()

    def export(
        self, dataset: _GeneratedDataset, output_path: Path, *, format: str
    ) -> Path:
        self.exported_paths.append(Path(output_path))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("dataset", encoding="utf-8")
        return Path(output_path)


class _RouteConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Client with the dataset allow-listed roots pinned under ``tmp_path``."""
    input_root = tmp_path / "data" / "captures"
    output_root = tmp_path / "output"
    input_root.mkdir(parents=True)
    output_root.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dataset_module, "_dataset_input_roots", lambda: [input_root])
    monkeypatch.setattr(dataset_module, "_dataset_output_roots", lambda: [output_root])
    monkeypatch.setitem(
        sys.modules,
        "src.shared.python.data_io.dataset_generator",
        types.SimpleNamespace(
            ControlProfile=_RouteConfig,
            DatasetGenerator=_RecordingDatasetGenerator,
            GeneratorConfig=_RouteConfig,
        ),
    )
    _RecordingDatasetGenerator.exported_paths = []

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_engine_manager] = lambda: _MockEngineManager()
    return TestClient(app)


def _write_valid_capture(path: Path) -> None:
    """Write a minimal valid swing-capture JSON the importer can parse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "joint_names": ["a", "b"],
        "times": [0.0, 0.01],
        "joint_angles": [[0.0, 0.0], [0.1, 0.1]],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_import_swing_rejects_absolute_file_path(client: TestClient) -> None:
    """An absolute path escaping the allow-listed input root is rejected."""
    resp = client.post(
        "/dataset/import-swing",
        json={"file_path": "/etc/passwd", "export_for_rl": False},
    )
    assert resp.status_code == 400


def test_import_swing_rejects_parent_traversal_file_path(client: TestClient) -> None:
    """A ``..`` traversal escaping the input root is rejected."""
    resp = client.post(
        "/dataset/import-swing",
        json={
            "file_path": "../../../../etc/passwd",
            "export_for_rl": False,
        },
    )
    assert resp.status_code == 400


def test_import_swing_rejects_output_path_escape(
    client: TestClient, tmp_path: Path
) -> None:
    """A contained input but an escaping ``output_path`` is rejected (no write)."""
    capture = tmp_path / "data" / "captures" / "swing.json"
    _write_valid_capture(capture)

    escape_target = tmp_path / "evil.json"
    resp = client.post(
        "/dataset/import-swing",
        json={
            "file_path": str(capture),
            "export_for_rl": True,
            "output_path": str(escape_target),
        },
    )
    assert resp.status_code == 400
    assert not escape_target.exists()


@pytest.mark.parametrize("output_path", ["../../evil", "/etc/passwd"])
def test_generate_dataset_rejects_output_path_escape(
    client: TestClient, tmp_path: Path, output_path: str
) -> None:
    """Dataset generation rejects traversal and absolute output targets."""
    resp = client.post(
        "/dataset/generate",
        json={"num_samples": 1, "duration": 0.01, "output_path": output_path},
    )

    assert resp.status_code == 400
    assert _RecordingDatasetGenerator.exported_paths == []
    assert not (tmp_path / "evil").exists()


def test_generate_dataset_accepts_contained_output_path(
    client: TestClient, tmp_path: Path
) -> None:
    """A relative output under the allow-listed output root still succeeds."""
    resp = client.post(
        "/dataset/generate",
        json={
            "num_samples": 1,
            "duration": 0.01,
            "output_path": "output/sub/ok",
        },
    )

    expected = (tmp_path / "output" / "sub" / "ok").resolve()
    assert resp.status_code == 200
    assert Path(resp.json()["export_path"]) == expected
    assert _RecordingDatasetGenerator.exported_paths == [expected]
    assert expected.read_text(encoding="utf-8") == "dataset"
