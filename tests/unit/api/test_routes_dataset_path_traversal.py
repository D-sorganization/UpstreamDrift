"""Path-traversal / LFI regression tests for ``/dataset/import-swing`` (#6926).

The endpoint accepts a ``file_path`` (read) and an optional ``output_path``
(write). Before the fix both flowed unvalidated into the importer, letting any
authenticated tenant read arbitrary files (``/etc/passwd``, other tenants'
captures) and write JSON to arbitrary writable paths. These tests assert both
paths are validated against allow-listed roots and that escapes return 400.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import dataset as dataset_module
from src.api.routes.dataset import router


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Client with the dataset allow-listed roots pinned under ``tmp_path``."""
    input_root = tmp_path / "data" / "captures"
    output_root = tmp_path / "output" / "rl_trajectories"
    input_root.mkdir(parents=True)
    output_root.mkdir(parents=True)

    monkeypatch.setattr(dataset_module, "_dataset_input_roots", lambda: [input_root])
    monkeypatch.setattr(dataset_module, "_dataset_output_roots", lambda: [output_root])

    app = FastAPI()
    app.include_router(router)
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
