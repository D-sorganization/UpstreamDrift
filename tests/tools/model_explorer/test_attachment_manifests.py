"""Attachment manifest parsing and attach-flow defaults."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

from src.tools.model_explorer._attachment_dialog import (
    AttachmentPointSelector,
    declared_payload_warnings,
)
from src.tools.model_explorer.attachment_manifest import (
    attachment_sidecar_path,
    load_attachment_manifest,
)

pytestmark = [pytest.mark.unit]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _write_model(path: Path) -> Path:
    path.write_text('<robot name="r"><link name="hand"/></robot>', encoding="utf-8")
    return path


def _write_manifest(model_path: Path, payload: dict[str, object]) -> Path:
    sidecar = attachment_sidecar_path(model_path)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar


def test_loads_declared_attachment_points_from_sidecar(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "human.urdf")
    _write_manifest(
        model,
        {
            "schema_version": 1,
            "attachment_points": [
                {
                    "name": "right hand tool mount",
                    "link_name": "hand",
                    "role": "tool-mount",
                    "interface_frame": {
                        "xyz": [0.1, 0.2, 0.3],
                        "rpy": [0.0, 1.57, 0.0],
                    },
                    "max_payload_kg": 2.5,
                    "tags": ["right", "hand"],
                }
            ],
        },
    )

    result = load_attachment_manifest(model)

    assert result.warnings == ()
    assert len(result.attachment_points) == 1
    point = result.attachment_points[0]
    assert point.name == "right hand tool mount"
    assert point.link_name == "hand"
    assert point.role == "tool-mount"
    assert point.interface_frame.xyz == (0.1, 0.2, 0.3)
    assert point.interface_frame.rpy == (0.0, 1.57, 0.0)
    assert point.max_payload_kg == 2.5
    assert point.tags == ("right", "hand")


def test_missing_manifest_returns_empty_result(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "human.urdf")

    result = load_attachment_manifest(model)

    assert result.attachment_points == ()
    assert result.warnings == ()


def test_checked_in_json_schema_is_valid_json() -> None:
    schema = Path("src/tools/model_explorer/attachment_manifest.schema.json")

    parsed = json.loads(schema.read_text(encoding="utf-8"))

    assert parsed["properties"]["schema_version"]["const"] == 1
    assert "attachment_points" in parsed["required"]


def test_malformed_manifest_warns_without_raising(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "human.urdf")
    attachment_sidecar_path(model).write_text("{not json", encoding="utf-8")

    result = load_attachment_manifest(model)

    assert result.attachment_points == ()
    assert result.warnings
    assert "invalid JSON" in result.warnings[0]


def test_invalid_entry_warns_without_hiding_valid_points(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "human.urdf")
    _write_manifest(
        model,
        {
            "schema_version": 1,
            "attachment_points": [
                {"name": "", "link_name": "bad", "role": "tool-mount"},
                {
                    "name": "left hand",
                    "link_name": "hand",
                    "role": "hand",
                    "interface_frame": {
                        "xyz": [0.0, 0.0, 0.0],
                        "rpy": [0.0, 0.0, 0.0],
                    },
                },
            ],
        },
    )

    result = load_attachment_manifest(model)

    assert [point.name for point in result.attachment_points] == ["left hand"]
    assert result.warnings
    assert "attachment_points[0].name" in result.warnings[0]


def test_model_library_exposes_imported_model_attachment_points(tmp_path: Path) -> None:
    from src.tools.model_explorer.model_library import ModelLibrary

    library = ModelLibrary(base_path=tmp_path / "models")
    imported = library._get_imported_models_path()
    model = _write_model(imported / "human.urdf")
    _write_manifest(
        model,
        {
            "schema_version": 1,
            "attachment_points": [
                {
                    "name": "hand mount",
                    "link_name": "hand",
                    "role": "tool-mount",
                    "interface_frame": {
                        "xyz": [0.0, 0.0, 0.1],
                        "rpy": [0.0, 0.0, 0.0],
                    },
                }
            ],
        },
    )

    models = library.discover_imported_models()

    assert models[0]["attachment_points"][0]["name"] == "hand mount"
    assert models[0]["attachment_points"][0]["link_name"] == "hand"


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def test_dialog_prefills_declared_interface_frame(qapp: QApplication) -> None:
    del qapp
    dialog = AttachmentPointSelector(
        ["base", "hand"],
        attachment_points=[
            {
                "name": "hand mount",
                "link_name": "hand",
                "role": "tool-mount",
                "interface_frame": {"xyz": [0.4, 0.5, 0.6], "rpy": [0.1, 0.2, 0.3]},
                "max_payload_kg": 1.5,
            }
        ],
    )

    config = dialog.get_configuration()

    assert config["parent_link"] == "hand"
    assert config["attachment_point"] == "hand mount"
    assert config["offset"] == (0.4, 0.5, 0.6)
    assert config["orientation"] == (0.1, 0.2, 0.3)


def test_declared_payload_warnings_include_exceeded_limits() -> None:
    warnings = declared_payload_warnings(
        [
            {
                "name": "hand mount",
                "link_name": "hand",
                "role": "tool-mount",
                "max_payload_kg": 1.0,
            },
            {
                "name": "flange",
                "link_name": "wrist",
                "role": "robot-flange",
                "max_payload_kg": 4.0,
            },
        ],
        payload_kg=2.5,
    )

    assert warnings == (
        "Attachment point 'hand mount' payload limit is 1.0 kg; selected payload is 2.5 kg.",
    )
