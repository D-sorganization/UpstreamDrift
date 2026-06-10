"""Tests for model explorer attachment sidecar manifests."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from src.tools.model_explorer.attachment_metadata import (
    attachment_warnings_for_link,
    load_attachment_manifest,
    sidecar_path_for_model,
)
from src.tools.model_explorer.frankenstein_editor.model import URDFModel
from src.tools.model_explorer.model_library import ModelLibrary
from tests.tools.model_explorer._fixtures import SIMPLE_URDF


def _write_model(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_path = tmp_path / "robot.urdf"
    model_path.write_text(SIMPLE_URDF, encoding="utf-8")
    return model_path


def _write_sidecar(model_path: Path, data: dict) -> Path:
    sidecar_path = sidecar_path_for_model(model_path)
    sidecar_path.write_text(json.dumps(data), encoding="utf-8")
    return sidecar_path


def _manifest_data() -> dict:
    return {
        "schema_version": 1,
        "attachments": [
            {
                "link_name": "hand",
                "role": "hand",
                "interface_frame": {
                    "xyz": [0.1, 0.2, 0.3],
                    "rpy": [0.0, 1.57, 0.0],
                },
                "max_payload_kg": 2.5,
                "tags": ["right", "tool"],
            }
        ],
    }


def test_load_attachment_manifest_parses_declared_points(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    _write_sidecar(model_path, _manifest_data())

    manifest = load_attachment_manifest(model_path, known_links={"base", "arm", "hand"})

    assert manifest.warnings == ()
    assert len(manifest.attachment_points) == 1
    point = manifest.get("hand")
    assert point is not None
    assert point.role == "hand"
    assert point.interface_frame.xyz == (0.1, 0.2, 0.3)
    assert point.max_payload_kg == 2.5
    assert point.tags == ("right", "tool")


def test_missing_attachment_sidecar_is_empty_and_nonfatal(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)

    manifest = load_attachment_manifest(model_path)

    assert manifest.attachment_points == ()
    assert manifest.warnings == ()


def test_malformed_attachment_sidecar_warns_without_crashing(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    sidecar_path_for_model(model_path).write_text("{not-json", encoding="utf-8")

    manifest = load_attachment_manifest(model_path)

    assert manifest.attachment_points == ()
    assert any(
        "failed to read attachment sidecar" in warning for warning in manifest.warnings
    )


def test_invalid_attachment_entries_warn_and_skip_bad_entries(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    _write_sidecar(
        model_path,
        {
            "schema_version": 1,
            "attachments": [
                {"link_name": "", "role": "hand"},
                {"link_name": "ghost", "role": "tool-mount"},
            ],
        },
    )

    manifest = load_attachment_manifest(model_path, known_links={"base"})

    assert [point.link_name for point in manifest.attachment_points] == ["ghost"]
    assert any(
        "link_name must be a non-empty string" in warning
        for warning in manifest.warnings
    )
    assert any(
        "'ghost' is not in the model" in warning for warning in manifest.warnings
    )


def test_model_library_exposes_attachment_metadata_for_imported_models(
    tmp_path: Path,
) -> None:
    model_path = _write_model(tmp_path / "source")
    _write_sidecar(model_path, _manifest_data())
    library = ModelLibrary(base_path=tmp_path / "library")
    imported_path = library.import_model(str(model_path))
    assert imported_path is not None
    _write_sidecar(imported_path, _manifest_data())

    models = library.discover_imported_models()
    imported = next(model for model in models if model["name"] == "robot.urdf")

    assert imported["attachments"]["attachment_points"][0]["link_name"] == "hand"
    assert "attachment_warnings" not in imported


def test_loaded_model_uses_declared_interface_frame_for_attachment_joint(
    tmp_path: Path,
) -> None:
    model_path = _write_model(tmp_path)
    _write_sidecar(model_path, _manifest_data())
    model = URDFModel.from_file(model_path)
    model.add_link(ET.Element("link", {"name": "club"}))

    joint_name, warnings = model.add_attachment_joint(
        parent_link="hand",
        child_link="club",
        payload_kg=2.0,
    )

    assert warnings == ()
    origin = model.joints[joint_name].find("origin")
    assert origin is not None
    assert origin.get("xyz") == "0.1 0.2 0.3"
    assert origin.get("rpy") == "0 1.57 0"


def test_editor_attachment_warnings_for_undeclared_link_and_payload(
    tmp_path: Path,
) -> None:
    model_path = _write_model(tmp_path)
    _write_sidecar(model_path, _manifest_data())
    model = URDFModel.from_file(model_path)

    undeclared = model.attachment_warnings("arm")
    too_heavy = model.attachment_warnings("hand", payload_kg=3.0)

    assert undeclared[0].severity == "warning"
    assert "not a declared attachment point" in undeclared[0].message
    assert "exceeds max_payload_kg" in too_heavy[0].message


def test_attachment_warning_helper_allows_no_manifest_links(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path)
    manifest = load_attachment_manifest(model_path)

    assert attachment_warnings_for_link(manifest, "base") == ()
