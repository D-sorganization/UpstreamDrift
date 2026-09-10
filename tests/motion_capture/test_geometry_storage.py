"""Capture geometry travels with its evidence and rejects changed scene data."""

import shutil
from pathlib import Path

import pytest

from src.motion_capture.coaching.geometry import ReferenceGeometry, ReferencePoint
from src.motion_capture.coaching.geometry_storage import load_geometry, save_geometry

pytestmark = pytest.mark.unit


def test_portable_capture_sidecar_and_changed_evidence(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    (root / "recordings.json").write_text('{"recordings": []}', encoding="utf-8")
    initial = load_geometry(root)
    edited = ReferenceGeometry(
        scene_id=initial.scene_id, points=(ReferencePoint(position_m=(1, 2, 3)),)
    )
    save_geometry(root, edited)
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    assert load_geometry(moved) == edited
    (moved / "intrinsics.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="scene"):
        load_geometry(moved)
    with pytest.raises(ValueError, match="scene"):
        save_geometry(moved, edited)


def test_missing_capture_evidence_is_not_fabricated(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_geometry(tmp_path)


def test_actual_reconstruction_camera_changes_invalidate_scene(tmp_path: Path) -> None:
    (tmp_path / "recordings.json").write_text("{}")
    camera = tmp_path / "reconstruct" / "reconstruction.json"
    camera.parent.mkdir()
    camera.write_text('{"cameras": []}')
    saved = load_geometry(tmp_path)
    save_geometry(tmp_path, saved)
    camera.write_text('{"cameras": [], "revision": 2}')
    with pytest.raises(ValueError, match="scene"):
        load_geometry(tmp_path)
