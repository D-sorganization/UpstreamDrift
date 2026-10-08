"""Visible head for the native viewers: OBJ export and engine registration."""

import json
from pathlib import Path

import pytest

from src.tools.native_viewer_export.backends._head import head_mesh_files

pytestmark = pytest.mark.unit
SPECS = Path(__file__).resolve().parents[4] / "docs/development/full_body_models"


def test_head_obj_files_are_written_in_the_head_body_frame(tmp_path: Path) -> None:
    spec = json.loads((SPECS / "full_body_spec_anthro_driver.json").read_text())
    files = head_mesh_files(spec, tmp_path)
    names = {f.name for f in files}
    assert {"skull", "neck", "nose", "eye_l", "eye_r", "ear_l", "ear_r"} <= names
    assert all(f.body.endswith("/Head") and f.path.is_file() for f in files)
    assert all(len(f.rgba) == 4 for f in files)
    first = (tmp_path / "head_skull.obj").read_text().splitlines()
    assert first[0].startswith("v ") and any(line.startswith("f ") for line in first)


def test_spec_without_a_head_body_yields_no_files(tmp_path: Path) -> None:
    spec = json.loads((SPECS / "full_body_spec_v1.json").read_text())
    assert head_mesh_files(spec, tmp_path) == []


def test_opensim_model_gets_head_meshes_and_no_head_capsule() -> None:
    osim = pytest.importorskip("opensim")
    from src.tools.native_viewer_export.backends import opensim_worker

    raw = (SPECS / "full_body_spec_anthro_driver.json").read_bytes()
    model, _ = opensim_worker.build_model(osim, raw)
    head = model.getBodySet().get("Head")
    names = [c.getName() for c in head.getComponentsList()]
    assert sum(n.startswith("vf") for n in names) >= 12  # meshes, not 3 capsule parts
