"""Shared body-mesh OBJ export for the native viewers (R8, #12685).

``_body.body_mesh_files`` writes one OBJ per resolved ``BodyMesh`` in its spec
body frame with the library RGBA, mirroring ``_head.head_mesh_files``. The
MeshCat and OpenSim backends register these files in R9a/R9b.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import document_from_dict, library_materials
from src.shared.python.model_appearance.body_meshes import resolve_appearance
from src.tools.native_viewer_export.backends._body import body_mesh_files

pytestmark = pytest.mark.unit
SPECS = Path(__file__).resolve().parents[4] / "docs/development/full_body_models"


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        tag, *rest = line.split()
        if tag == "v":
            verts.append([float(x) for x in rest])
        elif tag == "f":
            faces.append([int(x) - 1 for x in rest])
    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


@pytest.fixture(scope="module")
def resolved():
    spec = json.loads((SPECS / "full_body_spec_anthro_driver.json").read_text())
    return resolve_appearance(
        spec, document_from_dict({"schema_version": "appearance-v1"})
    )


def test_one_obj_per_body_mesh_round_trips_its_vertices(resolved, tmp_path) -> None:
    files = body_mesh_files(resolved, tmp_path)
    assert len(files) == len(resolved.meshes)
    assert len({f.path for f in files}) == len(files)
    for item, mesh in zip(files, resolved.meshes, strict=True):
        assert item.body == mesh.body and item.path.is_file()
        verts, faces = _read_obj(item.path)
        expected = mesh.mesh_in_body_frame
        np.testing.assert_allclose(verts, expected.vertices, atol=1e-6)
        assert np.array_equal(faces, expected.faces)


def test_obj_colours_come_from_the_library(resolved, tmp_path) -> None:
    materials = library_materials(
        document_from_dict({"schema_version": "appearance-v1"})
    )
    for item, mesh in zip(
        body_mesh_files(resolved, tmp_path), resolved.meshes, strict=True
    ):
        assert item.material_name == mesh.material_name
        assert item.rgba == materials[mesh.material_name].rgba


def test_empty_resolution_writes_nothing(resolved, tmp_path) -> None:
    import dataclasses

    empty = dataclasses.replace(resolved, meshes=())
    assert body_mesh_files(empty, tmp_path / "none") == []
