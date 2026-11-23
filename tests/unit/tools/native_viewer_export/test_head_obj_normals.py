"""Native hair shading preserves the shared surface and its body frame."""

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import (
    build_head_parts,
    head_frame_in_body,
    place_parts,
    resolve_head_anchor,
)
from src.shared.python.model_appearance.body_skin_mesh import vertex_normals
from src.shared.python.model_appearance.head import drawn_head_length
from src.shared.python.model_appearance.schema import AppearanceDocument, HeadSettings
from src.tools.native_viewer_export.backends._head import head_mesh_files, write_obj

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.mark.parametrize("model", ["cc0_mesh", "parametric"])
def test_hair_obj_has_unit_normals_in_the_placed_head_frame(
    tmp_path: Path, model: str
) -> None:
    spec = {"bodies": [{"name": "golfer/Head"}], "subject": {"stature_m": 1.8}}
    doc = AppearanceDocument(
        head=HeadSettings(model=model, forward_axis="-y", up_axis="+z")
    )
    anchor = resolve_head_anchor(spec)
    assert anchor is not None
    parts = place_parts(
        build_head_parts(drawn_head_length(spec, anchor, doc.head), doc.head),
        head_frame_in_body(anchor, doc.head),
    )
    meshes = {part.name: part.mesh for part in parts}
    for exported in head_mesh_files(spec, tmp_path, doc):
        lines = exported.path.read_text().splitlines()
        mesh = meshes[exported.name]
        vertices = np.array(
            [line.split()[1:] for line in lines if line.startswith("v ")], dtype=float
        )
        np.testing.assert_allclose(vertices, mesh.vertices, atol=5.1e-7, rtol=0)
        faces = [line.split()[1:] for line in lines if line.startswith("f ")]
        if exported.name != "hair_mesh":
            assert not any(line.startswith("vn ") for line in lines)
            assert faces == [[str(i + 1) for i in face] for face in mesh.faces]
            continue
        normals = np.array(
            [line.split()[1:] for line in lines if line.startswith("vn ")], dtype=float
        )
        assert normals.shape == mesh.vertices.shape
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(
            normals, vertex_normals(mesh.vertices, mesh.faces), atol=5.1e-7
        )
        assert "s 1" in lines
        assert faces == [[f"{i + 1}//{i + 1}" for i in face] for face in mesh.faces]
        first = exported.path.read_bytes()
        head_mesh_files(spec, tmp_path, doc)
        assert exported.path.read_bytes() == first


def test_default_obj_writer_preserves_plain_face_indices(tmp_path: Path) -> None:
    path = tmp_path / "plain.obj"
    write_obj(path, [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    assert path.read_text().splitlines()[-1] == "f 1 2 3"


def test_smooth_obj_rejects_degenerate_normals_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "bad.obj"
    with pytest.raises(ValueError, match="nonzero vertex normals"):
        write_obj(path, np.zeros((3, 3)), np.array([[0, 1, 2]]), smooth=True)
    assert not path.exists()
