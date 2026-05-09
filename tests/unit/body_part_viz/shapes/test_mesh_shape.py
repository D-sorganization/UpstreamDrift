"""Unit tests for ``body_part_viz.shapes.mesh_shape``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from src.shared.python.body_part_viz import BodyPartShape
from src.shared.python.body_part_viz.shapes import MeshShape


def _box_mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(1.0, 2.0, 3.0))


def _write(mesh: trimesh.Trimesh, path: Path) -> None:
    mesh.export(str(path))


@pytest.mark.parametrize("ext", ["stl", "obj", "ply", "glb"])
def test_load_box_round_trip(tmp_path: Path, ext: str) -> None:
    box = _box_mesh()
    path = tmp_path / f"box.{ext}"
    _write(box, path)

    shape = MeshShape.load(path)

    assert isinstance(shape, BodyPartShape)
    assert shape.vertices_at_rest().shape[1] == 3
    assert shape.faces().shape[1] == 3
    assert shape.faces().dtype == np.int64
    # OBB extents of an axis-aligned box equal its (sorted) extents.
    assert sorted(shape.rest_dimensions) == pytest.approx(
        sorted([1.0, 2.0, 3.0]), abs=1e-6
    )
    # Re-centred on OBB centroid: vertices should be roughly symmetric.
    centroid = shape.vertices_at_rest().mean(axis=0)
    assert np.allclose(centroid, np.zeros(3), atol=1e-6)


def test_load_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MeshShape.load(tmp_path / "does_not_exist.stl")


def test_load_gltf_rejected(tmp_path: Path) -> None:
    fake = tmp_path / "fake.gltf"
    fake.write_text("{}")
    with pytest.raises(ValueError, match="use .glb|GLTF"):
        MeshShape.load(fake)


def test_load_unsupported_extension(tmp_path: Path) -> None:
    fake = tmp_path / "fake.xyz"
    fake.write_text("nope")
    with pytest.raises(ValueError, match="unsupported mesh extension"):
        MeshShape.load(fake)


def test_load_decimates_to_budget(tmp_path: Path) -> None:
    sphere = trimesh.creation.icosphere(subdivisions=6)
    assert len(sphere.vertices) > 5000
    path = tmp_path / "sphere.ply"
    sphere.export(str(path))

    shape = MeshShape.load(path, max_vertices=500)
    assert len(shape.vertices_at_rest()) <= 500


def test_load_max_vertices_below_floor(tmp_path: Path) -> None:
    box = _box_mesh()
    p = tmp_path / "b.stl"
    box.export(str(p))
    with pytest.raises(ValueError, match="max_vertices"):
        MeshShape.load(p, max_vertices=1)


def test_load_invalid_strategy(tmp_path: Path) -> None:
    sphere = trimesh.creation.icosphere(subdivisions=4)
    p = tmp_path / "s.stl"
    sphere.export(str(p))
    with pytest.raises(ValueError, match="decimation_strategy"):
        MeshShape.load(p, max_vertices=100, decimation_strategy="bogus")  # type: ignore[arg-type]


def test_load_uniform_strategy(tmp_path: Path) -> None:
    sphere = trimesh.creation.icosphere(subdivisions=5)
    p = tmp_path / "s.ply"
    sphere.export(str(p))
    shape = MeshShape.load(p, max_vertices=300, decimation_strategy="uniform")
    assert len(shape.vertices_at_rest()) <= 300


def test_constructor_validates_vertices_shape() -> None:
    with pytest.raises(ValueError, match="vertices"):
        MeshShape(
            vertices=np.zeros((4, 2)),
            faces=np.zeros((1, 3), dtype=np.int64),
            rest_dimensions=(1.0, 1.0, 1.0),
        )


def test_constructor_validates_vertices_type() -> None:
    with pytest.raises(TypeError, match="vertices"):
        MeshShape(
            vertices=[[0, 0, 0]],  # type: ignore[arg-type]
            faces=np.zeros((1, 3), dtype=np.int64),
            rest_dimensions=(1.0, 1.0, 1.0),
        )


def test_constructor_validates_faces_type() -> None:
    with pytest.raises(TypeError, match="faces"):
        MeshShape(
            vertices=np.zeros((4, 3)),
            faces=[[0, 1, 2]],  # type: ignore[arg-type]
            rest_dimensions=(1.0, 1.0, 1.0),
        )


def test_constructor_validates_faces_shape() -> None:
    with pytest.raises(ValueError, match="faces"):
        MeshShape(
            vertices=np.zeros((4, 3)),
            faces=np.zeros((1, 4), dtype=np.int64),
            rest_dimensions=(1.0, 1.0, 1.0),
        )


def test_constructor_rejects_empty_vertices() -> None:
    with pytest.raises(ValueError, match="vertex"):
        MeshShape(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int64),
            rest_dimensions=(1.0, 1.0, 1.0),
        )


def test_constructor_rejects_empty_faces() -> None:
    with pytest.raises(ValueError, match="face"):
        MeshShape(
            vertices=np.zeros((4, 3)),
            faces=np.zeros((0, 3), dtype=np.int64),
            rest_dimensions=(1.0, 1.0, 1.0),
        )


def test_constructor_rejects_bad_rest_dimensions() -> None:
    with pytest.raises(ValueError, match="rest_dimensions"):
        MeshShape(
            vertices=np.zeros((4, 3)),
            faces=np.zeros((1, 3), dtype=np.int64),
            rest_dimensions=(1.0, 2.0),  # type: ignore[arg-type]
        )


def test_constructor_rejects_zero_rest_dimensions() -> None:
    with pytest.raises(ValueError, match="rest_dimensions"):
        MeshShape(
            vertices=np.zeros((4, 3)),
            faces=np.zeros((1, 3), dtype=np.int64),
            rest_dimensions=(1.0, 0.0, 1.0),
        )


def test_constructor_rejects_empty_shape_id() -> None:
    with pytest.raises(ValueError, match="shape_id"):
        MeshShape(
            vertices=np.zeros((4, 3)),
            faces=np.zeros((1, 3), dtype=np.int64),
            rest_dimensions=(1.0, 1.0, 1.0),
            shape_id="",
        )


def test_default_shape_id_anonymous() -> None:
    shape = MeshShape(
        vertices=np.zeros((4, 3)),
        faces=np.zeros((1, 3), dtype=np.int64),
        rest_dimensions=(1.0, 1.0, 1.0),
    )
    assert shape.shape_id == "mesh:anonymous"


def test_default_shape_id_from_source_stem() -> None:
    shape = MeshShape(
        vertices=np.zeros((4, 3)),
        faces=np.zeros((1, 3), dtype=np.int64),
        rest_dimensions=(1.0, 1.0, 1.0),
        source_path=Path("/tmp/femur.stl"),
    )
    assert shape.shape_id == "mesh:femur"


def test_transform_single_frame(tmp_path: Path) -> None:
    from src.shared.python.body_part_viz import (
        BindingKind,
        FittedShape,
        MarkerBinding,
    )

    path = tmp_path / "b.stl"
    _write(_box_mesh(), path)
    shape = MeshShape.load(path)

    binding = MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", "b"))
    fitted = FittedShape(
        shape_id=shape.shape_id,
        binding=binding,
        centroid=np.array([[10.0, 0.0, 0.0]]),
        rotation_matrix=np.eye(3)[None, :, :],
        scale=np.ones((1, 3)),
        valid_mask=np.ones((1,), dtype=bool),
    )
    out = shape.transform(fitted)
    assert out.shape == (len(shape.vertices_at_rest()), 3)
    assert np.allclose(out.mean(axis=0), np.array([10.0, 0.0, 0.0]), atol=1e-6)


def test_transform_multi_frame(tmp_path: Path) -> None:
    from src.shared.python.body_part_viz import (
        BindingKind,
        FittedShape,
        MarkerBinding,
    )

    path = tmp_path / "b.stl"
    _write(_box_mesh(), path)
    shape = MeshShape.load(path)

    binding = MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", "b"))
    n = 3
    fitted = FittedShape(
        shape_id=shape.shape_id,
        binding=binding,
        centroid=np.zeros((n, 3)),
        rotation_matrix=np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
        scale=np.ones((n, 3)) * 2.0,
        valid_mask=np.ones((n,), dtype=bool),
    )
    out = shape.transform(fitted)
    assert out.shape == (n, len(shape.vertices_at_rest()), 3)
    # 2x scaling doubles vertex magnitudes.
    assert np.allclose(out[0], shape.vertices_at_rest() * 2.0, atol=1e-6)


def test_transform_zero_frames(tmp_path: Path) -> None:
    from src.shared.python.body_part_viz import (
        BindingKind,
        FittedShape,
        MarkerBinding,
    )

    path = tmp_path / "b.stl"
    _write(_box_mesh(), path)
    shape = MeshShape.load(path)

    binding = MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", "b"))
    fitted = FittedShape(
        shape_id=shape.shape_id,
        binding=binding,
        centroid=np.zeros((0, 3)),
        rotation_matrix=np.zeros((0, 3, 3)),
        scale=np.zeros((0, 3)),
        valid_mask=np.zeros((0,), dtype=bool),
    )
    out = shape.transform(fitted)
    assert out.shape == (0, len(shape.vertices_at_rest()), 3)


def test_transform_mismatched_shape_id(tmp_path: Path) -> None:
    from src.shared.python.body_part_viz import (
        BindingKind,
        FittedShape,
        MarkerBinding,
    )

    path = tmp_path / "b.stl"
    _write(_box_mesh(), path)
    shape = MeshShape.load(path)

    binding = MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", "b"))
    fitted = FittedShape(
        shape_id="other",
        binding=binding,
        centroid=np.zeros((1, 3)),
        rotation_matrix=np.eye(3)[None, :, :],
        scale=np.ones((1, 3)),
        valid_mask=np.ones((1,), dtype=bool),
    )
    with pytest.raises(ValueError, match="shape_id"):
        shape.transform(fitted)


def test_load_malformed_stl(tmp_path: Path) -> None:
    bad = tmp_path / "bad.stl"
    bad.write_bytes(b"not a real STL file")
    with pytest.raises(ValueError):
        MeshShape.load(bad)


def test_load_empty_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scene container with no triangle meshes should ValueError."""
    from src.shared.python.body_part_viz.shapes import _mesh_io

    box_path = tmp_path / "b.glb"
    _box_mesh().export(str(box_path))

    def _fake_load(*_args: object, **_kwargs: object) -> trimesh.Scene:
        return trimesh.Scene()

    monkeypatch.setattr(_mesh_io.trimesh, "load", _fake_load, raising=True)
    with pytest.raises(ValueError, match="no triangle mesh"):
        MeshShape.load(box_path)


def test_load_propagates_trimesh_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.shared.python.body_part_viz.shapes import _mesh_io

    p = tmp_path / "b.stl"
    _box_mesh().export(str(p))

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk on fire")

    monkeypatch.setattr(_mesh_io.trimesh, "load", _raise, raising=True)
    with pytest.raises(ValueError, match="failed to read"):
        MeshShape.load(p)


def test_load_rejects_empty_vertex_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.shared.python.body_part_viz.shapes import _mesh_io

    p = tmp_path / "b.stl"
    _box_mesh().export(str(p))

    empty = trimesh.Trimesh(
        vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64)
    )
    monkeypatch.setattr(_mesh_io.trimesh, "load", lambda *a, **k: empty, raising=True)
    with pytest.raises(ValueError, match="no vertices"):
        MeshShape.load(p)


def test_load_rejects_no_faces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.shared.python.body_part_viz.shapes import _mesh_io

    p = tmp_path / "b.stl"
    _box_mesh().export(str(p))

    no_faces = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.zeros((0, 3), dtype=np.int64),
    )
    monkeypatch.setattr(
        _mesh_io.trimesh, "load", lambda *a, **k: no_faces, raising=True
    )
    with pytest.raises(ValueError, match="no faces"):
        MeshShape.load(p)


def test_load_unknown_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.shared.python.body_part_viz.shapes import _mesh_io

    p = tmp_path / "b.stl"
    _box_mesh().export(str(p))

    def _fake_load(*_args: object, **_kwargs: object) -> dict[str, int]:
        return {"unexpected": 1}

    monkeypatch.setattr(_mesh_io.trimesh, "load", _fake_load, raising=True)
    with pytest.raises(ValueError, match="unsupported mesh container"):
        MeshShape.load(p)


def test_load_100k_vertex_mesh_decimated(tmp_path: Path) -> None:
    sphere = trimesh.creation.icosphere(subdivisions=7)
    assert len(sphere.vertices) >= 100_000 or len(sphere.vertices) > 5000
    path = tmp_path / "huge.ply"
    sphere.export(str(path))
    shape = MeshShape.load(path, max_vertices=5000)
    assert len(shape.vertices_at_rest()) <= 5000
