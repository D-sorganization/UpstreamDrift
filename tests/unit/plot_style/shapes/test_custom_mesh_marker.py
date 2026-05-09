"""Unit tests for :class:`CustomMeshMarker`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.plot_style import (
    CustomMeshSpec,
    MarkerShape,
    MarkerShapeRenderer,
    MarkerStyle,
)
from src.shared.python.plot_style.shapes import CustomMeshMarker


def _tetra_spec() -> CustomMeshSpec:
    """A tetrahedron spec — 4 vertices, 4 triangles."""
    verts = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
        dtype=np.int64,
    )
    return CustomMeshSpec(name="tetra", vertices=verts, faces=faces)


def test_default_counts_from_spec() -> None:
    m = CustomMeshMarker(_tetra_spec())
    v, f = m.mesh(MarkerStyle(shape=MarkerShape.CUSTOM_MESH, custom_mesh=_tetra_spec()))
    assert v.shape == (4, 3)
    assert f.shape == (4, 3)


def test_unit_radius_after_normalisation() -> None:
    spec = _tetra_spec()
    # Inflate the spec to a non-unit size; normalisation should rescale.
    big = CustomMeshSpec(
        name="big_tetra",
        vertices=spec.vertices * 17.0,
        faces=spec.faces,
    )
    m = CustomMeshMarker(big)
    v, _ = m.mesh(
        MarkerStyle(shape=MarkerShape.CUSTOM_MESH, custom_mesh=big, size_px=2.0)
    )
    radii = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(radii.max(), 1.0, atol=1e-9)


def test_scale_linearity() -> None:
    spec = _tetra_spec()
    m = CustomMeshMarker(spec)
    v1, _ = m.mesh(
        MarkerStyle(shape=MarkerShape.CUSTOM_MESH, custom_mesh=spec, size_px=2.0)
    )
    v2, _ = m.mesh(
        MarkerStyle(shape=MarkerShape.CUSTOM_MESH, custom_mesh=spec, size_px=4.0)
    )
    np.testing.assert_allclose(v2, 2.0 * v1, atol=1e-12)


def test_protocol_runtime_check() -> None:
    assert isinstance(CustomMeshMarker(_tetra_spec()), MarkerShapeRenderer)


def test_shape_id() -> None:
    assert CustomMeshMarker.shape_id == MarkerShape.CUSTOM_MESH.value


def test_invalid_source_type() -> None:
    with pytest.raises(TypeError):
        CustomMeshMarker(123)  # type: ignore[arg-type]


def test_zero_extent_mesh_rejected() -> None:
    degenerate = CustomMeshSpec(
        name="point",
        vertices=np.zeros((3, 3), dtype=np.float64),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="zero extent"):
        CustomMeshMarker(degenerate)


def test_mesh_rejects_non_style() -> None:
    m = CustomMeshMarker(_tetra_spec())
    with pytest.raises(TypeError):
        m.mesh("style")  # type: ignore[arg-type]


def test_mesh_rejects_non_custom_mesh_style() -> None:
    m = CustomMeshMarker(_tetra_spec())
    with pytest.raises(ValueError, match="CUSTOM_MESH"):
        m.mesh(MarkerStyle(shape=MarkerShape.SPHERE))


def test_load_from_obj_path(tmp_path: Path) -> None:
    """OBJ loading round-trip via :func:`load_mesh`."""
    obj_path = tmp_path / "cube.obj"
    obj_path.write_text(
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "v 0 0 1\n"
        "v 1 0 1\n"
        "v 1 1 1\n"
        "v 0 1 1\n"
        "f 1 2 3\n"
        "f 1 3 4\n"
        "f 5 7 6\n"
        "f 5 8 7\n"
        "f 1 5 6\n"
        "f 1 6 2\n"
        "f 2 6 7\n"
        "f 2 7 3\n"
        "f 3 7 8\n"
        "f 3 8 4\n"
        "f 4 8 5\n"
        "f 4 5 1\n"
    )
    m = CustomMeshMarker(obj_path)
    assert m.name == "cube"
    assert m.spec.vertices.shape == (8, 3)
    v, _ = m.mesh(
        MarkerStyle(shape=MarkerShape.CUSTOM_MESH, custom_mesh=m.spec, size_px=2.0)
    )
    radii = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(radii.max(), 1.0, atol=1e-9)


def test_load_from_path_string(tmp_path: Path) -> None:
    """String paths are accepted alongside :class:`Path`."""
    obj_path = tmp_path / "tri.obj"
    obj_path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 3 4\nf 1 4 2\nf 2 4 3\n"
    )
    m = CustomMeshMarker(str(obj_path), name="custom_name")
    assert m.name == "custom_name"
