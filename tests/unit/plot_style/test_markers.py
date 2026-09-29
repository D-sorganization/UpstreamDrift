"""Unit tests for :class:`MarkerShape`, :class:`CustomMeshSpec`, :class:`MarkerStyle`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.plot_style import (
    CustomMeshSpec,
    MarkerShape,
    MarkerStyle,
    StaticColor,
)

# ---------- MarkerShape -------------------------------------------------


def test_marker_shape_round_trip_through_string() -> None:
    for shape in MarkerShape:
        assert str(shape) == shape.value
        assert MarkerShape(shape.value) is shape


# ---------- CustomMeshSpec ---------------------------------------------


def _tetra() -> CustomMeshSpec:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return CustomMeshSpec(name="tetra", vertices=vertices, faces=faces)


def test_custom_mesh_spec_happy_path() -> None:
    mesh = _tetra()
    assert mesh.name == "tetra"
    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape == (4, 3)


def test_custom_mesh_spec_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CustomMeshSpec(name="", vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))


def test_custom_mesh_spec_rejects_non_array_vertices() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        CustomMeshSpec(
            name="x",
            vertices=[[0, 0, 0]],  # type: ignore[arg-type]
            faces=np.zeros((1, 3), dtype=np.int32),
        )


def test_custom_mesh_spec_rejects_wrong_vertex_shape() -> None:
    with pytest.raises(ValueError, match=r"\(V, 3\)"):
        CustomMeshSpec(
            name="x",
            vertices=np.zeros((4, 2)),
            faces=np.zeros((1, 3), dtype=np.int32),
        )


def test_custom_mesh_spec_rejects_wrong_face_shape() -> None:
    with pytest.raises(ValueError, match=r"\(F, 3\)"):
        CustomMeshSpec(
            name="x",
            vertices=np.zeros((3, 3)),
            faces=np.zeros((1, 4), dtype=np.int32),
        )


def test_custom_mesh_spec_rejects_float_face_indices() -> None:
    with pytest.raises(TypeError, match="integer"):
        CustomMeshSpec(
            name="x",
            vertices=np.zeros((3, 3)),
            faces=np.zeros((1, 3), dtype=float),
        )


def test_custom_mesh_spec_rejects_oob_face_index() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        CustomMeshSpec(
            name="x",
            vertices=np.zeros((3, 3)),
            faces=np.array([[0, 1, 99]], dtype=np.int32),
        )


def test_custom_mesh_spec_rejects_negative_face_index() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        # Need int dtype since negative -> int kind
        CustomMeshSpec(
            name="x",
            vertices=np.zeros((3, 3)),
            faces=np.array([[0, 1, -1]], dtype=np.int32),
        )


def test_custom_mesh_spec_rejects_non_string_vertices() -> None:
    with pytest.raises(TypeError, match="vertices dtype"):
        CustomMeshSpec(
            name="x",
            vertices=np.array([["a", "b", "c"]], dtype=object),
            faces=np.array([[0, 0, 0]], dtype=np.int32),
        )


def test_custom_mesh_spec_rejects_non_array_faces() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        CustomMeshSpec(
            name="x",
            vertices=np.zeros((3, 3)),
            faces=[[0, 1, 2]],  # type: ignore[arg-type]
        )


# ---------- MarkerStyle -------------------------------------------------


def test_marker_style_default_construction() -> None:
    style = MarkerStyle()
    assert style.shape is MarkerShape.SPHERE
    assert math.isclose(style.size_px, 6.0)
    assert isinstance(style.fill_color, StaticColor)


def test_marker_style_explicit_fields() -> None:
    style = MarkerStyle(
        shape=MarkerShape.CUBE,
        size_px=10.0,
        edge_color="red",
        edge_width=1.5,
        fill_color=StaticColor("#00ff00"),
        opacity=0.5,
    )
    assert style.shape is MarkerShape.CUBE
    assert style.size_px == 10.0
    assert style.edge_color == "red"


def test_marker_style_rejects_non_enum_shape() -> None:
    with pytest.raises(TypeError, match="MarkerShape"):
        MarkerStyle(shape="sphere")  # type: ignore[arg-type]


def test_marker_style_rejects_zero_size() -> None:
    with pytest.raises(ValueError, match="> 0"):
        MarkerStyle(size_px=0.0)


def test_marker_style_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="> 0"):
        MarkerStyle(size_px=-1.0)


def test_marker_style_rejects_non_finite_size() -> None:
    with pytest.raises(ValueError, match="finite"):
        MarkerStyle(size_px=math.inf)


def test_marker_style_rejects_non_numeric_size() -> None:
    with pytest.raises(TypeError, match="numeric"):
        MarkerStyle(size_px="6")  # type: ignore[arg-type]


def test_marker_style_rejects_negative_edge_width() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        MarkerStyle(edge_width=-0.1)


def test_marker_style_allows_zero_edge_width() -> None:
    style = MarkerStyle(edge_width=0.0)
    assert style.edge_width == 0.0


def test_marker_style_rejects_non_numeric_edge_width() -> None:
    with pytest.raises(TypeError, match="numeric"):
        MarkerStyle(edge_width="0.5")  # type: ignore[arg-type]


def test_marker_style_rejects_opacity_above_one() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MarkerStyle(opacity=1.5)


def test_marker_style_rejects_opacity_below_zero() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MarkerStyle(opacity=-0.1)


def test_marker_style_rejects_non_numeric_opacity() -> None:
    with pytest.raises(TypeError, match="numeric"):
        MarkerStyle(opacity="full")  # type: ignore[arg-type]


def test_marker_style_rejects_empty_edge_color() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MarkerStyle(edge_color="")


def test_marker_style_rejects_unparseable_edge_color() -> None:
    with pytest.raises(ValueError, match="parseable"):
        MarkerStyle(edge_color="not_a_color")


def test_marker_style_rejects_non_color_scale_fill() -> None:
    with pytest.raises(TypeError, match="ColorScale"):
        MarkerStyle(fill_color="#ff0000")  # type: ignore[arg-type]


def test_marker_style_custom_mesh_requires_mesh() -> None:
    with pytest.raises(ValueError, match="custom_mesh"):
        MarkerStyle(shape=MarkerShape.CUSTOM_MESH)


def test_marker_style_custom_mesh_pair_happy_path() -> None:
    mesh = _tetra()
    style = MarkerStyle(shape=MarkerShape.CUSTOM_MESH, custom_mesh=mesh)
    assert style.custom_mesh is mesh


def test_marker_style_mesh_without_custom_mesh_shape_fails() -> None:
    mesh = _tetra()
    with pytest.raises(ValueError, match="only allowed"):
        MarkerStyle(shape=MarkerShape.SPHERE, custom_mesh=mesh)


def test_marker_style_rejects_non_mesh_object() -> None:
    with pytest.raises(ValueError, match="custom_mesh"):
        # We pass a non-CustomMeshSpec object with shape=CUSTOM_MESH; the
        # type-check raises only after the iff check. Use SPHERE+object
        # instead to hit the type branch.
        MarkerStyle(shape=MarkerShape.CUSTOM_MESH, custom_mesh=None)


def test_marker_style_custom_mesh_wrong_type_branch() -> None:
    # MarkerShape is CUSTOM_MESH but custom_mesh is not CustomMeshSpec.
    with pytest.raises(TypeError, match="CustomMeshSpec"):
        MarkerStyle(
            shape=MarkerShape.CUSTOM_MESH,
            custom_mesh="not a mesh",  # type: ignore[arg-type]
        )
