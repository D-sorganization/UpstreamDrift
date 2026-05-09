"""Tests for shapes._mesh_primitives."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz.shapes._mesh_primitives import (
    make_cylinder,
    make_ellipsoid,
    make_uv_sphere,
)


def test_uv_sphere_vertex_count() -> None:
    verts, faces = make_uv_sphere(16, 8)
    assert verts.shape == (16 * 9, 3)
    assert faces.shape[1] == 3
    assert faces.shape[0] == 2 * 16 * 8


def test_uv_sphere_unit_radius() -> None:
    verts, _ = make_uv_sphere(20, 10)
    norms = np.linalg.norm(verts, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-9)


def test_uv_sphere_validates_facets() -> None:
    with pytest.raises(ValueError):
        make_uv_sphere(2, 4)
    with pytest.raises(ValueError):
        make_uv_sphere(8, 1)
    with pytest.raises(TypeError):
        make_uv_sphere(8.0, 4)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        make_uv_sphere(8, 4.0)  # type: ignore[arg-type]


def test_cylinder_counts() -> None:
    verts, faces = make_cylinder(1.0, 0.5, 16)
    assert verts.shape == (2 * (16 + 1), 3)
    assert faces.shape == (4 * 16, 3)


def test_cylinder_watertight_indices_in_range() -> None:
    verts, faces = make_cylinder(1.0, 0.5, 16)
    assert int(faces.max()) == verts.shape[0] - 1
    assert int(faces.min()) == 0


def test_cylinder_validation() -> None:
    with pytest.raises(ValueError):
        make_cylinder(0.0, 0.5, 16)
    with pytest.raises(ValueError):
        make_cylinder(1.0, -0.1, 16)
    with pytest.raises(ValueError):
        make_cylinder(1.0, 0.5, 2)


def test_ellipsoid_axis_ratio() -> None:
    a, b, c = 2.0, 1.0, 0.5
    verts, _ = make_ellipsoid(a, b, c, 16, 8)
    eq = (verts[:, 0] / a) ** 2 + (verts[:, 1] / b) ** 2 + (verts[:, 2] / c) ** 2
    np.testing.assert_allclose(eq, 1.0, atol=1e-9)


def test_ellipsoid_validation() -> None:
    with pytest.raises(ValueError):
        make_ellipsoid(0.0, 1.0, 1.0, 16, 8)
    with pytest.raises(ValueError):
        make_ellipsoid(1.0, -1.0, 1.0, 16, 8)
