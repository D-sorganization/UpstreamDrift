"""Unit tests for :mod:`src.shared.python.golf_viz` geometry builders.

Pure-numpy mesh/line constructors shared by every golf simulation GUI.
No Qt/OpenGL import is required to exercise them.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.golf_viz import (
    circle_fan_vertices,
    disc_mesh,
    flagstick_lines,
    grid_surface_mesh,
    rect_vertices,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# rect_vertices
# ---------------------------------------------------------------------------


def test_rect_vertices_two_triangles() -> None:
    verts = rect_vertices(0.0, 0.0, 2.0, 4.0)
    assert verts.shape == (6, 3)


def test_rect_vertices_span_corners() -> None:
    verts = rect_vertices(1.0, 2.0, 3.0, 5.0, z=0.7)
    assert verts[:, 0].min() == pytest.approx(1.0)
    assert verts[:, 0].max() == pytest.approx(4.0)
    assert verts[:, 1].min() == pytest.approx(2.0)
    assert verts[:, 1].max() == pytest.approx(7.0)
    assert np.allclose(verts[:, 2], 0.7)


def test_rect_vertices_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError):
        rect_vertices(0.0, 0.0, -1.0, 2.0)
    with pytest.raises(ValueError):
        rect_vertices(0.0, 0.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# circle_fan_vertices (triangle soup, matches legacy renderer contract)
# ---------------------------------------------------------------------------


def test_circle_fan_vertex_count() -> None:
    segments = 16
    verts = circle_fan_vertices(0.0, 0.0, 1.0, segments=segments)
    # (segments - 1) triangles * 3 vertices each.
    assert verts.shape == (3 * (segments - 1), 3)


def test_circle_fan_within_radius() -> None:
    verts = circle_fan_vertices(2.0, -3.0, 1.5, segments=24, z=0.1)
    radial = np.hypot(verts[:, 0] - 2.0, verts[:, 1] + 3.0)
    assert radial.max() <= 1.5 + 1e-9
    assert np.allclose(verts[:, 2], 0.1)


def test_circle_fan_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        circle_fan_vertices(0.0, 0.0, 0.0)  # radius must be > 0
    with pytest.raises(ValueError):
        circle_fan_vertices(0.0, 0.0, 1.0, segments=2)  # need >= 3


# ---------------------------------------------------------------------------
# disc_mesh (indexed triangle fan)
# ---------------------------------------------------------------------------


def test_disc_mesh_shapes() -> None:
    segments = 32
    verts, faces = disc_mesh((0.0, 0.0), 1.0, segments=segments)
    assert verts.shape == (segments + 1, 3)  # centre + ring
    assert faces.shape == (segments, 3)
    assert faces.max() < verts.shape[0]
    assert faces.min() == 0


def test_disc_mesh_area_approximates_circle() -> None:
    segments = 256
    radius = 2.0
    verts, faces = disc_mesh((0.0, 0.0), radius, segments=segments)
    area = 0.0
    for a, b, c in faces:
        pa, pb, pc = verts[a], verts[b], verts[c]
        area += 0.5 * abs(
            (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pc[0] - pa[0]) * (pb[1] - pa[1])
        )
    assert area == pytest.approx(np.pi * radius**2, rel=1e-3)


def test_disc_mesh_centre_at_requested_position() -> None:
    verts, _ = disc_mesh((5.0, -1.0), 0.5, z=0.2)
    np.testing.assert_allclose(verts[0], (5.0, -1.0, 0.2))


def test_disc_mesh_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        disc_mesh((0.0, 0.0), -1.0)
    with pytest.raises(ValueError):
        disc_mesh((0.0, 0.0), 1.0, segments=2)


# ---------------------------------------------------------------------------
# grid_surface_mesh
# ---------------------------------------------------------------------------


def test_grid_surface_mesh_shapes() -> None:
    xs = np.linspace(0.0, 10.0, 5)
    ys = np.linspace(0.0, 4.0, 3)
    zz = np.zeros((ys.size, xs.size))
    verts, faces, colors = grid_surface_mesh(xs, ys, zz)
    assert verts.shape == (xs.size * ys.size, 3)
    assert faces.shape == (2 * (xs.size - 1) * (ys.size - 1), 3)
    assert colors.shape == (xs.size * ys.size, 4)
    assert faces.max() < verts.shape[0]


def test_grid_surface_mesh_preserves_elevation() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0])
    zz = np.array([[0.0, 0.5, 1.0], [0.2, 0.7, 1.2]])
    verts, _, _ = grid_surface_mesh(xs, ys, zz)
    # Vertices are row-major over (y, x); z column must equal flattened zz.
    np.testing.assert_allclose(verts[:, 2], zz.reshape(-1))


def test_grid_surface_mesh_accepts_custom_colors() -> None:
    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    zz = np.zeros((2, 2))
    custom = np.tile((0.2, 0.4, 0.6, 1.0), (4, 1))
    _, _, colors = grid_surface_mesh(xs, ys, zz, colors=custom)
    np.testing.assert_allclose(colors, custom)


def test_grid_surface_mesh_rejects_wrong_colors_shape() -> None:
    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    zz = np.zeros((2, 2))
    with pytest.raises(ValueError, match="colors shape"):
        grid_surface_mesh(xs, ys, zz, colors=np.zeros((3, 4)))


def test_grid_surface_mesh_rejects_shape_mismatch() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0])
    bad = np.zeros((3, 3))  # should be (2, 3)
    with pytest.raises(ValueError):
        grid_surface_mesh(xs, ys, bad)


def test_grid_surface_mesh_rejects_degenerate_grid() -> None:
    with pytest.raises(ValueError):
        grid_surface_mesh(np.array([0.0]), np.array([0.0, 1.0]), np.zeros((2, 1)))


# ---------------------------------------------------------------------------
# flagstick_lines
# ---------------------------------------------------------------------------


def test_flagstick_lines_endpoints() -> None:
    pts = flagstick_lines((3.0, 4.0), z=0.1, height=1.5)
    assert pts.shape == (2, 3)
    np.testing.assert_allclose(pts[0], (3.0, 4.0, 0.1))
    np.testing.assert_allclose(pts[1], (3.0, 4.0, 1.6))


def test_flagstick_rejects_non_positive_height() -> None:
    with pytest.raises(ValueError):
        flagstick_lines((0.0, 0.0), height=0.0)
