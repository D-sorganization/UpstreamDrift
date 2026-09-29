"""Unit tests for :class:`DiamondMarker`."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.plot_style import MarkerShape, MarkerShapeRenderer, MarkerStyle
from src.shared.python.plot_style.shapes import DiamondMarker

# Octahedron: 6 vertices on the unit axes, 8 triangular faces.
EXPECTED_VERTS = 6
EXPECTED_FACES = 8


def test_diamond_marker_default_counts() -> None:
    m = DiamondMarker()
    v, f = m.mesh(MarkerStyle(size_px=2.0))
    assert v.shape == (EXPECTED_VERTS, 3)
    assert f.shape == (EXPECTED_FACES, 3)


def test_diamond_marker_unit_radius_bbox() -> None:
    m = DiamondMarker()
    v, _ = m.mesh(MarkerStyle(size_px=2.0))
    extent = v.max(axis=0) - v.min(axis=0)
    np.testing.assert_allclose(extent, [2.0, 2.0, 2.0], atol=1e-9)
    radii = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(radii.max(), 1.0, atol=1e-9)
    np.testing.assert_allclose(radii.min(), 1.0, atol=1e-9)


def test_diamond_marker_scale_linearity() -> None:
    m = DiamondMarker()
    v1, _ = m.mesh(MarkerStyle(size_px=3.0))
    v2, _ = m.mesh(MarkerStyle(size_px=6.0))
    np.testing.assert_allclose(v2, 2.0 * v1, atol=1e-12)


def test_diamond_marker_protocol_runtime_check() -> None:
    assert isinstance(DiamondMarker(), MarkerShapeRenderer)


def test_diamond_marker_shape_id() -> None:
    assert DiamondMarker.shape_id == MarkerShape.DIAMOND.value


def test_diamond_marker_mesh_rejects_non_style() -> None:
    m = DiamondMarker()
    with pytest.raises(TypeError):
        m.mesh(None)  # type: ignore[arg-type]


def test_face_indices_valid() -> None:
    m = DiamondMarker()
    _, f = m.mesh(MarkerStyle(size_px=2.0))
    assert int(f.min()) == 0
    assert int(f.max()) == EXPECTED_VERTS - 1
