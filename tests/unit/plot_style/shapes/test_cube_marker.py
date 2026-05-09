"""Unit tests for :class:`CubeMarker`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.plot_style import MarkerShape, MarkerShapeRenderer, MarkerStyle
from src.shared.python.plot_style.shapes import CubeMarker

# A cube has 8 vertices and 12 outward-facing triangles.
EXPECTED_VERTS = 8
EXPECTED_FACES = 12


def test_default_counts() -> None:
    m = CubeMarker()
    v, f = m.mesh(MarkerStyle(size_px=2.0))
    assert v.shape == (EXPECTED_VERTS, 3)
    assert f.shape == (EXPECTED_FACES, 3)
    assert v.dtype == np.float64
    assert f.dtype == np.int64


def test_unit_radius_bounding_sphere() -> None:
    m = CubeMarker()
    v, _ = m.mesh(MarkerStyle(size_px=2.0))  # radius==1
    radii = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(radii.max(), 1.0, atol=1e-9)
    # Half-edge of inscribed cube == 1/sqrt(3).
    extent_half = (v.max(axis=0) - v.min(axis=0)) / 2.0
    np.testing.assert_allclose(extent_half, [1.0 / math.sqrt(3.0)] * 3, atol=1e-9)


def test_scale_linearity() -> None:
    m = CubeMarker()
    v1, _ = m.mesh(MarkerStyle(size_px=2.0))
    v2, _ = m.mesh(MarkerStyle(size_px=4.0))
    np.testing.assert_allclose(v2, 2.0 * v1, atol=1e-12)


def test_protocol_runtime_check() -> None:
    assert isinstance(CubeMarker(), MarkerShapeRenderer)


def test_shape_id() -> None:
    assert CubeMarker.shape_id == MarkerShape.CUBE.value


def test_face_indices_in_range() -> None:
    m = CubeMarker()
    _, f = m.mesh(MarkerStyle(size_px=2.0))
    assert int(f.min()) == 0
    assert int(f.max()) == EXPECTED_VERTS - 1


def test_mesh_rejects_non_style() -> None:
    m = CubeMarker()
    with pytest.raises(TypeError):
        m.mesh(123)  # type: ignore[arg-type]
