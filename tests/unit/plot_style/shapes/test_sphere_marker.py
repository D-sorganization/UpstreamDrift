"""Unit tests for :class:`SphereMarker`."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.plot_style import MarkerShape, MarkerShapeRenderer, MarkerStyle
from src.shared.python.plot_style.shapes import SphereMarker

# Reference counts at default (n_lon=16, n_lat=8).
EXPECTED_VERTS = 16 * (8 + 1)  # = 144
EXPECTED_FACES = 2 * 16 * 8  # = 256


def test_default_counts() -> None:
    m = SphereMarker()
    v, f = m.mesh(MarkerStyle(size_px=2.0))
    assert v.shape == (EXPECTED_VERTS, 3)
    assert f.shape == (EXPECTED_FACES, 3)
    assert v.dtype == np.float64
    assert f.dtype == np.int64


def test_unit_radius_bbox() -> None:
    m = SphereMarker()
    v, _ = m.mesh(MarkerStyle(size_px=2.0))  # radius == 1
    extent = v.max(axis=0) - v.min(axis=0)
    np.testing.assert_allclose(extent, [2.0, 2.0, 2.0], atol=1e-9)
    radii = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(radii.max(), 1.0, atol=1e-9)


def test_scale_linearity() -> None:
    m = SphereMarker()
    v1, _ = m.mesh(MarkerStyle(size_px=2.0))
    v2, _ = m.mesh(MarkerStyle(size_px=4.0))
    np.testing.assert_allclose(v2, 2.0 * v1, atol=1e-12)


def test_protocol_runtime_check() -> None:
    assert isinstance(SphereMarker(), MarkerShapeRenderer)


def test_shape_id() -> None:
    assert SphereMarker.shape_id == MarkerShape.SPHERE.value


def test_invalid_n_lon_type() -> None:
    with pytest.raises(TypeError):
        SphereMarker(n_lon=16.0)  # type: ignore[arg-type]


def test_invalid_n_lat_type() -> None:
    with pytest.raises(TypeError):
        SphereMarker(n_lat=True)  # type: ignore[arg-type]


def test_invalid_n_lon_value() -> None:
    with pytest.raises(ValueError):
        SphereMarker(n_lon=2)


def test_invalid_n_lat_value() -> None:
    with pytest.raises(ValueError):
        SphereMarker(n_lat=1)


def test_mesh_rejects_non_style() -> None:
    m = SphereMarker()
    with pytest.raises(TypeError):
        m.mesh("not a style")  # type: ignore[arg-type]


def test_size_px_negative_rejected_by_style() -> None:
    # MarkerStyle itself enforces size_px > 0; verify the contract is intact.
    with pytest.raises(ValueError):
        MarkerStyle(size_px=-1.0)


def test_returned_arrays_are_independent() -> None:
    m = SphereMarker()
    _, f1 = m.mesh(MarkerStyle(size_px=2.0))
    _, f2 = m.mesh(MarkerStyle(size_px=2.0))
    f1[0, 0] = 999
    assert f2[0, 0] != 999
