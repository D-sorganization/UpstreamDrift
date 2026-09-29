"""Unit tests for :class:`StarMarker`."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.plot_style import MarkerShape, MarkerShapeRenderer, MarkerStyle
from src.shared.python.plot_style.shapes import StarMarker

# Default n_points = 5: 2*5 equator + 2 apex = 12 vertices, 4*5 = 20 triangles.
EXPECTED_VERTS = 12
EXPECTED_FACES = 20


def test_star_marker_default_counts() -> None:
    m = StarMarker()
    v, f = m.mesh(MarkerStyle(size_px=2.0))
    assert v.shape == (EXPECTED_VERTS, 3)
    assert f.shape == (EXPECTED_FACES, 3)


def test_star_marker_unit_radius_bbox() -> None:
    m = StarMarker()
    v, _ = m.mesh(MarkerStyle(size_px=2.0))
    extent = v.max(axis=0) - v.min(axis=0)
    # Z extent is exactly +/-1 thanks to the apexes; XY may be slightly less.
    assert np.isclose(extent[2], 2.0, atol=1e-9)
    radii = np.linalg.norm(v, axis=1)
    # Outer equator points and apexes all sit on the unit sphere.
    np.testing.assert_allclose(radii.max(), 1.0, atol=1e-9)


def test_star_marker_scale_linearity() -> None:
    m = StarMarker()
    v1, _ = m.mesh(MarkerStyle(size_px=4.0))
    v2, _ = m.mesh(MarkerStyle(size_px=8.0))
    np.testing.assert_allclose(v2, 2.0 * v1, atol=1e-12)


def test_star_marker_protocol_runtime_check() -> None:
    assert isinstance(StarMarker(), MarkerShapeRenderer)


def test_star_marker_shape_id() -> None:
    assert StarMarker.shape_id == MarkerShape.STAR.value


def test_n_points_min() -> None:
    m = StarMarker(n_points=3)
    v, f = m.mesh(MarkerStyle(size_px=2.0))
    assert v.shape == (2 * 3 + 2, 3)
    assert f.shape == (4 * 3, 3)


def test_n_points_too_small() -> None:
    with pytest.raises(ValueError):
        StarMarker(n_points=2)


def test_n_points_wrong_type() -> None:
    with pytest.raises(TypeError):
        StarMarker(n_points=5.0)  # type: ignore[arg-type]


def test_n_points_bool_rejected() -> None:
    with pytest.raises(TypeError):
        StarMarker(n_points=True)  # type: ignore[arg-type]


def test_inner_ratio_zero_rejected() -> None:
    with pytest.raises(ValueError):
        StarMarker(inner_ratio=0.0)


def test_inner_ratio_one_rejected() -> None:
    with pytest.raises(ValueError):
        StarMarker(inner_ratio=1.0)


def test_inner_ratio_wrong_type() -> None:
    with pytest.raises(TypeError):
        StarMarker(inner_ratio="x")  # type: ignore[arg-type]


def test_inner_ratio_bool_rejected() -> None:
    with pytest.raises(TypeError):
        StarMarker(inner_ratio=True)  # type: ignore[arg-type]


def test_inner_ratio_nan_rejected() -> None:
    with pytest.raises(ValueError):
        StarMarker(inner_ratio=float("nan"))


def test_star_marker_mesh_rejects_non_style() -> None:
    m = StarMarker()
    with pytest.raises(TypeError):
        m.mesh(42)  # type: ignore[arg-type]
