"""Unit tests for :class:`CrossMarker`."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.plot_style import MarkerShape, MarkerShapeRenderer, MarkerStyle
from src.shared.python.plot_style.shapes import CrossMarker

# Three orthogonal bars, each 8 vertices and 12 triangles.
EXPECTED_VERTS = 24
EXPECTED_FACES = 36


def test_default_counts() -> None:
    m = CrossMarker()
    v, f = m.mesh(MarkerStyle(size_px=2.0))
    assert v.shape == (EXPECTED_VERTS, 3)
    assert f.shape == (EXPECTED_FACES, 3)


def test_unit_radius_bbox() -> None:
    m = CrossMarker()
    v, _ = m.mesh(MarkerStyle(size_px=2.0))
    extent = v.max(axis=0) - v.min(axis=0)
    # Each bar reaches +/-1 along its long axis.
    np.testing.assert_allclose(extent, [2.0, 2.0, 2.0], atol=1e-9)


def test_scale_linearity() -> None:
    m = CrossMarker()
    v1, _ = m.mesh(MarkerStyle(size_px=2.0))
    v2, _ = m.mesh(MarkerStyle(size_px=4.0))
    np.testing.assert_allclose(v2, 2.0 * v1, atol=1e-12)


def test_protocol_runtime_check() -> None:
    assert isinstance(CrossMarker(), MarkerShapeRenderer)


def test_shape_id() -> None:
    assert CrossMarker.shape_id == MarkerShape.CROSS.value


def test_thickness_zero_rejected() -> None:
    with pytest.raises(ValueError):
        CrossMarker(thickness=0.0)


def test_thickness_too_large_rejected() -> None:
    with pytest.raises(ValueError):
        CrossMarker(thickness=2.0)


def test_thickness_non_numeric_rejected() -> None:
    with pytest.raises(TypeError):
        CrossMarker(thickness="thick")  # type: ignore[arg-type]


def test_thickness_bool_rejected() -> None:
    with pytest.raises(TypeError):
        CrossMarker(thickness=True)  # type: ignore[arg-type]


def test_thickness_nan_rejected() -> None:
    with pytest.raises(ValueError):
        CrossMarker(thickness=float("nan"))


def test_custom_thickness() -> None:
    m = CrossMarker(thickness=0.5)
    v, _ = m.mesh(MarkerStyle(size_px=2.0))
    # Long axis still reaches +/-1.
    assert np.isclose(v.max(), 1.0)
    assert np.isclose(v.min(), -1.0)


def test_mesh_rejects_non_style() -> None:
    m = CrossMarker()
    with pytest.raises(TypeError):
        m.mesh(object())  # type: ignore[arg-type]
