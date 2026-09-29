"""Tests for EllipsoidShape."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz import BodyPartShape
from src.shared.python.body_part_viz.shapes import EllipsoidShape

from ._helpers import make_identity_fitted


def test_ellipsoid_vertex_count() -> None:
    e = EllipsoidShape(1.0, 1.0, 1.0)
    assert e.vertices_at_rest().shape == (16 * 9, 3)


def test_ellipsoid_protocol() -> None:
    assert isinstance(EllipsoidShape(1.0, 1.0, 1.0), BodyPartShape)


def test_ellipsoid_axis_relation() -> None:
    a, b, c = 2.0, 1.5, 0.5
    e = EllipsoidShape(a, b, c, n_lon=20, n_lat=10)
    v = e.vertices_at_rest()
    eq = (v[:, 0] / a) ** 2 + (v[:, 1] / b) ** 2 + (v[:, 2] / c) ** 2
    np.testing.assert_allclose(eq, 1.0, atol=1e-9)


def test_ellipsoid_validation() -> None:
    with pytest.raises(ValueError):
        EllipsoidShape(0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        EllipsoidShape(1.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        EllipsoidShape(1.0, 1.0, 1.0, n_lon=2)
    with pytest.raises(ValueError):
        EllipsoidShape(1.0, 1.0, 1.0, n_lat=1)
    with pytest.raises(TypeError):
        EllipsoidShape(1.0, 1.0, 1.0, n_lon=4.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        EllipsoidShape(1.0, 1.0, 1.0, n_lat=4.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        EllipsoidShape(1.0, 1.0, 1.0, shape_id="")


def test_ellipsoid_props_and_faces() -> None:
    e = EllipsoidShape(1.0, 2.0, 3.0, n_lon=12, n_lat=6)
    assert e.n_lon == 12 and e.n_lat == 6
    assert e.faces().shape[1] == 3


def test_ellipsoid_transform_identity() -> None:
    e = EllipsoidShape(1.0, 1.0, 1.0)
    fitted = make_identity_fitted("e", n_frames=1)
    out = e.transform(fitted)
    np.testing.assert_allclose(out[0], e.vertices_at_rest(), atol=1e-12)
