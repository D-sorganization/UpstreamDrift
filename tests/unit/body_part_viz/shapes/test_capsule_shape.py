"""Tests for CapsuleShape."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz import BodyPartShape
from src.shared.python.body_part_viz.shapes import CapsuleShape

from ._helpers import make_identity_fitted


def test_capsule_vertex_count() -> None:
    n_facets, n_lat = 16, 8
    cap = CapsuleShape(1.0, 0.2, n_facets=n_facets, n_lat=n_lat)
    # Two cylinder rings (rim is shared with hemispheres) + per-hemisphere
    # interior rings + apex == 2*n_facets + 2*((n_lat-1)*n_facets + 1).
    expected = 2 * n_facets + 2 * ((n_lat - 1) * n_facets + 1)
    assert cap.vertices_at_rest().shape == (expected, 3)


def test_capsule_protocol() -> None:
    assert isinstance(CapsuleShape(1.0, 0.2), BodyPartShape)


def test_capsule_extent_in_x() -> None:
    length, radius = 1.0, 0.3
    cap = CapsuleShape(length, radius, n_facets=12, n_lat=6)
    v = cap.vertices_at_rest()
    np.testing.assert_allclose(v[:, 0].min(), -radius, atol=1e-9)
    np.testing.assert_allclose(v[:, 0].max(), length + radius, atol=1e-9)


def test_capsule_endcap_radius_constant() -> None:
    cap = CapsuleShape(1.0, 0.4, n_facets=16, n_lat=8)
    v = cap.vertices_at_rest()
    # Vertices in the cylindrical body should sit on radius 0.4 in yz.
    body = v[(v[:, 0] >= 0.0) & (v[:, 0] <= 1.0)]
    yz = np.linalg.norm(body[:, 1:], axis=1)
    assert yz.max() <= 0.4 + 1e-9


def test_capsule_validation() -> None:
    with pytest.raises(ValueError):
        CapsuleShape(0.0, 0.1)
    with pytest.raises(ValueError):
        CapsuleShape(1.0, -0.1)
    with pytest.raises(ValueError):
        CapsuleShape(1.0, 0.1, n_facets=2)
    with pytest.raises(ValueError):
        CapsuleShape(1.0, 0.1, n_lat=1)
    with pytest.raises(TypeError):
        CapsuleShape(1.0, 0.1, n_facets=4.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CapsuleShape(1.0, 0.1, n_lat=4.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CapsuleShape(1.0, 0.1, shape_id="")


def test_capsule_props_and_faces() -> None:
    cap = CapsuleShape(1.0, 0.1, n_facets=12, n_lat=6)
    assert cap.n_facets == 12 and cap.n_lat == 6
    assert cap.faces().shape[1] == 3


def test_capsule_transform_identity() -> None:
    cap = CapsuleShape(1.0, 0.1)
    fitted = make_identity_fitted("c", n_frames=1)
    out = cap.transform(fitted)
    np.testing.assert_allclose(out[0], cap.vertices_at_rest(), atol=1e-12)
