"""Tests for CylinderShape."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz import BodyPartShape
from src.shared.python.body_part_viz.shapes import CylinderShape

from ._helpers import make_identity_fitted


def test_cylinder_default_counts() -> None:
    cyl = CylinderShape()
    assert cyl.vertices_at_rest().shape == (2 * (16 + 1), 3)
    assert cyl.faces().shape == (4 * 16, 3)
    assert cyl.rest_dimensions == (1.0, 0.05)


def test_cylinder_protocol() -> None:
    assert isinstance(CylinderShape(), BodyPartShape)


def test_cylinder_validation() -> None:
    with pytest.raises(ValueError):
        CylinderShape(n_facets=2)
    with pytest.raises(ValueError):
        CylinderShape(radius=-0.1)
    with pytest.raises(ValueError):
        CylinderShape(length=0.0)
    with pytest.raises(TypeError):
        CylinderShape(n_facets=4.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CylinderShape(shape_id="")


def test_cylinder_anisotropic_scale() -> None:
    cyl = CylinderShape(length=1.0, radius=1.0, n_facets=8)
    scale = np.array([[2.0, 3.0, 4.0]])
    fitted = make_identity_fitted("c", n_frames=1, scale=scale)
    out = cyl.transform(fitted)
    rest = cyl.vertices_at_rest()
    np.testing.assert_allclose(out[0, :, 0], rest[:, 0] * 2.0, atol=1e-12)
    np.testing.assert_allclose(out[0, :, 1], rest[:, 1] * 3.0, atol=1e-12)
    np.testing.assert_allclose(out[0, :, 2], rest[:, 2] * 4.0, atol=1e-12)


def test_cylinder_n_facets_property() -> None:
    cyl = CylinderShape(n_facets=20)
    assert cyl.n_facets == 20
