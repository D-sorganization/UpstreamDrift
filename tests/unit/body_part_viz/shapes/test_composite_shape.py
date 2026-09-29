"""Tests for CompositeShape."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz import BodyPartShape
from src.shared.python.body_part_viz.shapes import (
    CompositeShape,
    CylinderShape,
    LineShape,
)

from ._helpers import make_identity_fitted


def _identity_4x4() -> np.ndarray:
    return np.eye(4)


def _translation_4x4(t: tuple[float, float, float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = t
    return m


def test_composite_protocol() -> None:
    comp = CompositeShape([(LineShape(1.0), _identity_4x4())])
    assert isinstance(comp, BodyPartShape)


def test_composite_two_cylinders_concat_vertices() -> None:
    c1 = CylinderShape(1.0, 0.1, n_facets=8)
    c2 = CylinderShape(1.0, 0.1, n_facets=8)
    comp = CompositeShape(
        [(c1, _identity_4x4()), (c2, _translation_4x4((1.0, 0.0, 0.0)))]
    )
    v = comp.vertices_at_rest()
    assert v.shape[0] == c1.vertices_at_rest().shape[0] * 2
    f = comp.faces()
    assert f.shape[0] == c1.faces().shape[0] * 2
    # Re-indexed faces: the second child's max index should reach
    # 2*V_child - 1.
    assert int(f.max()) == v.shape[0] - 1


def test_composite_local_translation_applied() -> None:
    line = LineShape(1.0)
    comp = CompositeShape(
        [
            (line, _identity_4x4()),
            (line, _translation_4x4((10.0, 0.0, 0.0))),
        ]
    )
    v = comp.vertices_at_rest()
    np.testing.assert_allclose(v[2], [10.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v[3], [11.0, 0.0, 0.0], atol=1e-12)


def test_composite_rest_dimensions_concat() -> None:
    line = LineShape(2.0)
    cyl = CylinderShape(1.0, 0.1)
    comp = CompositeShape([(line, _identity_4x4()), (cyl, _identity_4x4())])
    assert comp.rest_dimensions == (2.0, 1.0, 0.1)


def test_composite_validation() -> None:
    with pytest.raises(ValueError):
        CompositeShape([])
    with pytest.raises(TypeError):
        CompositeShape("oops")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CompositeShape([(LineShape(1.0),)])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        CompositeShape([("not a shape", _identity_4x4())])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        CompositeShape([(LineShape(1.0), [[1, 0], [0, 1]])])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        CompositeShape([(LineShape(1.0), np.eye(3))])
    with pytest.raises(ValueError):
        bad = np.eye(4)
        bad[0, 0] = np.nan
        CompositeShape([(LineShape(1.0), bad)])
    with pytest.raises(ValueError):
        CompositeShape([(LineShape(1.0), _identity_4x4())], shape_id="")


def test_composite_transform_identity() -> None:
    line = LineShape(1.0)
    comp = CompositeShape([(line, _identity_4x4())])
    fitted = make_identity_fitted("comp", n_frames=2)
    out = comp.transform(fitted)
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(out[0], comp.vertices_at_rest(), atol=1e-12)


def test_composite_children_property() -> None:
    line = LineShape(1.0)
    comp = CompositeShape([(line, _identity_4x4())])
    assert len(comp.children) == 1
    assert comp.children[0][0] is line
