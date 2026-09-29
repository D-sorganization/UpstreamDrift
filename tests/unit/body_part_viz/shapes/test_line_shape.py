"""Tests for LineShape."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.body_part_viz import BodyPartShape
from src.shared.python.body_part_viz.shapes import LineShape

from ._helpers import make_identity_fitted


def test_line_vertex_face_counts() -> None:
    line = LineShape(2.0)
    verts = line.vertices_at_rest()
    assert verts.shape == (2, 3)
    assert line.faces().shape == (0, 3)
    np.testing.assert_allclose(verts, [[0, 0, 0], [2, 0, 0]])


def test_line_protocol_isinstance() -> None:
    assert isinstance(LineShape(1.0), BodyPartShape)


def test_line_transform_identity_returns_input() -> None:
    line = LineShape(1.0)
    fitted = make_identity_fitted("line", n_frames=1)
    out = line.transform(fitted)
    assert out.shape == (1, 2, 3)
    np.testing.assert_allclose(out[0], line.vertices_at_rest(), atol=1e-12)


def test_line_transform_rotation() -> None:
    line = LineShape(1.0)
    rot = np.array(
        [
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )
    fitted = make_identity_fitted("line", n_frames=1, rotation=rot)
    out = line.transform(fitted)
    np.testing.assert_allclose(out[0, 1], [0.0, 1.0, 0.0], atol=1e-12)


def test_line_transform_invalid_frame_is_nan() -> None:
    line = LineShape(1.0)
    fitted = make_identity_fitted("line", n_frames=2, valid=False)
    out = line.transform(fitted)
    assert out.shape == (2, 2, 3)
    assert np.all(np.isnan(out))


def test_line_validation() -> None:
    with pytest.raises(ValueError):
        LineShape(0.0)
    with pytest.raises(ValueError):
        LineShape(-1.0)
    with pytest.raises(TypeError):
        LineShape("nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LineShape(1.0, shape_id="")
