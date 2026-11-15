"""The fast inside test agrees with the exact winding number (epic #15150)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends import qa
from src.shared.python.model_appearance.hand_backends.subdivision import (
    loop_subdivision,
)

pytestmark = pytest.mark.unit


def _blob() -> tuple[np.ndarray, np.ndarray]:
    """A closed, rounded, off-axis mesh of 512 triangles."""
    vertices = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int64,
    )
    for _ in range(3):
        operator, faces = loop_subdivision(faces, len(vertices))
        vertices = operator @ vertices
    return vertices * [0.03, 0.02, 0.05] + [0.01, -0.02, 0.03], faces


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_ray_test_matches_the_winding_number(axis: int) -> None:
    vertices, faces = _blob()
    rng = np.random.default_rng(15150)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    points = rng.uniform(low, high, size=(4000, 3))
    exact = qa._winding_numbers(points, vertices, faces) > 0.5
    assert 0.2 < exact.mean() < 0.8
    fast = qa._ray_says_inside(points, vertices[faces], axis)
    np.testing.assert_array_equal(fast, exact)


def test_depth_inside_counts_only_points_inside_the_mesh() -> None:
    vertices, faces = _blob()
    centre = vertices.mean(axis=0, keepdims=True)
    outside = vertices.max(axis=0, keepdims=True) + 0.01
    assert qa._depth_inside_m(outside, vertices, faces) == 0.0
    assert qa._depth_inside_m(np.empty((0, 3)), vertices, faces) == 0.0
    depth = qa._depth_inside_m(np.vstack([centre, outside]), vertices, faces)
    assert 0.005 < depth < 0.03
