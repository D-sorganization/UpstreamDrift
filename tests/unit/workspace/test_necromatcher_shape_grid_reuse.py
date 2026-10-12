"""Per-render grid allocation and absolute-pixel registration laws."""

import numpy as np
import pytest
from src.shared.python.body_part_viz.renderers import (
    SurfaceMesh,
    render_surface_layer,
)
from src.shared.python.motion_matching.historical_fit import CameraProjection

pytestmark = pytest.mark.unit


def camera():
    return CameraProjection(np.eye(3), np.eye(3), np.zeros(3))


@pytest.mark.parametrize("size", [(1280, 720), (320, 240)])
def test_coordinate_grid_allocated_once_for_many_front_triangles(monkeypatch, size):
    mesh = SurfaceMesh(
        "many",
        np.array([[2.0, 2.0, 1.0], [6.0, 2.0, 1.0], [2.0, 6.0, 1.0]]),
        np.tile([[0, 1, 2]], (24, 1)),
        (7, 8, 9),
    )
    original = np.meshgrid
    calls = []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "meshgrid", counted)
    layer = render_surface_layer((mesh,), camera(), size)
    assert layer.mask[3, 3]
    assert len(calls) == 1


@pytest.mark.parametrize("offset", [(2, 3), (17, 11)])
def test_reused_grid_preserves_absolute_pixel_centers_depth_and_identity(offset):
    x, y = offset
    vertices = np.array(
        [[x, y, 1.0], [x + 4, y, 1.0], [x + 4, y + 4, 1.0], [x, y + 4, 1.0]]
    )
    mesh = SurfaceMesh("square", vertices, np.array([[0, 1, 2], [0, 2, 3]]), (7, 8, 9))
    layer = render_surface_layer((mesh,), camera(), (32, 24))
    expected = np.zeros((24, 32), bool)
    expected[y : y + 5, x : x + 5] = True
    np.testing.assert_array_equal(layer.mask, expected)
    np.testing.assert_array_equal(layer.depth[expected], np.ones(25))
    np.testing.assert_array_equal(
        layer.geometry_ids[expected], np.full(25, "square", object)
    )
    np.testing.assert_array_equal(layer.pixels[expected], np.tile([7, 8, 9], (25, 1)))
    assert np.isposinf(layer.depth[~expected]).all()
    assert not layer.pixels[~expected].any()
