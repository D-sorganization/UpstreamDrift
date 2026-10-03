"""Coordinate gather allocation and exact overlapping-surface laws."""

import numpy as np
import pytest
from src.shared.python.body_part_viz.renderers import SurfaceMesh, render_surface_layer
from src.shared.python.body_part_viz.renderers import projective_renderer
from src.shared.python.motion_matching.historical_fit import CameraProjection

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("size", [(32, 24), (320, 240)])
def test_raster_depth_uses_roi_views_without_coordinate_gathers(monkeypatch, size):
    gathers = []

    class TrackedDepth(np.ndarray):
        def __getitem__(self, key):
            if isinstance(key, tuple) and any(isinstance(k, np.ndarray) for k in key):
                gathers.append("read")
            return super().__getitem__(key)

        def __setitem__(self, key, value):
            if isinstance(key, tuple) and any(isinstance(k, np.ndarray) for k in key):
                gathers.append("write")
            return super().__setitem__(key, value)

    original = np.full

    def tracked(*args, **kwargs):
        value = original(*args, **kwargs)
        return (
            value.view(TrackedDepth)
            if value.ndim == 2 and value.dtype.kind == "f"
            else value
        )

    monkeypatch.setattr(projective_renderer.np, "full", tracked)
    mesh = SurfaceMesh(
        "many",
        np.array([[2.0, 2.0, 1.0], [6.0, 2.0, 1.0], [2.0, 6.0, 1.0]]),
        np.tile([[0, 1, 2]], (8, 1)),
        (7, 8, 9),
    )
    camera = CameraProjection(np.eye(3), np.eye(3), np.zeros(3))
    layer = render_surface_layer((mesh,), camera, size)
    assert layer.mask[3, 3]
    assert gathers == []


def test_roi_updates_preserve_exact_nearest_depth_color_and_sorted_ties():
    camera = CameraProjection(np.eye(3), np.eye(3), np.zeros(3))
    square = np.array(
        [[4.0, 3.0, 1.0], [8.0, 3.0, 1.0], [8.0, 7.0, 1.0], [4.0, 7.0, 1.0]]
    )
    near = square[[0, 1, 3]]
    far = SurfaceMesh("far", square * 2, np.array([[0, 1, 2], [0, 2, 3]]), (1, 2, 3))
    a = SurfaceMesh("a", near, np.array([[0, 1, 2]]), (4, 5, 6))
    z = SurfaceMesh("z", near, np.array([[0, 1, 2]]), (7, 8, 9))
    layer = render_surface_layer((z, far, a), camera, (16, 12))
    expected = np.zeros((12, 16), bool)
    expected[3:8, 4:9] = True
    ys, xs = np.indices(expected.shape)
    front = expected & ((xs - 4) + (ys - 3) <= 4)
    np.testing.assert_array_equal(layer.mask, expected)
    np.testing.assert_array_equal(layer.depth[front], np.ones(front.sum()))
    np.testing.assert_array_equal(
        layer.depth[expected & ~front], np.full((expected & ~front).sum(), 2.0)
    )
    np.testing.assert_array_equal(
        layer.pixels[front], np.tile([4, 5, 6], (front.sum(), 1))
    )
    np.testing.assert_array_equal(
        layer.pixels[expected & ~front],
        np.tile([1, 2, 3], ((expected & ~front).sum(), 1)),
    )
    assert (layer.geometry_ids[front] == "a").all()
    assert (layer.geometry_ids[expected & ~front] == "far").all()
    assert np.isposinf(layer.depth[~expected]).all()
    assert not layer.pixels[~expected].any()
    assert (layer.geometry_ids[~expected] == "").all()
