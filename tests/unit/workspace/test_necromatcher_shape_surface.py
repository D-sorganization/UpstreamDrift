"""Fail-first geometry/compositing laws for the shared original-camera surface."""

import numpy as np
import pytest
from src.shared.python.body_part_viz.overlay_options import ShapeOverlayOptions
from src.shared.python.body_part_viz.renderers.projective_renderer import (
    SurfaceMesh,
    render_surface_layer,
    composite_surface,
)
from src.shared.python.motion_matching.historical_fit import CameraProjection
from src.shared.python.body_part_viz.renderers.projective_renderer import SurfaceLayer

pytestmark = pytest.mark.unit


def test_front_batch_prunes_off_raster_and_degenerate_faces_without_coverage_change(
    monkeypatch,
):
    from src.shared.python.body_part_viz.renderers import projective_renderer as module

    vertices = np.array(
        [
            [-0.5, -0.5, 1.0],
            [0.5, -0.5, 1.0],
            [0.0, 0.5, 1.0],
            [20.0, 20.0, 1.0],
            [21.0, 20.0, 1.0],
            [20.0, 21.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],
        ]
    )
    mesh = SurfaceMesh(
        "one", vertices, np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), (1, 2, 3)
    )
    reference = render_surface_layer(
        (triangle(1.0, (1, 2, 3), "one"),), camera(), (9, 9)
    )
    calls = []
    original = module._raster_triangle

    def raster(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(module, "_raster_triangle", raster)
    layer = render_surface_layer((mesh,), camera(), (9, 9))
    assert len(calls) == 1
    np.testing.assert_array_equal(layer.mask, reference.mask)
    np.testing.assert_array_equal(layer.pixels, reference.pixels)
    np.testing.assert_array_equal(layer.depth, reference.depth)


def test_front_mesh_batches_canonical_camera_calls_independent_of_triangle_count():
    class CountedCamera:
        def __init__(self):
            self.depth_calls = self.projection_calls = 0
            self.canonical = camera()

        def camera_points(self, points):
            self.depth_calls += 1
            return self.canonical.camera_points(points)

        def project(self, points):
            self.projection_calls += 1
            return self.canonical.project(points)

    mesh = SurfaceMesh(
        "square",
        np.array(
            [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [-0.5, 0.5, 1.0]]
        ),
        np.array([[0, 1, 2], [0, 2, 3]]),
        (1, 2, 3),
    )
    counted = CountedCamera()
    layer = render_surface_layer((mesh,), counted, (9, 9))
    assert layer.mask[4, 4]
    assert counted.depth_calls == 1 and counted.projection_calls == 1


@pytest.mark.parametrize(
    "vertices", [np.ones((3, 3), bool), np.full((3, 3), "1"), np.full((3, 3), np.nan)]
)
def test_surface_vertices_reject_coerced_or_nonfinite_geometry(vertices):
    with pytest.raises((TypeError, ValueError)):
        SurfaceMesh("invalid", vertices, np.array([[0, 1, 2]]), (1, 2, 3))


def test_compositor_rejects_nonarray_source_explicitly():
    layer = render_surface_layer((), camera(), (9, 9))
    with pytest.raises(TypeError, match="array"):
        composite_surface([], layer, ShapeOverlayOptions(0.5))


def test_subpixel_rectangle_has_analytical_integer_center_silhouette():
    # Independent camera law x'=8*x+4: square bounds are (.25, 3.25).
    world = np.array(
        [
            [-3.75, -3.75, 8.0],
            [-0.75, -3.75, 8.0],
            [-0.75, -0.75, 8.0],
            [-3.75, -0.75, 8.0],
        ]
    )
    mesh = SurfaceMesh("square", world, np.array([[0, 1, 2], [0, 2, 3]]), (1, 2, 3))
    layer = render_surface_layer((mesh,), camera(), (9, 9))
    expected = np.zeros((9, 9), bool)
    expected[1:4, 1:4] = True
    np.testing.assert_array_equal(layer.mask, expected)
    np.testing.assert_allclose(layer.depth[layer.mask], 8.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pixels", np.zeros((2, 2, 3), float)),
        ("mask", np.zeros((2, 2), np.uint8)),
        ("depth", np.zeros((3, 2), float)),
        ("depth", np.full((2, 2), -np.inf)),
        ("geometry_ids", np.full((2, 2), "uncovered", object)),
    ],
)
def test_layer_rejects_malformed_dtype_shape_or_uncovered_state(field, value):
    fields = {
        "pixels": np.zeros((2, 2, 3), np.uint8),
        "mask": np.zeros((2, 2), bool),
        "depth": np.full((2, 2), np.inf),
        "geometry_ids": np.full((2, 2), "", object),
    }
    fields[field] = value
    with pytest.raises(ValueError):
        SurfaceLayer(**fields)


def test_rendered_layer_is_immutable_and_cannot_be_tampered_before_compositing():
    layer = render_surface_layer((triangle(1.0, (1, 2, 3), "one"),), camera(), (9, 9))
    for array in (layer.pixels, layer.mask, layer.depth, layer.geometry_ids):
        assert not array.flags.writeable


def test_layer_rejects_covered_pixel_without_positive_finite_depth():
    with pytest.raises(ValueError, match="depth"):
        SurfaceLayer(
            np.zeros((2, 2, 3), np.uint8),
            np.ones((2, 2), bool),
            np.full((2, 2), np.inf),
            np.full((2, 2), "one", object),
        )


def test_triangle_crossing_camera_plane_clips_without_projecting_behind_camera():
    mesh = SurfaceMesh(
        "crossing",
        np.array([[-0.1, -0.1, -0.2], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]]),
        np.array([[0, 1, 2]]),
        (1, 2, 3),
    )
    layer = render_surface_layer((mesh,), camera(), (9, 9))
    assert layer.mask.any()
    assert np.isfinite(layer.depth[layer.mask]).all()
    assert (layer.depth[layer.mask] >= 1e-6).all()


@pytest.mark.parametrize("value", [True, "0.5", -0.1, 1.1, float("nan"), float("inf")])
def test_opacity_rejects_malformed_values(value):
    with pytest.raises((TypeError, ValueError)):
        ShapeOverlayOptions(value)


def camera():
    return CameraProjection(
        np.array([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]),
        np.eye(3),
        np.zeros(3),
    )


def triangle(depth, color, identity):
    # Same screen footprint at each depth. Existing camera projection is authority.
    vertices = np.array([[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]]) * depth
    return SurfaceMesh(identity, vertices, np.array([[0, 1, 2]]), color)


def test_nearer_surface_wins_independent_of_submission_order():
    near = triangle(1.0, (200, 20, 10), "near")
    far = triangle(2.0, (10, 20, 200), "far")
    a = render_surface_layer((near, far), camera(), (9, 9))
    b = render_surface_layer((far, near), camera(), (9, 9))
    np.testing.assert_array_equal(a.pixels, b.pixels)
    np.testing.assert_array_equal(a.mask, b.mask)
    np.testing.assert_allclose(a.depth, b.depth)
    assert a.mask[4, 4] and a.geometry_ids[4, 4] == "near"
    assert a.depth[4, 4] == pytest.approx(1.0)
    np.testing.assert_array_equal(a.pixels[4, 4], [200, 20, 10])


def test_opacity_zero_is_exact_source_identity_and_does_not_mutate():
    source = np.arange(9 * 9 * 3, dtype=np.uint8).reshape(9, 9, 3)
    before = source.copy()
    layer = render_surface_layer((triangle(1.0, (200, 20, 10), "a"),), camera(), (9, 9))
    np.testing.assert_array_equal(
        composite_surface(source, layer, ShapeOverlayOptions(0.0)), source
    )
    np.testing.assert_array_equal(source, before)


def test_opacity_one_changes_only_surface_coverage():
    source = np.full((9, 9, 3), 70, dtype=np.uint8)
    layer = render_surface_layer((triangle(1.0, (200, 20, 10), "a"),), camera(), (9, 9))
    result = composite_surface(source, layer, ShapeOverlayOptions(1.0))
    np.testing.assert_array_equal(result[~layer.mask], source[~layer.mask])
    np.testing.assert_array_equal(result[layer.mask], layer.pixels[layer.mask])


def test_half_opacity_rounding_is_deterministic():
    source = np.full((9, 9, 3), 70, dtype=np.uint8)
    layer = render_surface_layer((triangle(1.0, (200, 20, 10), "a"),), camera(), (9, 9))
    result = composite_surface(source, layer, ShapeOverlayOptions(0.5))
    np.testing.assert_array_equal(result[4, 4], [135, 45, 40])


def test_saved_camera_registered_under_unequal_focal_lengths_and_offset():
    c = CameraProjection(
        np.array([[12.0, 0.0, 5.0], [0.0, 6.0, 3.0], [0.0, 0.0, 1.0]]),
        np.eye(3),
        np.array([0.1, 0.2, 0.0]),
    )
    mesh = triangle(1.0, (200, 20, 10), "registered")
    pixels = c.project(mesh.vertices)
    np.testing.assert_allclose(pixels, [[0.2, 1.2], [12.2, 1.2], [6.2, 7.2]])
    layer = render_surface_layer((mesh,), c, (14, 10))
    assert layer.mask[4, 6] and not layer.mask[0, 6]


def test_surface_geometry_copies_inputs_and_rejects_invalid_faces():
    vertices = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    mesh = SurfaceMesh("one", vertices, np.array([[0, 1, 2]]), (1, 2, 3))
    vertices[:] = 99
    assert mesh.vertices[0, 2] == 1 and not mesh.vertices.flags.writeable
    with pytest.raises(ValueError):
        SurfaceMesh("bad", mesh.vertices, np.array([[0, 1, 3]]), (1, 2, 3))


@pytest.mark.parametrize("value", [False, "2", -1, 0, 1.5])
def test_raster_dimensions_are_strict_positive_integers(value):
    with pytest.raises((TypeError, ValueError)):
        render_surface_layer((triangle(1.0, (1, 2, 3), "a"),), camera(), (value, 9))


def test_equal_depth_tie_has_deterministic_geometry_identity():
    a = triangle(1.0, (200, 20, 10), "a")
    b = triangle(1.0, (10, 20, 200), "b")
    first = render_surface_layer((a, b), camera(), (9, 9))
    second = render_surface_layer((b, a), camera(), (9, 9))
    np.testing.assert_array_equal(first.pixels, second.pixels)
    assert first.geometry_ids[4, 4] == "a"


def test_duplicate_geometry_identity_is_rejected():
    with pytest.raises(ValueError):
        render_surface_layer(
            (triangle(1.0, (1, 2, 3), "same"), triangle(2.0, (3, 2, 1), "same")),
            camera(),
            (9, 9),
        )


def test_perspective_depth_matches_known_slanted_plane():
    # Camera center ray intersects the plane z=1+.5*x at z=1.
    mesh = SurfaceMesh(
        "slanted",
        np.array([[-0.5, -0.5, 0.75], [0.5, -0.5, 1.25], [0.0, 0.5, 1.0]]),
        np.array([[0, 1, 2]]),
        (1, 2, 3),
    )
    layer = render_surface_layer((mesh,), camera(), (9, 9))
    assert layer.depth[4, 4] == pytest.approx(1.0)


def test_wrong_source_size_is_rejected():
    layer = render_surface_layer((triangle(1.0, (1, 2, 3), "one"),), camera(), (9, 9))
    with pytest.raises(ValueError):
        composite_surface(
            np.zeros((8, 9, 3), dtype=np.uint8), layer, ShapeOverlayOptions(0.5)
        )
