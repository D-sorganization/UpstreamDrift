"""Fail-first public surface layer seam: the compositor and audit share geometry."""

from types import SimpleNamespace
import numpy as np
import pytest
from src.shared.python.body_part_viz.overlay_options import ShapeOverlayOptions
from src.shared.python.body_part_viz.renderers import SurfaceLayer, composite_surface
from src.shared.python.motion_matching.historical_fit import CameraProjection
from src.shared.python.motion_matching.visual_skeleton import Shape
from src.shared.python.workspace.necromatcher_shape_overlay import (
    NativeShapeOverlay,
    _hint_surface,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "source",
    [
        [],
        np.zeros((0, 9, 3), np.uint8),
        np.zeros((9, 9), np.uint8),
        np.zeros((9, 9, 4), np.uint8),
        np.zeros((9, 9, 3), float),
    ],
)
def test_zero_opacity_still_rejects_malformed_source_before_fk(source):
    _, binding, calls = overlay_case()
    overlay = NativeShapeOverlay(ShapeOverlayOptions(0), (), {})
    with pytest.raises((TypeError, ValueError)):
        overlay.composite(binding, np.zeros(1), source)
    assert not calls


def overlay_case():
    surface = _hint_surface(
        Shape("body", (0.0, 0.0, 0.0), (0.5, 1.0, 0.25), "box", "box"), 0
    )
    calls = []

    def poses(mapping, pose):
        calls.append(mapping)
        return {
            "body": (
                np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                np.array([0.0, 0.0, 4.0]),
            )
        }

    camera = CameraProjection(
        np.array([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]),
        np.eye(3),
        np.zeros(3),
    )
    binding = SimpleNamespace(
        plant=SimpleNamespace(frame_poses=poses), review_inputs=lambda: (camera, {})
    )
    return NativeShapeOverlay(ShapeOverlayOptions(0.35), (surface,), {}), binding, calls


def test_public_layer_is_canonical_composite_geometry_and_sealed():
    overlay, binding, calls = overlay_case()
    layer = overlay.surface_layer(binding, np.zeros(1), (9, 9))
    assert isinstance(layer, SurfaceLayer) and len(calls) == 1
    assert layer.mask[4, 4] and layer.depth[4, 4] == pytest.approx(3.75)
    assert layer.geometry_ids[4, 4] == "hint:0:box"
    assert not layer.mask.flags.writeable and not layer.depth.flags.writeable
    source = np.full((9, 9, 3), 70, np.uint8)
    np.testing.assert_array_equal(
        overlay.composite(binding, np.zeros(1), source),
        composite_surface(source, layer, overlay.options),
    )


@pytest.mark.parametrize("size", [(True, 9), (9, 0), (9, 9.5), [9, 9], (9,)])
def test_invalid_public_layer_size_rejects_before_fk_or_camera(size):
    overlay, binding, calls = overlay_case()
    with pytest.raises((TypeError, ValueError)):
        overlay.surface_layer(binding, np.zeros(1), size)
    assert not calls


def test_zero_opacity_composite_remains_no_fk_copy():
    _, binding, calls = overlay_case()
    overlay = NativeShapeOverlay(ShapeOverlayOptions(0), (), {})
    source = np.full((9, 9, 3), 70, np.uint8)
    result = overlay.composite(binding, np.zeros(1), source)
    np.testing.assert_array_equal(result, source)
    assert result is not source and not calls
