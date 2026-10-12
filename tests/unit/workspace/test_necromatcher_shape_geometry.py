"""Shared visual proxies retain authored dimensions and body-local placement."""

import numpy as np
import pytest
from src.shared.python.body_part_viz.shapes import BoxShape
from src.shared.python.workspace.necromatcher_shape_overlay import (
    shape_overlay_provenance,
)
from src.shared.python.body_part_viz.overlay_options import ShapeOverlayOptions

pytestmark = pytest.mark.unit


def test_actual_native_provider_shape_transform_matches_marker_fk_at_rotated_pose(
    native_fit_case, monkeypatch
):
    from src.shared.python.workspace.necromatcher_native import load_native_fit_binding
    from src.shared.python.workspace import necromatcher_shape_overlay as module

    library, source, _ = native_fit_case
    library.add_fit("native-fit", "practice", source)
    binding = load_native_fit_binding(library, "native-fit")
    overlay = module.NativeShapeOverlay.prepare(binding, ShapeOverlayOptions(1.0))
    pose = np.asarray(binding.fit["q"][0], float)
    pose[:6] = [0.13, -0.07, 0.25, 0.21, -0.17, 0.31]
    captured = []
    original_render = module.render_surface_layer

    def render(meshes, camera, size):
        captured.extend(meshes)
        return original_render(meshes, camera, size)

    monkeypatch.setattr(module, "render_surface_layer", render)
    layer = overlay.surface_layer(binding, pose, (64, 64))
    assert layer.mask.any() and not layer.depth.flags.writeable
    local = overlay.surfaces[0]
    world = captured[0]
    rotation, _ = binding.plant.frame_poses(
        {"body": (local.body, (0.0, 0.0, 0.0))}, pose
    )[local.body]
    assert not np.allclose(rotation, np.eye(3))
    vertices = local.shape.vertices_at_rest()[[0, 3, 7]]
    attachments = {
        f"vertex{i}": (local.body, tuple(point)) for i, point in enumerate(vertices)
    }
    expected = binding.plant.marker_positions(pose, attachments)
    np.testing.assert_allclose(world.vertices[[0, 3, 7]], expected, atol=1e-12)


def test_shape_world_rotation_and_saved_projection_use_one_batched_public_fk():
    from types import SimpleNamespace
    from src.shared.python.motion_matching.historical_fit import CameraProjection
    from src.shared.python.motion_matching.visual_skeleton import Shape
    from src.shared.python.workspace.necromatcher_shape_overlay import (
        NativeShapeOverlay,
        _hint_surface,
    )

    surface = _hint_surface(
        Shape("body", (0.0, 0.0, 0.0), (0.5, 1.0, 0.25), "box", "box"), 0
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    calls = []

    def poses(mapping, pose):
        calls.append(mapping)
        return {"body": (rotation, np.array([0.0, 0.0, 4.0]))}

    camera = CameraProjection(
        np.array([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]),
        np.eye(3),
        np.zeros(3),
    )
    binding = SimpleNamespace(
        plant=SimpleNamespace(frame_poses=poses), review_inputs=lambda: (camera, {})
    )
    source = np.full((9, 9, 3), 70, np.uint8)
    result = NativeShapeOverlay(ShapeOverlayOptions(1), (surface,), {}).composite(
        binding, np.zeros(1), source
    )
    # Rotated native box: near face z=3.75, x extents +-1 and y extents +-.5.
    expected = np.zeros((9, 9), bool)
    expected[3:6, 2:7] = True
    np.testing.assert_array_equal(np.any(result != source, axis=2), expected)
    assert len(calls) == 1 and set(calls[0]) == {"body"}


def test_provenance_hash_domains_are_independent_and_metadata_owns_its_hints(
    native_fit_case,
):
    import hashlib
    import json
    from dataclasses import asdict
    from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton

    _, _, payload = native_fit_case
    definition = payload["provenance"]["native_definition"]
    record = shape_overlay_provenance(
        definition, payload["model_hash"], ShapeOverlayOptions(0.5)
    )
    expected = hashlib.sha256(
        json.dumps(definition, allow_nan=False).encode()
    ).hexdigest()
    visual = json.dumps(
        asdict(derive_visual_skeleton(definition)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert record["definition_sha256"] == "sha256:" + expected
    assert record["native_xml_sha256"] == payload["model_hash"]
    assert (
        record["visual_description_sha256"]
        == "sha256:" + hashlib.sha256(visual).hexdigest()
    )
    assert record["definition_sha256"] != record["native_xml_sha256"]
    assert record["physical_geometry_qualified"] is False
    if definition.get("visual_hints"):
        record["authored_visual_hints"]["changed"] = True
        assert "changed" not in definition["visual_hints"]


def test_capsule_keeps_authored_radius_and_body_local_endpoints():
    from src.shared.python.motion_matching.visual_skeleton import Capsule
    from src.shared.python.workspace.necromatcher_shape_overlay import _capsule_surface

    surface = _capsule_surface(
        Capsule("arm", (0.0, 1.0, 0.0), (0.0, 2.0, 0.0), 0.03), 0
    )
    vertices = surface.shape.vertices_at_rest()
    assert surface.body == "arm"
    np.testing.assert_allclose(vertices.min(axis=0), [-0.03, 0.97, -0.03], atol=1e-12)
    np.testing.assert_allclose(vertices.max(axis=0), [0.03, 2.03, 0.03], atol=1e-12)


def test_saved_box_hint_keeps_body_center_and_authored_half_sizes():
    from src.shared.python.motion_matching.visual_skeleton import Shape
    from src.shared.python.workspace.necromatcher_shape_overlay import _hint_surface

    surface = _hint_surface(
        Shape("torso", (1.0, 2.0, 3.0), (0.1, 0.2, 0.3), "box", "saved"), 0
    )
    np.testing.assert_allclose(
        surface.shape.vertices_at_rest().min(axis=0), [0.9, 1.8, 2.7]
    )
    np.testing.assert_allclose(
        surface.shape.vertices_at_rest().max(axis=0), [1.1, 2.2, 3.3]
    )


def test_box_extents_use_native_half_sizes_and_faces_are_triangles():
    box = BoxShape((0.1, 0.2, 0.3))
    np.testing.assert_allclose(box.vertices_at_rest().min(axis=0), [-0.1, -0.2, -0.3])
    np.testing.assert_allclose(box.vertices_at_rest().max(axis=0), [0.1, 0.2, 0.3])
    assert box.faces().shape == (12, 3)


@pytest.mark.parametrize(
    "half", [(0.0, 1.0, 1.0), (1.0, float("nan"), 1.0), (1.0, 1.0), (True, 1.0, 1.0)]
)
def test_box_rejects_malformed_dimensions(half):
    with pytest.raises((TypeError, ValueError)):
        BoxShape(half)
