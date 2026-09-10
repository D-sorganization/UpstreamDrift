"""Native meshes preserve world frames, visibility and viewport handle ownership."""

import numpy as np
import pytest

from src.motion_capture.coaching import (
    ReferenceGeometry,
    ReferencePlane,
    ReferencePoint,
)
from src.motion_capture.coaching.native_geometry import NativeGeometryRenderer

pytestmark = pytest.mark.unit


class Viewport:
    def __init__(self):
        self.meshes = {}
        self.serial = 0
        self.fail = False

    def add_mesh(self, vertices, faces, *, color, alpha):
        if self.fail:
            raise RuntimeError("renderer unavailable")
        self.serial += 1
        self.meshes[self.serial] = (vertices, faces, color, alpha)
        return self.serial

    def remove_mesh(self, handle):
        del self.meshes[handle]


def document():
    return ReferenceGeometry(
        scene_id="scene",
        planes=(
            ReferencePlane(
                origin_m=(0, 2, 0),
                along_m=(1, 2, 0),
                across_m=(0, 2, 1),
                opacity=0.4,
                last_s=1,
            ),
        ),
    )


def test_native_plane_uses_explicit_z_up_frame_and_replaces_handles():
    view = Viewport()
    renderer = NativeGeometryRenderer(view, scene_id="scene", frame="world_Zup")
    renderer.render(document(), 0)
    vertices, faces, _, alpha = next(iter(view.meshes.values()))
    np.testing.assert_allclose(vertices[:, 2], 2)
    assert faces.shape == (2, 3) and alpha == 0.4
    renderer.render(document(), 1)
    assert len(view.meshes) == 1
    renderer.render(document(), 2)
    assert not view.meshes
    renderer.clear()


def test_native_scene_rejection_and_failed_render_keep_existing_meshes():
    view = Viewport()
    renderer = NativeGeometryRenderer(view, scene_id="scene", frame="adr0041_y_up")
    renderer.render(document(), 0)
    before = set(view.meshes)
    with pytest.raises(ValueError, match="scene"):
        renderer.render(document().model_copy(update={"scene_id": "other"}), 0)
    assert set(view.meshes) == before
    view.fail = True
    with pytest.raises(RuntimeError):
        renderer.render(document(), 0)
    assert set(view.meshes) == before


def test_native_point_preserves_metric_position_colour_and_opacity():
    view = Viewport()
    renderer = NativeGeometryRenderer(view, scene_id="scene", frame="adr0041_y_up")
    point = ReferencePoint(position_m=(1, 2, 3), colour="#ff0000", opacity=0.6)
    renderer.render(ReferenceGeometry(scene_id="scene", points=(point,)), 0)
    vertices, faces, colour, alpha = next(iter(view.meshes.values()))
    np.testing.assert_allclose(vertices.mean(axis=0), (1, 2, 3))
    assert faces.shape == (8, 3)
    assert colour == (1, 0, 0) and alpha == 0.6
    renderer.render(
        ReferenceGeometry(scene_id="scene", points=(point.changed(visible=False),)), 0
    )
    assert not view.meshes


def test_partial_submission_failure_removes_new_meshes_only(monkeypatch):
    view = Viewport()
    renderer = NativeGeometryRenderer(view, scene_id="scene", frame="world_Zup")
    renderer.render(document(), 0)
    before = set(view.meshes)
    original = view.add_mesh

    def submit(vertices, faces, *, color, alpha):
        if view.serial == 2:
            raise RuntimeError("second mesh failed")
        return original(vertices, faces, color=color, alpha=alpha)

    monkeypatch.setattr(view, "add_mesh", submit)
    geometry = ReferenceGeometry(
        scene_id="scene",
        planes=document().planes,
        points=(ReferencePoint(position_m=(0, 0, 0)),),
    )
    with pytest.raises(RuntimeError, match="second mesh"):
        renderer.render(geometry, 0)
    assert set(view.meshes) == before
