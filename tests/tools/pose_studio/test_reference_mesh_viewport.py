"""Pose Studio supports the shared reference mesh protocol without losing its pose."""

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PyQt6")

from src.motion_capture.coaching.native_geometry import NativeGeometryRenderer
from src.tools.pose_studio.widgets.view_3d import View3D
from tests.motion_capture.test_native_reference_geometry import document
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_pose_view_preserves_skeleton_while_replacing_reference_mesh():
    _app()
    view = View3D()
    before = len(view._ax.collections)
    renderer = NativeGeometryRenderer(view, scene_id="scene", frame="world_Zup")
    renderer.render(document(), 0)
    assert len(view._ax.collections) == before + 1
    renderer.render(document(), 1)
    assert len(view._ax.collections) == before + 1
    renderer.clear()
    assert len(view._ax.collections) == before
    view.close()
