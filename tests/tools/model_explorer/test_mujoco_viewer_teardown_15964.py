"""MuJoCo viewer timer must not render into a destroyed Qt viewport."""

from unittest.mock import Mock

import numpy as np
import pytest
from PyQt6 import sip

from src.tools.model_explorer.mujoco_viewer import MuJoCoViewerWidget

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_destroyed_viewport_stops_timer_and_ignores_queued_render(qtbot, monkeypatch):
    """Delete the actual QLabel before the viewer, then deliver a stale callback."""
    monkeypatch.setattr(MuJoCoViewerWidget, "_setup_renderer", lambda self: None)
    viewer = MuJoCoViewerWidget()
    qtbot.addWidget(viewer)
    renderer = Mock()
    renderer.render.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
    viewer._renderer = renderer
    assert viewer._render_timer.isActive()
    sip.delete(viewer._viewport)
    assert not viewer._render_timer.isActive()
    viewer._update_render()
    renderer.render.assert_not_called()


def test_live_viewport_still_receives_frames(qtbot, monkeypatch):
    """Keep normal rendering active while the viewport remains alive."""
    monkeypatch.setattr(MuJoCoViewerWidget, "_setup_renderer", lambda self: None)
    viewer = MuJoCoViewerWidget()
    qtbot.addWidget(viewer)
    renderer = Mock()
    renderer.render.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
    viewer._renderer = renderer
    viewer._update_render()
    renderer.render.assert_called_once()
    assert not viewer._viewport.pixmap().isNull()


def test_live_renderer_errors_are_not_suppressed(qtbot, monkeypatch):
    """Preserve normal error reporting when an active renderer fails."""
    monkeypatch.setattr(MuJoCoViewerWidget, "_setup_renderer", lambda self: None)
    viewer = MuJoCoViewerWidget()
    qtbot.addWidget(viewer)
    renderer = Mock()
    renderer.render.side_effect = RuntimeError("render failure")
    viewer._renderer = renderer
    viewer._render_timer.stop()
    with pytest.raises(RuntimeError, match="render failure"):
        viewer._update_render()
