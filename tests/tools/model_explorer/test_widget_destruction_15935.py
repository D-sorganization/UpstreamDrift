"""Qt destruction callbacks must not leak into the rest of the suite (#15935).

``MuJoCoViewerWidget`` runs a 20 FPS render timer. The timer is parented to the
widget (issue #13058), but the viewport ``QLabel`` is a separate child that Qt
can delete first, and any caller that still holds the Python widget can then
reach ``_update_render``. The resulting
``RuntimeError: wrapped C/C++ object of type QLabel has been deleted`` escapes
through the Qt event loop and is attributed to whichever unrelated test happens
to be processing events, which is how the merge-group unit suite went red in
run 46119538502 while failing ``tests/tools/test_terrain_engine_gui.py``.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6 import sip
from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication

from src.tools.model_explorer.mujoco_viewer import MuJoCoViewerWidget

pytestmark = [pytest.mark.unit, pytest.mark.ui]

_APP: QApplication | None = None


def _ensure_qapp() -> QApplication:
    """Return the process-wide QApplication, creating it on first use."""
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app
    return app


class _StubRenderer:
    """Offscreen renderer stand-in that always produces a frame."""

    def render(self) -> np.ndarray:
        """Return a non-empty RGB frame so ``_update_render`` reaches the viewport."""
        return np.zeros((8, 8, 3), dtype=np.uint8)


def _widget_with_frames() -> MuJoCoViewerWidget:
    """Return a viewer widget whose renderer always yields a frame."""
    _ensure_qapp()
    widget = MuJoCoViewerWidget()
    widget._renderer = _StubRenderer()
    return widget


def _drain_deferdeletes() -> None:
    """Execute pending ``deleteLater()`` requests immediately."""
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def test_render_after_viewport_deletion_does_not_raise() -> None:
    """A render tick on a deleted viewport is a no-op, not an event-loop error."""
    widget = _widget_with_frames()
    viewport = widget._viewport

    viewport.deleteLater()
    _drain_deferdeletes()
    assert sip.isdeleted(viewport)

    widget._update_render()


def test_render_timer_stops_when_viewport_is_destroyed() -> None:
    """Destroying the viewport must stop the timer that draws into it."""
    widget = _widget_with_frames()
    assert widget._render_timer.isActive()

    widget._viewport.deleteLater()
    _drain_deferdeletes()

    assert not widget._render_timer.isActive()


def test_render_timer_stops_when_widget_is_destroyed() -> None:
    """Destroying the widget stops the timer before it can tick on dead children."""
    widget = _widget_with_frames()

    widget.deleteLater()
    _drain_deferdeletes()

    assert sip.isdeleted(widget._render_timer)


def test_render_before_viewport_deletion_still_paints() -> None:
    """The live path still paints the frame into the viewport."""
    widget = _widget_with_frames()
    widget.show()

    widget._update_render()

    assert not sip.isdeleted(widget._viewport)
    assert widget._viewport.pixmap() is not None
