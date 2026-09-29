"""The video is the star: central live view, docked controls, log drawer (#9846).

Measured offscreen on ``main`` before this change, at a 1600x900 window: the
controls were the ``PaneHost`` central widget and took **3190 px** while each
of the three viewing docks got **68 px**, and the whole tile reported a
``minimumSizeHint().width()`` of **3276 px** — wider than a 1920 px screen.
The reflowing action grid (#9844) brought that floor to **1737 px**, still
too wide for a laptop panel, because the header (1821 px unwrapped) and the
record transport (982 px) then set it. With the inversion here the tile
measures **423 px** and, at a 1280x800 window, the central live view holds
**656 px** against **240 px** of controls.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtCore import QByteArray, QSettings
from PyQt6.QtWidgets import QApplication, QMainWindow
from src.tools.capture_rig import gui
from src.tools.capture_rig.layout import LAST_LAYOUT, STATE_VERSION
from src.tools.capture_rig.panes import LOG_KEY

pytestmark = [pytest.mark.unit, pytest.mark.ui]

#: A laptop panel is 1366 px wide; the tile must fit well inside one.
MAX_TILE_WIDTH = 900
WINDOW = (1280, 800)

_APP: QApplication | None = None  # a dropped QApplication aborts the process


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # type: ignore[assignment]
    return _APP  # type: ignore[return-value]


def _settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "layouts.ini"), QSettings.Format.IniFormat)


def _shown(tmp_path: Path) -> tuple[QMainWindow, gui.CaptureRigWidget]:
    """The tile in a 1280x800 window, laid out and ready to measure."""
    app = _app()
    widget = gui.CaptureRigWidget(settings=_settings(tmp_path))
    window = QMainWindow()
    window.setCentralWidget(widget)
    window.resize(*WINDOW)
    window.show()
    for _ in range(3):
        app.processEvents()
    return window, widget


def test_the_tile_fits_a_laptop_panel(tmp_path: Path) -> None:
    """3276 px on main, 1737 px with the grid fix alone, <= 900 px here.

    The floor used to be the controls sitting in the central widget; it is
    now the widest single control, because the header, the status strip, the
    layout bar and the record transport all wrap and every dock scrolls.
    """
    _app()
    widget = gui.CaptureRigWidget(settings=_settings(tmp_path))
    assert widget.minimumSizeHint().width() <= MAX_TILE_WIDTH
    widget.shutdown()


def test_the_live_view_is_wider_than_the_controls_at_1280x800(tmp_path: Path) -> None:
    """Measured: 656 px of video against a 240 px control column."""
    window, widget = _shown(tmp_path)
    live = widget.panes.centralWidget()
    assert live is not None
    controls = max(
        widget.panes.docks["rail"].width(), widget.panes.docks["inputs"].width()
    )
    assert live.width() > controls
    assert live.width() > window.width() // 3
    widget.shutdown()
    window.close()


def test_the_log_is_a_drawer_that_starts_shut_and_takes_no_stretch(
    tmp_path: Path,
) -> None:
    """The log had stretch 1 in the middle column and ate every spare pixel."""
    window, widget = _shown(tmp_path)
    assert not widget.panes.is_visible(LOG_KEY)
    assert not widget.log_toggle.isChecked()
    live_before = widget.panes.centralWidget()
    assert live_before is not None
    width_before, height_before = live_before.width(), live_before.height()

    widget.log_toggle.setChecked(True)
    for _ in range(3):
        _app().processEvents()
    assert widget.panes.is_visible(LOG_KEY)
    # The drawer is tabbed behind the action grid, so the video keeps its box.
    assert (live_before.width(), live_before.height()) == (width_before, height_before)

    widget.log_toggle.setChecked(False)
    for _ in range(3):
        _app().processEvents()
    assert not widget.panes.is_visible(LOG_KEY)
    widget.shutdown()
    window.close()


def test_reset_layout_leaves_the_drawer_shut(tmp_path: Path) -> None:
    window, widget = _shown(tmp_path)
    widget.log_toggle.setChecked(True)
    widget.layout_bar.reset_button.click()
    for _ in range(3):
        _app().processEvents()
    assert not widget.panes.is_visible(LOG_KEY)
    assert not widget.log_toggle.isChecked()  # the header stops lying
    widget.shutdown()
    window.close()


def test_a_layout_saved_by_the_old_arrangement_is_declined_not_crashed(
    tmp_path: Path,
) -> None:
    """Pre-#9846 states name docks that no longer exist (preview was one).

    They are written with an older ``STATE_VERSION``, so Qt declines them and
    the tile keeps its default instead of half-restoring or raising.
    """
    settings = _settings(tmp_path)
    _app()
    scratch = gui.CaptureRigWidget(settings=settings)
    old = QMainWindow()
    old.setObjectName("old")
    stale = old.saveState(STATE_VERSION - 1).data()  # an arrangement from before
    scratch.shutdown()
    settings.setValue(f"layouts/{LAST_LAYOUT}/state", QByteArray(stale))
    settings.setValue("layouts/bay/state", QByteArray(stale))
    settings.sync()

    widget = gui.CaptureRigWidget(settings=settings)  # restores the stale last
    assert set(widget.panes.docks) >= {"rail", "inputs", LOG_KEY}
    assert not widget.panes.is_visible(LOG_KEY)
    assert widget.layout_bar.apply("bay") is False  # declined, and says so
    assert not widget.panes.is_visible(LOG_KEY)
    widget.shutdown()


def test_the_rail_shows_the_workflow_and_its_actions_run_commands() -> None:
    """The rail replaces the old step list and drives the tile's ``trigger``."""
    _app()
    widget = gui.CaptureRigWidget()
    assert widget.rail.current_key == "setup"
    assert widget.rail.primary_action is not None
    assert set(widget.rail.rows) == set(widget.workflow.statuses())
    assert widget.rail.kind_of("setup") == "current"
    assert widget.rail.primary is not None
    assert (
        widget.rail.primary.text() == dict(widget._ACTIONS)[widget.rail.primary_action]
    )

    widget.capture.plan_edit.setText("")  # make the action fail loudly
    widget.rail.action_triggered.emit("plan_check")
    assert "choose a plan file" in widget.log.toPlainText()
    widget.shutdown()
