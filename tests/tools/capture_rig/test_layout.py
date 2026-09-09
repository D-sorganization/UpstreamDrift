"""Movable panes, scroll areas and saved layouts for the Capture Rig tile."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QApplication, QLabel, QScrollArea, QWidget
from src.motion_capture.rig.plan import RigPlan
from src.motion_capture.rig.sources import SyntheticFrameSource
from src.tools.capture_rig import gui
from src.tools.capture_rig.commands import PlanSelection
from src.tools.capture_rig.layout import (
    LAST_LAYOUT,
    LayoutBar,
    LayoutStore,
    PaneHost,
)
from src.tools.capture_rig.preview import PreviewPanel

pytestmark = [pytest.mark.unit, pytest.mark.ui]

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # type: ignore[assignment]
    return _APP  # type: ignore[return-value]


def _settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "layouts.ini"), QSettings.Format.IniFormat)


def _pump(app: QApplication, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def _host() -> PaneHost:
    return PaneHost(
        QLabel("controls"),
        {
            "a": ("Pane A", QLabel("a"), Qt.DockWidgetArea.RightDockWidgetArea),
            "b": ("Pane B", QLabel("b"), Qt.DockWidgetArea.BottomDockWidgetArea),
        },
    )


def test_store_saves_lists_loads_and_deletes_named_layouts(tmp_path: Path) -> None:
    _app()
    store = LayoutStore(_settings(tmp_path))
    assert store.names() == []
    store.save("bay", b"\x01\x02", [300, 500])
    store.save("desk", b"\x03", [])
    store.save(LAST_LAYOUT, b"\x09", [1, 2])
    assert store.names() == ["bay", "desk"]  # the last layout is hidden
    assert store.load("bay") == (b"\x01\x02", [300, 500])
    assert store.load("desk") == (b"\x03", [])
    assert store.load("nope") is None
    store.delete("bay")
    assert store.names() == ["desk"]
    with pytest.raises(Exception, match="non-empty"):
        store.save(" ", b"", [])
    with pytest.raises(Exception, match="'/'"):
        store.load("a/b")


def test_pane_host_wraps_panes_in_scroll_areas_and_round_trips_state() -> None:
    _app()
    host = _host()
    assert set(host.docks) == {"a", "b"}
    assert all(isinstance(d.widget(), QScrollArea) for d in host.docks.values())
    host.set_floating("a", True)
    host.docks["b"].hide()
    floated = host.state()
    host.reset()
    assert not host.is_floating("a") and not host.docks["b"].isHidden()
    assert host.restore(floated)
    assert host.is_floating("a") and host.docks["b"].isHidden()
    assert not host.restore(b"")
    with pytest.raises(Exception, match="at least one pane"):
        PaneHost(QWidget(), {})


def test_layout_bar_saves_applies_and_restores_last(tmp_path: Path) -> None:
    _app()
    host = _host()
    store = LayoutStore(_settings(tmp_path))
    bar = LayoutBar(host, store, ask_name=lambda: "floating a")
    assert not bar.load_button.isEnabled()
    host.set_floating("a", True)
    bar.save_button.click()  # asks for a name through the injected callback
    assert bar.combo.currentText() == "floating a" and bar.load_button.isEnabled()
    host.reset()
    assert not host.is_floating("a")
    assert bar.apply("floating a") and host.is_floating("a")
    assert not bar.apply("unknown")
    bar.save_last()
    host.reset()
    assert bar.restore_last() and host.is_floating("a")
    bar.delete("floating a")
    assert bar.combo.count() == 0 and not bar.delete_button.isEnabled()


def test_pane_scrolls_when_its_content_is_wider_than_the_dock() -> None:
    _app()
    big = QLabel("x")
    big.setMinimumSize(3000, 100)
    host = PaneHost(
        QLabel("c"), {"big": ("Big", big, Qt.DockWidgetArea.RightDockWidgetArea)}
    )
    host.resize(600, 300)
    host.show()
    _pump(_app(), 0.1)
    area = host.docks["big"].widget()
    assert isinstance(area, QScrollArea)
    assert area.horizontalScrollBar().maximum() > 0
    host.close()


def test_preview_tiles_do_not_force_the_window_wide(tmp_path: Path) -> None:
    app = _app()
    plan = tmp_path / "plan.json"
    plan.write_text(
        '{"schema_version": "rig-plan/1.0.0", "name": "three", "cameras": ['
        '{"view": "cam_a", "serial": "1"}, {"view": "cam_b", "serial": "2"},'
        '{"view": "cam_c", "serial": "3"}]}',
        encoding="utf-8",
    )

    def synthetic(plan: RigPlan) -> dict[str, SyntheticFrameSource]:
        return {c.view: SyntheticFrameSource(c.identity) for c in plan.cameras}

    panel = PreviewPanel(source_factory=synthetic)
    panel.resize(600, 300)
    panel.show()
    panel.start(PlanSelection(plan=plan))
    _pump(app, 1.5)
    assert all(panel.frames_seen(v) for v in panel.views())
    assert panel.minimumSizeHint().width() < 400  # three tiles, no forced width
    panel.resize(300, 200)
    _pump(app, 0.3)
    assert panel.width() == 300
    panel.stop()
    panel.close()


def test_tile_hosts_panes_and_restores_the_last_layout(tmp_path: Path) -> None:
    _app()
    widget = gui.CaptureRigWidget(settings=_settings(tmp_path))
    assert set(widget.panes.docks) == {"preview", "playback", "results"}
    widget.panes.set_floating("preview", True)
    widget.layout_bar.save_as("bay")
    widget.shutdown()  # remembers the arrangement
    again = gui.CaptureRigWidget(settings=_settings(tmp_path))
    assert again.panes.is_floating("preview")
    assert again.layout_bar.combo.findText("bay") >= 0
    again.layout_bar.reset_button.click()
    assert not again.panes.is_floating("preview")
    again.shutdown()
