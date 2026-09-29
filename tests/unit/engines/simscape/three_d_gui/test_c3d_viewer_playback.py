"""Playback / shortcut tests for the C3D viewer's 3D tab."""

from __future__ import annotations

import sys

import pytest

from ._viewer_test_helpers import make_synthetic_model

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qt_app():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


@pytest.fixture()
def tab(qt_app):
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore

    model = make_synthetic_model(
        ["m0", "m1", "m2", "m3", "m4"], n_frames=100, point_rate=100.0
    )
    t = Viewer3DTab()
    t.update_from_model(model)
    # Force a deterministic single-marker selection for predictable assertions.
    t.list_markers_3d.clearSelection()
    item = t.list_markers_3d.item(0)
    assert item is not None
    item.setSelected(True)
    return t


def test_play_pause_toggle(tab) -> None:
    assert tab.is_playing is False
    tab.toggle_play()
    assert tab.is_playing is True
    assert tab._timer.isActive()
    tab.toggle_play()
    assert tab.is_playing is False
    assert not tab._timer.isActive()


def test_speed_validation(tab) -> None:
    with pytest.raises(TypeError):
        tab.set_speed("fast")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tab.set_speed(0.0)
    with pytest.raises(ValueError):
        tab.set_speed(-1.0)
    tab.set_speed(2.0)
    assert tab.combo_speed.currentData() == pytest.approx(2.0)


def test_frame_validation(tab) -> None:
    with pytest.raises(TypeError):
        tab.set_frame("0")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tab.set_frame(999)
    tab.set_frame(10)
    assert tab.slider_frame.value() == 10


def test_timer_advances_slider(tab) -> None:
    tab.slider_frame.setValue(0)
    # Manually trigger a tick rather than relying on the event loop.
    tab._on_timer_tick()
    assert tab.slider_frame.value() == 1
    tab._on_timer_tick()
    assert tab.slider_frame.value() == 2


def test_loop_wraparound_on(tab) -> None:
    tab.check_loop.setChecked(True)
    tab.slider_frame.setValue(tab._n_frames - 1)
    tab._on_timer_tick()
    assert tab.slider_frame.value() == 0


def test_loop_wraparound_off_stops(tab) -> None:
    tab.check_loop.setChecked(False)
    tab.play()
    tab.slider_frame.setValue(tab._n_frames - 1)
    tab._on_timer_tick()
    # Should have stopped, slider stays at last frame.
    assert tab.is_playing is False
    assert tab.slider_frame.value() == tab._n_frames - 1


def test_step_frame(tab) -> None:
    tab.slider_frame.setValue(50)
    tab.step_frame(1)
    assert tab.slider_frame.value() == 51
    tab.step_frame(-10)
    assert tab.slider_frame.value() == 41
    # Clamped at bounds.
    tab.step_frame(-9999)
    assert tab.slider_frame.value() == 0


def test_keyboard_shortcuts(qt_app, tab) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    tab.show()
    QTest.qWaitForWindowExposed(tab)
    tab.setFocus()

    tab.slider_frame.setValue(0)
    QTest.keyClick(tab, Qt.Key.Key_Right)
    assert tab.slider_frame.value() == 1
    QTest.keyClick(tab, Qt.Key.Key_End)
    assert tab.slider_frame.value() == tab._n_frames - 1
    QTest.keyClick(tab, Qt.Key.Key_Home)
    assert tab.slider_frame.value() == 0
    QTest.keyClick(tab, Qt.Key.Key_Space)
    assert tab.is_playing is True
    QTest.keyClick(tab, Qt.Key.Key_Space)
    assert tab.is_playing is False


def test_event_buttons_jump_to_time(qt_app) -> None:
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore

    model = make_synthetic_model(["m0"], n_frames=100, include_events=True)
    t = Viewer3DTab()
    t.update_from_model(model)
    assert len(t._event_buttons) == 4
    t.jump_to_time(model.events[2].time)
    expected = 2 * 100 // 3
    assert t.slider_frame.value() == expected
