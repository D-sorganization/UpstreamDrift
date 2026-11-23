"""Native Qt input drives the lower-body panel and real MuJoCo simulation.

Only the external passive OpenGL viewer is replaced: no window, renderer,
physics step, controller, history, or Qt signal is mocked.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator

import numpy as np
import pytest

try:
    import mujoco  # noqa: F401 - optional native dependency probe
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QPushButton, QSlider
except (ImportError, OSError) as exc:
    pytest.skip(f"PyQt6/MuJoCo not loadable: {exc}", allow_module_level=True)

from src.engines.lower_body_model.builder import build_lower_body_xml
from src.engines.lower_body_model.launch_pyqt6 import ControlPanel
from src.engines.lower_body_model.simulator import LowerBodySimulator
from src.shared.python.theme.theme_manager import ThemeManager

pytestmark = pytest.mark.unit


class _HeadlessViewer:
    """Passive-viewer boundary sink; keep it alive until fixture teardown."""

    def __init__(self) -> None:
        self.closed = threading.Event()

    def is_running(self) -> bool:
        return not self.closed.is_set()

    def sync(self) -> None:
        """No OpenGL presentation in an offscreen test."""


@pytest.fixture
def application() -> QApplication:
    """Keep QTest usable when the diff runner disables pytest-qt."""
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    return app


@pytest.fixture
def panel(
    application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ControlPanel]:
    # Each launch owns a real manager, with no stale window from another test.
    monkeypatch.setattr(ThemeManager, "_instance", None)
    sink = _HeadlessViewer()
    sim = LowerBodySimulator(build_lower_body_xml())
    window = ControlPanel(sim, sink)
    window.show()
    QTest.qWait(20)
    assert window.isVisible()
    try:
        yield window
    finally:
        # Also stop the unfixed implementation after a red lifecycle assertion.
        sink.closed.set()
        window.physics_thread.join(timeout=2)
        window.timer.stop()
        window.close()
        window.deleteLater()
        application.processEvents()


def _wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(300):
        if predicate():
            return
        QTest.qWait(10)
    assert predicate(), "Lower-body action did not finish within 3 seconds"


def _slider_value(slider: QSlider, value: int) -> None:
    slider.setFocus()
    QTest.keyClick(slider, Qt.Key.Key_Home)
    for _ in range(value - slider.minimum()):
        QTest.keyClick(slider, Qt.Key.Key_Right)
    assert slider.value() == value


def _click_named(panel: ControlPanel, text: str) -> None:
    buttons = [b for b in panel.findChildren(QPushButton) if b.text() == text]
    assert len(buttons) == 1
    QTest.mouseClick(buttons[0], Qt.MouseButton.LeftButton)


def test_launch_edit_stance_gains_and_reset(panel: ControlPanel) -> None:
    assert panel.windowTitle() == "Lower Body Control Panel"
    _slider_value(panel.tilt_slider, 20)
    _slider_value(panel.knee_slider, 30)
    _slider_value(panel.foot_slider, 10)
    _slider_value(panel.kp_slider, 50)
    _slider_value(panel.kd_slider, 5)
    _click_named(panel, "Apply Initial Stance & Reset")

    sim = panel.sim
    assert panel.tilt_lbl.text() == "20"
    assert panel.knee_lbl.text() == "30"
    assert panel.foot_lbl.text() == "10"
    assert sim.kp_stability == 50
    assert sim.kd_stability == 5
    assert sim.data.joint("r_knee").qpos[0] == pytest.approx(np.radians(30))
    np.testing.assert_allclose(sim.data.qpos, sim.qpos_target)
    assert sim.data.time == 0
    assert not sim.history


def test_play_pause_scrub_and_full_reset(panel: ControlPanel) -> None:
    sim = panel.sim
    initial = sim.data.qpos.copy()
    QTest.mouseClick(panel.play_btn, Qt.MouseButton.LeftButton)
    assert panel.play_btn.text() == "Pause"
    assert not panel.timeline_slider.isEnabled()
    _wait_until(lambda: len(sim.history) >= 10)
    QTest.mouseClick(panel.play_btn, Qt.MouseButton.LeftButton)
    assert not panel.is_playing
    assert panel.timeline_slider.isEnabled()
    assert sim.data.time > 0
    assert np.isfinite(sim.data.qpos).all()
    assert not np.array_equal(sim.data.qpos, initial)

    first = sim.history[0]
    _slider_value(panel.timeline_slider, 0)
    np.testing.assert_allclose(sim.data.qpos, first["qpos"])
    assert sim.data.time == pytest.approx(first["time"])
    paused = sim.data.qpos.copy()
    QTest.qWait(60)
    np.testing.assert_array_equal(sim.data.qpos, paused)

    QTest.mouseClick(panel.full_reset_btn, Qt.MouseButton.LeftButton)
    assert sim.data.time == 0
    assert not sim.history
    assert panel.timeline_slider.maximum() == 0
    assert panel.play_btn.text() == "Play"
    np.testing.assert_allclose(sim.data.qpos, initial)


def test_close_while_playing_stops_panel_thread_and_timer(panel: ControlPanel) -> None:
    QTest.mouseClick(panel.play_btn, Qt.MouseButton.LeftButton)
    _wait_until(lambda: len(panel.sim.history) >= 5)
    assert panel.close()
    _wait_until(lambda: not panel.physics_thread.is_alive())
    assert not panel.timer.isActive()
    assert not panel.is_playing
    stopped_time = panel.sim.data.time
    QTest.qWait(60)
    assert panel.sim.data.time == stopped_time
