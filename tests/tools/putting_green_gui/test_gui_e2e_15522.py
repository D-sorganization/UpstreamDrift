"""Real Qt input -> putting solver -> metrics/playback, without a GL context.

Run explicitly with ``QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyqt6
python3 -m pytest tests/tools/putting_green_gui/test_gui_e2e_15522.py -m e2e``.
Only the optional OpenGL import is disabled; Qt, timers and physics are real.
This exercises the supported no-renderer path, not 3D rendering correctness.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_QT_API", "pyqt6")
pytest.importorskip("pytestqt")
qtcore = pytest.importorskip("PyQt6.QtCore")
if not isinstance(qtcore.PYQT_VERSION_STR, str):
    pytest.skip("Real PyQt6 is required, not test stubs", allow_module_level=True)

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QDoubleSpinBox, QPushButton  # noqa: E402
from pytestqt.qtbot import QtBot  # noqa: E402

from src.shared.python.ui.units import UnitSystem, set_unit_preference  # noqa: E402
from src.tools.putting_green_gui.gui import (  # noqa: E402
    PuttingGreenWidget,
    PuttingGreenWindow,
)

pytestmark = [pytest.mark.e2e, pytest.mark.headless_safe]


@pytest.fixture
def window(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> Iterator[PuttingGreenWindow]:
    # Exercise the application's ImportError fallback without replacing any
    # widgets, signals, solver calls or trajectory data with test doubles.
    monkeypatch.setitem(sys.modules, "pyqtgraph.opengl", None)
    set_unit_preference(UnitSystem.METRIC)
    win = PuttingGreenWindow()
    qtbot.addWidget(win)
    win.resize(1280, 900)
    win.show()
    qtbot.waitUntil(win.isVisible)
    try:
        yield win
    finally:
        win.close()
        widget = win.centralWidget()
        assert isinstance(widget, PuttingGreenWidget)
        assert not widget._anim_timer.isActive()
        assert not win.isVisible()


def _click(widget: PuttingGreenWidget, label: str) -> None:
    buttons = [b for b in widget.findChildren(QPushButton) if label in b.text()]
    assert len(buttons) == 1, f"Expected one {label!r} button"
    button = buttons[0]
    assert button.isVisible() and button.isEnabled()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)


def _edit(spin: QDoubleSpinBox, text: str) -> None:
    spin.setFocus()
    QTest.keyClick(spin, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(spin, text)
    QTest.keyClick(spin, Qt.Key.Key_Return)
    assert spin.value() == pytest.approx(float(text))


@pytest.mark.parametrize(
    "preset,speed,distance", [("Short", 1.5, 1.52), ("Long", 4.0, 9.14)]
)
def test_preset_edit_and_real_simulation(
    window: PuttingGreenWindow, preset: str, speed: float, distance: float
) -> None:
    widget = window.centralWidget()
    assert isinstance(widget, PuttingGreenWidget)
    assert widget._gl_view is None
    assert "Configure putt parameters" in widget._results_text.toPlainText()
    _click(widget, preset)
    assert widget._speed_spin.value() == pytest.approx(speed)
    assert widget._distance_spin.value() == pytest.approx(distance)
    _edit(widget._slope_spin, "0")
    _edit(widget._aim_spin, "8")
    _click(widget, "Simulate Putt")

    scene = widget._scene
    assert scene is not None
    trajectory = scene.trajectory_xyz
    assert trajectory.shape[0] > 10 and trajectory.shape[1] == 3
    assert np.isfinite(trajectory).all()
    assert scene.duration_s > 0 and scene.total_roll_m > 0
    assert scene.launch_speed_ms > speed
    # An edited positive aim on a flat green produces positive lateral travel.
    assert trajectory[-1, 1] > trajectory[0, 1] + 0.1
    text = widget._results_text.toPlainText()
    assert text.startswith("Putting Simulation - ")
    assert f"{scene.total_roll_m:.2f} m" in text
    assert f"{scene.duration_s:.2f} s" in text
    assert scene.roll_model in text
    assert widget._run_btn.isEnabled()

    # A second real run must replace the first result and react to keyboard input.
    _edit(widget._speed_spin, "1.0")
    _click(widget, "Simulate Putt")
    slower = widget._scene
    assert slower is not None and slower is not scene
    assert slower.launch_speed_ms < scene.launch_speed_ms
    assert slower.total_roll_m < scene.total_roll_m
    assert (
        widget.playback_bar().scrub_slider.maximum() == len(slower.trajectory_xyz) - 1
    )


def test_play_pause_scrub_and_replay_without_opengl(
    window: PuttingGreenWindow, qtbot: QtBot
) -> None:
    widget = window.centralWidget()
    assert isinstance(widget, PuttingGreenWidget)
    _click(widget, "Short")
    _click(widget, "Simulate Putt")
    bar = widget.playback_bar()
    slider = bar.scrub_slider
    assert slider.maximum() > 0
    qtbot.waitUntil(lambda: slider.value() > 0, timeout=2000)
    _click(widget, "Pause")
    assert not widget._anim_timer.isActive()
    paused = slider.value()
    QTest.qWait(100)
    assert slider.value() == paused
    QTest.keyClick(slider, Qt.Key.Key_End)
    assert slider.value() == slider.maximum()
    assert widget._anim_index == slider.maximum()
    assert bar.play_btn.text() == "Play"
    QTest.keyClick(slider, Qt.Key.Key_Home)
    assert slider.value() == 0
    _click(widget, "Play")
    qtbot.waitUntil(lambda: slider.value() > 0, timeout=2000)
    qtbot.waitUntil(lambda: bar.play_btn.text() == "Play", timeout=6000)
    assert slider.value() == slider.maximum()
    assert not widget._anim_timer.isActive()
    _click(widget, "Play")
    qtbot.waitUntil(lambda: 0 < slider.value() < slider.maximum(), timeout=2000)
    assert widget._anim_timer.isActive()
