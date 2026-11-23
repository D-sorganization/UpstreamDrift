"""Real Qt events drive the pendulum launch, solver, analysis and playback.

Run with ``QT_QPA_PLATFORM=offscreen pytest -m e2e <this file>``.
Requires PyQt6, pytest-qt, scipy, matplotlib and the pinned Tools submodule;
the solver and widgets are never mocked. QSettings are isolated by conftest.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_QT_API", "pyqt6")

qt = pytest.importorskip("PyQt6")
if getattr(qt, "__ud_fake__", False):
    pytest.skip("Pendulum E2E requires real PyQt6", allow_module_level=True)
pytest.importorskip("pytestqt")
pytest.importorskip("PyQt6.QtTest")

from PyQt6 import sip
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QLineEdit
from pytestqt.qtbot import QtBot

from src.shared.python.pendulum_simulator.__main__ import get_dockable_ui
from src.shared.python.pendulum_simulator.gui.main_window import MainWindow

pytestmark = [pytest.mark.e2e, pytest.mark.headless_safe]


@pytest.fixture
def window(qtbot: QtBot) -> Iterator[MainWindow]:
    """Launch the real application; drain solver threads before closing it."""
    widget = get_dockable_ui()
    qtbot.addWidget(widget)
    widget.resize(1280, 800)
    widget.show()
    qtbot.waitUntil(widget.isVisible)
    try:
        yield widget
    finally:
        for panel in widget._panels:
            panel._timer.stop()
            thread = getattr(panel, "_sim_thread", None)
            if thread is not None and not sip.isdeleted(thread):
                thread.quit()
                assert thread.wait(10_000), "Pendulum solver failed to stop"
        widget.close()


def _type_value(edit: QLineEdit, text: str) -> None:
    """Replace an input using keyboard events, including editingFinished."""
    edit.setFocus()
    QTest.keyClick(edit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(edit, text)
    QTest.keyClick(edit, Qt.Key.Key_Tab)
    assert edit.text() == text


@pytest.mark.parametrize("model_index, state_columns", [(0, 4), (1, 6)])
def test_launch_run_analyse_scrub_and_reset(
    window: MainWindow, qtbot: QtBot, model_index: int, state_columns: int
) -> None:
    """First-run completion publishes the new trajectory to every consumer."""
    toolbar = window._toolstrip
    # Select the model through its visible combo, rather than calling a slot.
    combo = toolbar.cmb_model
    combo.setFocus()
    QTest.keyClick(combo, Qt.Key.Key_Home)
    for _ in range(model_index):
        QTest.keyClick(combo, Qt.Key.Key_Down)
    assert window._tabs.currentIndex() == model_index
    panel = window._panels[model_index]
    controls = panel.controls
    _type_value(controls.inp_tend.edit, "0.2")
    if model_index == 0:
        _type_value(controls.inp_dt.edit, "0.01")
    assert panel.current_n_steps() == 0

    with qtbot.waitSignal(panel.sim_finished, timeout=15_000):
        QTest.mouseClick(toolbar.btn_run, Qt.MouseButton.LeftButton)
    result = panel._result
    assert result is not None
    assert result.t[-1] == pytest.approx(0.2)
    assert result.states.shape == (result.n_steps, state_columns)
    assert np.isfinite(result.states).all()
    assert not np.allclose(result.states[0], result.states[-1])
    assert window._analysis_tab._result is result
    assert toolbar._frame_slider.maximum() == result.n_steps - 1
    assert toolbar.btn_run.isEnabled()
    assert panel.pendulum._result is result

    # Pause auto-play, scrub with keyboard events, then exercise play/pause.
    if toolbar.btn_play.isChecked():
        QTest.mouseClick(toolbar.btn_play, Qt.MouseButton.LeftButton)
    slider = toolbar._frame_slider
    slider.setFocus()
    QTest.keyClick(slider, Qt.Key.Key_End)
    assert panel.pendulum._current_idx == result.n_steps - 1
    QTest.keyClick(slider, Qt.Key.Key_Home)
    assert panel.pendulum._current_idx == 0
    QTest.mouseClick(toolbar.btn_play, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: panel.pendulum._current_idx > 0, timeout=5000)
    QTest.mouseClick(toolbar.btn_play, Qt.MouseButton.LeftButton)
    assert not panel._timer.isActive()

    QTest.mouseClick(toolbar.btn_reset, Qt.MouseButton.LeftButton)
    assert panel.current_n_steps() == 0
    assert panel.pendulum._result is None
    assert not panel._timer.isActive()
