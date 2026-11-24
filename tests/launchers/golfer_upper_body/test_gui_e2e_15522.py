"""Native QTest flows for the constrained, seven-actuator golfer model.

Run with QT_QPA_PLATFORM=offscreen and the pinned Tools submodule initialized.
The launcher, solver thread, closed-chain physics and playback are real.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_QT_API", "pyqt6")

qt = pytest.importorskip("PyQt6")
if getattr(qt, "__ud_fake__", False):
    pytest.skip("Golfer E2E requires real PyQt6", allow_module_level=True)
pytest.importorskip("pytestqt")
pytest.importorskip("PyQt6.QtTest")

from PyQt6 import sip
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLineEdit, QMessageBox
from pytestqt.qtbot import QtBot

from src.shared.python.pendulum_simulator.__main__ import get_dockable_ui
from src.shared.python.pendulum_simulator.gui.main_window import MainWindow

pytestmark = pytest.mark.unit


@pytest.fixture
def dialogs(qapp: QApplication) -> Iterator[list[str]]:
    """Dismiss real modal dialogs via clicks, retaining messages for assertions."""
    messages: list[str] = []

    def dismiss() -> None:
        dialog = qapp.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            messages.append(dialog.text())
            QTest.mouseClick(
                dialog.button(QMessageBox.StandardButton.Ok), Qt.MouseButton.LeftButton
            )

    timer = QTimer()
    timer.timeout.connect(dismiss)
    timer.start(10)
    try:
        yield messages
    finally:
        timer.stop()


def _release_theme_manager(widget: MainWindow) -> None:
    """Drop a ThemeManager singleton bound to *widget* before it is deleted.

    The singleton keeps the first window it was created with; once this window
    is gone, later windows (e.g. the cross-engine dashboard) would restyle a
    deleted C++ object.
    """
    for module in list(sys.modules.values()):
        manager_cls = getattr(module, "ThemeManager", None)
        reset = getattr(manager_cls, "reset_instance", None)
        instance = getattr(manager_cls, "_instance", None)
        if callable(reset) and getattr(instance, "main_window", None) is widget:
            reset()


@pytest.fixture
def window(qtbot: QtBot, dialogs: list[str]) -> Iterator[MainWindow]:
    """Select the golfer through Qt events and drain all workers on teardown."""
    widget = get_dockable_ui()
    qtbot.addWidget(widget)
    widget.resize(1280, 800)
    widget.show()
    qtbot.waitUntil(widget.isVisible)
    try:
        combo = widget._toolstrip.cmb_model
        combo.setFocus()
        QTest.keyClick(combo, Qt.Key.Key_End)
        assert widget._tabs.currentWidget() is widget._golfer_panel
        assert combo.currentText() == "Upper Body"
        yield widget
    finally:
        for panel in widget._panels:
            panel._timer.stop()
            thread = getattr(panel, "_sim_thread", None)
            if thread is not None and not sip.isdeleted(thread):
                thread.quit()
                assert thread.wait(10_000), "Golfer solver failed to stop"
        _release_theme_manager(widget)
        widget.close()


def _type(edit: QLineEdit, value: str) -> None:
    edit.setFocus()
    QTest.keyClick(edit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(edit, value)
    QTest.keyClick(edit, Qt.Key.Key_Tab)
    assert edit.text() == value


def _run(window: MainWindow, qtbot: QtBot) -> None:
    panel = window._golfer_panel
    with qtbot.waitSignal(panel.sim_finished, timeout=20_000):
        QTest.mouseClick(window._toolstrip.btn_run, Qt.MouseButton.LeftButton)
    result = panel._result
    assert result is not None
    assert result.t[-1] == pytest.approx(0.05)
    # Eight generalized coordinates (including club) and eight velocities.
    assert result.states.shape == (result.n_steps, 16)
    assert np.isfinite(result.states).all()
    assert panel.pendulum._result is result
    assert window._analysis_tab._result is result
    assert window._toolstrip.btn_run.isEnabled()
    assert all(p._result is None for p in window._panels[:2])
    # Pause through the visible toolbar before asserting frame positions.
    if window._toolstrip.btn_play.isChecked():
        QTest.mouseClick(window._toolstrip.btn_play, Qt.MouseButton.LeftButton)


@pytest.mark.parametrize("backswing", [False, True], ids=["address", "backswing"])
def test_golfer_preset_run_closed_chain_and_playback(
    window: MainWindow, qtbot: QtBot, dialogs: list[str], backswing: bool
) -> None:
    panel = window._golfer_panel
    controls = panel.controls
    if backswing:
        combo = controls.preset_combo
        combo.setFocus()
        QTest.keyClick(combo, Qt.Key.Key_End)
    expected_hub = [-5.0, 2.0] if backswing else [0.0]
    assert controls.get_params()["hub_coeffs"] == expected_hub
    _type(controls.inp_tend.edit, "0.05")
    _run(window, qtbot)
    assert not dialogs
    result = panel._result
    assert not np.allclose(result.states[0], result.states[-1])
    assert result.constraint_violation_at(0) < 1e-6
    assert result.constraint_violation_at(result.n_steps - 1) < 1e-3
    assert np.isfinite(result.positions_at(result.n_steps - 1)["club_tip"]).all()
    assert len(result.torques_at(0)) == 7
    assert result.torques_at(0)[0] == pytest.approx(expected_hub[0])
    assert result.params.L_club == pytest.approx(1.1)

    slider = window._toolstrip._frame_slider
    slider.setFocus()
    QTest.keyClick(slider, Qt.Key.Key_End)
    assert panel.pendulum._current_idx == result.n_steps - 1
    QTest.keyClick(slider, Qt.Key.Key_Home)
    assert panel.pendulum._current_idx == 0
    QTest.mouseClick(window._toolstrip.btn_play, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: panel.pendulum._current_idx > 0, timeout=5000)
    QTest.mouseClick(window._toolstrip.btn_play, Qt.MouseButton.LeftButton)
    assert not panel._timer.isActive()
    QTest.mouseClick(window._toolstrip.btn_reset, Qt.MouseButton.LeftButton)
    assert panel._result is None
    assert panel.pendulum._result is None
    assert panel.current_n_steps() == 0
    assert not panel._timer.isActive()


def test_golfer_invalid_duration_recovers_through_real_dialog(
    window: MainWindow, qtbot: QtBot, dialogs: list[str]
) -> None:
    panel = window._golfer_panel
    _type(panel.controls.inp_tend.edit, "-1")
    QTest.mouseClick(window._toolstrip.btn_run, Qt.MouseButton.LeftButton)
    assert len(dialogs) == 1 and "positive" in dialogs.pop().lower()
    assert panel._result is None
    assert window._toolstrip.btn_run.isEnabled()
    _type(panel.controls.inp_tend.edit, "0.05")
    _run(window, qtbot)
    assert not dialogs
