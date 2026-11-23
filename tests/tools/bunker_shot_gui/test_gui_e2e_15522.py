"""Drive the standalone BunkerShot3D window with native offscreen Qt events.

Run explicitly with ``pytest -m e2e tests/tools/bunker_shot_gui/test_gui_e2e_15522.py``.
The real F0 solver runs in its production worker; only mesh/grid resolution
is reduced, using the same controls a user edits. No solver or Qt mocks.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QAbstractSpinBox,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QWidget,
)
from pytestqt.qtbot import QtBot  # noqa: E402

from src.tools.bunker_shot_gui.gui import BunkerShotWindow  # noqa: E402

pytestmark = [pytest.mark.e2e, pytest.mark.ui, pytest.mark.headless_safe]


def _reveal(control: QWidget) -> None:
    """Scroll the actual input into view before delivering a mouse event."""
    parent = control.parentWidget()
    while parent is not None:
        if isinstance(parent, QScrollArea):
            parent.ensureWidgetVisible(control)
            break
        parent = parent.parentWidget()


def _edit(control: QLineEdit | QAbstractSpinBox, text: str) -> None:
    """Replace a field through keyboard input and commit it with Tab."""
    _reveal(control)
    QTest.mouseClick(control, Qt.MouseButton.LeftButton)
    QTest.keyClick(control, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(control, Qt.Key.Key_Backspace)
    QTest.keyClicks(control, text)
    QTest.keyClick(control, Qt.Key.Key_Tab)


def _click(button: QPushButton) -> None:
    _reveal(button)
    assert button.isVisible() and button.isEnabled()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)


@pytest.fixture
def window(qtbot: QtBot) -> Iterator[BunkerShotWindow]:
    """Launch the production window and always join its worker on teardown."""
    launched = BunkerShotWindow()
    qtbot.addWidget(launched)
    try:
        launched.resize(1280, 900)
        launched.show()
        qtbot.waitUntil(launched.isVisible)
        conditions = launched.widget._conditions
        for control, value in (
            (conditions._grid, "2"),
            (conditions._stations, "5"),
            (conditions._profile_points, "12"),
        ):
            _edit(control, value)
            assert control.value() == int(value)
        yield launched
    finally:
        launched.widget.cleanup()
        launched.close()


def _run(window: BunkerShotWindow, qtbot: QtBot) -> None:
    workbench = window.widget
    with qtbot.waitSignal(workbench.evaluation_changed, timeout=40000):
        _click(workbench._run_button)
    qtbot.waitUntil(lambda: not workbench.action_bar.is_running, timeout=5000)
    assert workbench.current_evaluation is not None
    assert workbench._run_button.isEnabled()


def test_launch_edit_and_run_real_f0(window: BunkerShotWindow, qtbot: QtBot) -> None:
    workbench = window.widget
    assert "BunkerShot3D" in window.windowTitle()
    assert "No shot has been run yet" in workbench.verdict_text
    assert not workbench.control_enabled("save_evaluation")
    _edit(workbench._design_a._name, "QTest wedge")
    _run(window, qtbot)

    assert "QTest wedge" in workbench.report_text
    assert "F0 dynamic RFT" in workbench.report_text
    assert "Peak resultant force" in workbench.report_text
    assert "BEYOND" in workbench.verdict_text.upper()
    assert workbench._bounce_map_a.values.size > 0
    assert workbench._window_map_b.values.size == 0
    assert workbench.control_enabled("save_evaluation")
    assert window._save_action.isEnabled()


def test_invalid_input_clears_previous_result_and_save_actions(
    window: BunkerShotWindow, qtbot: QtBot
) -> None:
    """An invalid run must not leave an old evaluation available for export."""
    workbench = window.widget
    _run(window, qtbot)
    _edit(workbench._design_a._name, "")
    _click(workbench._run_button)

    assert "INPUT ERROR" in workbench.verdict_text
    assert "model was never asked for a number" in workbench.report_text
    assert not workbench.action_bar.is_running
    assert workbench.current_evaluation is None
    assert not workbench.control_enabled("save_evaluation")
    assert not workbench.control_enabled("export_views")
    assert not window._save_action.isEnabled()
    assert not window._export_action.isEnabled()
    assert workbench._bounce_map_a.values.size == 0
    assert not workbench._field_a.is_playing
    assert not workbench._scene_a.has_shot
    assert not workbench._traces_a.has_shot

    _edit(workbench._design_a._name, "Recovered wedge")
    _run(window, qtbot)
    assert "Recovered wedge" in workbench.report_text
    assert window._save_action.isEnabled()


def test_compare_uses_both_edited_designs(
    window: BunkerShotWindow, qtbot: QtBot
) -> None:
    workbench = window.widget
    _edit(workbench._design_a._name, "Candidate left")
    _edit(workbench._design_b._name, "Candidate right")
    with qtbot.waitSignal(workbench.evaluation_changed, timeout=40000):
        _click(workbench._compare_button)
    qtbot.waitUntil(lambda: not workbench.action_bar.is_running, timeout=5000)

    assert "Candidate left" in workbench.report_text
    assert "Candidate right" in workbench.report_text
    assert workbench._bounce_map_a.values.size > 0
    assert workbench._bounce_map_b.values.size > 0
    assert workbench._run_button.isEnabled()
    assert workbench._compare_button.isEnabled()
