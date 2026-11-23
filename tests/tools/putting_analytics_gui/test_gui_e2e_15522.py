"""End-to-end PyQt6 interaction tests for the putting analytics window (#15522).

``tests/ui/tools/putting_analytics_gui/test_gui.py`` calls ``run_study()`` and
sets widget values programmatically. These tests instead drive the real window
the way a user does: ``QTest`` key presses type into the line edits and spin
boxes, a keyboard step picks the study, and a real left-button click on
"Run study" starts the off-thread worker. Assertions read what the user sees:
the button state, the status line and the result table.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, Qt, QThread  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

from src.tools.putting_analytics_gui.gui import MainWidget  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

TIMEOUT_S = 90.0
_NAV_KEYS = (Qt.Key.Key_Control, Qt.Key.Key_A)


def _pump_until(predicate: Callable[[], bool], timeout_s: float = TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        QThread.msleep(5)
    QCoreApplication.processEvents()
    return predicate()


def _type_into(widget: QWidget, text: str) -> None:
    """Focus ``widget``, replace its content and type ``text`` key by key."""
    widget.setFocus()
    QTest.keyClick(widget, _NAV_KEYS[1], Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(widget, text)


def _click(widget: QWidget) -> None:
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    assert isinstance(app, QApplication)
    return app


@pytest.fixture
def window(qapp: QApplication) -> Iterator[MainWidget]:
    widget = MainWidget()
    widget.show()
    assert QTest.qWaitForWindowExposed(widget)
    yield widget
    widget.cleanup()
    widget.close()


def _type_inputs(widget: MainWidget) -> None:
    _type_into(widget.baseline_distances, "1 3 6 10 20 40")
    _type_into(widget.baseline_putts, "1.0 1.2 1.5 1.7 1.9 2.2")
    _type_into(widget.rollouts, "30")
    _type_into(widget.aims, "0 2")
    _type_into(widget.paces, "1")


def test_window_launches_idle_and_ready(window: MainWidget) -> None:
    assert window.isVisible()
    assert window.windowTitle() == "Putting Analytics"
    assert window.run_button.isEnabled()
    assert not window.is_running
    assert window.table.rowCount() == 0
    assert window.mode_combo.count() == 3
    assert window.current_mode() == "optimize"


def test_clicking_run_with_empty_baseline_reports_the_problem(
    window: MainWidget,
) -> None:
    _click(window.run_button)
    assert not window.is_running
    assert window.status_label.text().startswith("Check the inputs")
    assert window.table.rowCount() == 0


def test_typed_inputs_reach_the_form_snapshot(window: MainWidget) -> None:
    _type_inputs(window)
    form = window.form_values()
    assert form.baseline_distances == "1 3 6 10 20 40"
    assert form.baseline_putts == "1.0 1.2 1.5 1.7 1.9 2.2"
    assert form.n == 30
    assert form.aims == "0 2"
    assert form.paces == "1"


def test_keyboard_step_selects_the_next_study(window: MainWidget) -> None:
    window.mode_combo.setFocus()
    QTest.keyClick(window.mode_combo, Qt.Key.Key_Down)
    assert window.current_mode() == "misread_cost"
    QTest.keyClick(window.mode_combo, Qt.Key.Key_Down)
    assert window.current_mode() == "library_study"


def test_clicking_run_shows_busy_then_fills_the_table(window: MainWidget) -> None:
    _type_inputs(window)
    _click(window.run_button)
    assert window.is_running
    assert not window.run_button.isEnabled()
    assert window.status_label.text().startswith("Running")
    assert _pump_until(lambda: not window.is_running), window.status_label.text()
    assert window.status_label.text().startswith("Done")
    assert window.run_button.isEnabled()
    assert window.table.rowCount() > 0
    first = window.table.item(0, 0)
    assert first is not None and first.text()


def test_clicking_run_on_the_misread_study_fills_the_table(
    window: MainWidget,
) -> None:
    _type_inputs(window)
    window.mode_combo.setFocus()
    QTest.keyClick(window.mode_combo, Qt.Key.Key_Down)
    _click(window.run_button)
    assert _pump_until(lambda: not window.is_running), window.status_label.text()
    assert window.status_label.text() == "Done: Cost of misreads."
    assert window.table.rowCount() > 0


def test_typed_out_of_range_aim_is_rejected_by_the_study(
    window: MainWidget,
) -> None:
    _type_inputs(window)
    _type_into(window.aims, "90")
    _click(window.run_button)
    assert _pump_until(lambda: not window.is_running), window.status_label.text()
    assert "rejected" in window.status_label.text()
    assert window.table.rowCount() == 0
    assert window.run_button.isEnabled()


def test_second_click_while_running_does_not_start_another_study(
    window: MainWidget, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts: list[Any] = []
    original = window._start_worker

    def counting(mode: str, args: Any) -> None:
        starts.append(mode)
        original(mode, args)

    monkeypatch.setattr(window, "_start_worker", counting)
    _type_inputs(window)
    _click(window.run_button)
    assert window.is_running
    window.run_study()  # a second request while busy is ignored
    assert starts == ["optimize"]
    assert _pump_until(lambda: not window.is_running)
