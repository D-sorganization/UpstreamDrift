"""Headless QTest end-to-end flows for the Matched Swing Results Browser (#15522).

Unlike ``test_matched_swing_browser_gui.py`` (which calls slots and setters
directly), these tests deliver real input events -- key presses into the search
box and combo, mouse clicks on table rows and buttons -- against the real
widget backed by the committed production ledger.  The only OS-facing piece
handled outside the code under test is the modal text dialog, which is closed by
clicking its real *Close* button from a zero-delay timer.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.unit

pytest.importorskip("PyQt6", reason="Matched Swing Browser requires PyQt6")

_DRIVER = "driver_g1_drake"


@pytest.fixture
def app():
    """Strong QApplication reference so the instance outlives every widget."""
    from PyQt6.QtWidgets import QApplication

    instance = QApplication.instance()
    held = instance if isinstance(instance, QApplication) else QApplication([])
    yield held
    held.processEvents()


@pytest.fixture
def window(app):
    try:
        from src.tools.matched_swing_browser.gui import MatchedSwingBrowserWindow
    except ImportError as exc:  # uninitialised vendor/ud-tools submodule in CI
        pytest.skip(f"pinned Tools tree (vendor/ud-tools) unavailable: {exc}")

    win = MatchedSwingBrowserWindow()
    if win.widget.total_runs_count == 0:
        win.close()
        pytest.skip("committed matched-swing ledger is not available in this checkout")
    win.resize(1280, 800)
    win.show()
    app.processEvents()
    yield win
    win.close()
    win.deleteLater()
    app.processEvents()


def _button(widget, text: str):
    from PyQt6.QtWidgets import QPushButton

    return next(b for b in widget.findChildren(QPushButton) if b.text() == text)


def _click(button, pos=None) -> None:
    """Deliver a real left mouse press/release (stubs mistype QTest statics)."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    if pos is None:
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)  # type: ignore[call-overload]
    else:
        QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=pos)  # type: ignore[call-overload]


def _key(widget, key) -> None:
    from PyQt6.QtTest import QTest

    QTest.keyClick(widget, key)  # type: ignore[call-overload]


def _click_row(widget, app, row: int) -> None:
    """Click the real table viewport at the centre of ``row``'s first cell."""

    table = widget.table
    rect = table.visualRect(table.model().index(row, 0))
    _click(table.viewport(), pos=rect.center())
    app.processEvents()


def _driver_row(widget) -> int:
    for r in range(widget.table.rowCount()):
        item = widget.table.item(r, 6)
        if item is not None and _DRIVER in item.text():
            return r
    pytest.skip(f"ledger has no {_DRIVER} run")


def _type(line_edit, text: str) -> None:
    from PyQt6.QtTest import QTest

    line_edit.setFocus()
    QTest.keyClicks(line_edit, text)  # type: ignore[call-arg,arg-type]


def test_launch_shows_ledger_and_autoselects_best_row(window, app) -> None:
    widget = window.widget
    assert window.isVisible()
    assert window.windowTitle() == "Matched Swing Results Browser"
    assert widget.table.rowCount() == widget.total_runs_count >= 40
    assert widget.table.currentRow() == 0
    assert widget.badge_label.text() not in ("", "—")
    assert widget.whole_rmse_val.text() != "—"


def test_typing_in_search_filters_then_reset_click_restores(window, app) -> None:
    widget = window.widget
    total = widget.table.rowCount()

    _type(widget.search_input, _DRIVER)
    app.processEvents()
    assert 1 <= widget.table.rowCount() < total
    for r in range(widget.table.rowCount()):
        assert _DRIVER in widget.table.item(r, 6).text()

    _click(widget.reset_button)
    app.processEvents()
    assert widget.search_input.text() == ""
    assert widget.table.rowCount() == total


def test_no_match_search_clears_details_and_disables_actions(window, app) -> None:
    widget = window.widget
    _type(widget.search_input, "zz-no-such-run-zz")
    app.processEvents()

    assert widget.table.rowCount() == 0
    assert widget.badge_label.text() == "—"
    assert widget.movie is None
    for label in ("Play", "Pause", "Restart", "Open in Tour Matching Viewer"):
        assert not _button(widget, label).isEnabled()

    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    widget.search_input.setFocus()
    for _ in range(len("zz-no-such-run-zz")):
        _key(widget.search_input, Qt.Key.Key_Backspace)
    app.processEvents()
    assert widget.table.rowCount() == widget.total_runs_count


def test_engine_combo_keyboard_selection_filters_rows(window, app) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    widget = window.widget
    total = widget.table.rowCount()
    combo = widget.engine_combo
    combo.setFocus()
    _key(combo, Qt.Key.Key_Down)
    app.processEvents()

    engine = combo.currentText()
    assert combo.currentIndex() == 1 and engine != "All"
    assert 0 < widget.table.rowCount() <= total
    for r in range(widget.table.rowCount()):
        assert widget.table.item(r, 0).text() == engine


def test_mouse_row_click_and_arrow_keys_update_details(window, app) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    widget = window.widget
    target = _driver_row(widget)
    _click_row(widget, app, target)
    assert widget.table.currentRow() == target
    assert "REJECTED" in widget.badge_label.text().upper()
    assert _DRIVER in widget._receipt_path_lbl.text()

    before = widget._receipt_path_lbl.text()
    widget.table.setFocus()
    _key(
        widget.table,
        Qt.Key.Key_Down if target + 1 < widget.table.rowCount() else Qt.Key.Key_Up,
    )
    app.processEvents()
    assert widget.table.currentRow() != target
    assert widget._receipt_path_lbl.text() != before


def test_playback_buttons_drive_the_gif_movie(window, app) -> None:
    from PyQt6.QtGui import QMovie

    widget = window.widget
    _click_row(widget, app, _driver_row(widget))
    movie = widget.movie
    assert movie is not None and movie.state() == QMovie.MovieState.Running


def test_one_frame_gif_keeps_running_past_its_frame_delay(window, app) -> None:
    """A still playback.gif must stay Running after its single frame delay."""
    from PyQt6.QtCore import QElapsedTimer
    from PyQt6.QtGui import QMovie

    widget = window.widget
    _click_row(widget, app, _driver_row(widget))
    movie = widget.movie
    assert movie is not None and movie.frameCount() == 1
    delay_ms = max(int(movie.nextFrameDelay()), 0)
    timer = QElapsedTimer()
    timer.start()
    while timer.elapsed() < delay_ms + 80:
        app.processEvents()
    assert movie.state() == QMovie.MovieState.Running
    _click(_button(widget, "Pause"))
    app.processEvents()
    assert movie.state() == QMovie.MovieState.Paused

    _click(_button(widget, "Pause"))
    assert movie.state() == QMovie.MovieState.Paused
    _click(_button(widget, "Play"))
    assert movie.state() == QMovie.MovieState.Running
    _click(_button(widget, "Restart"))
    assert widget.movie is movie and movie.state() == QMovie.MovieState.Running


def test_view_receipt_json_opens_modal_closed_by_real_click(window, app) -> None:
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QDialog, QPushButton, QTextEdit

    widget = window.widget
    _click_row(widget, app, _driver_row(widget))
    seen: dict[str, str] = {}

    def close_dialog() -> None:
        dialog = app.activeModalWidget()
        if not isinstance(dialog, QDialog):
            QTimer.singleShot(10, close_dialog)
            return
        seen["title"] = dialog.windowTitle()
        seen["text"] = dialog.findChild(QTextEdit).toPlainText()
        _click(next(b for b in dialog.findChildren(QPushButton) if b.text()))

    QTimer.singleShot(0, close_dialog)
    _click(_button(widget, "View Receipt JSON"))

    assert seen["title"].startswith("Receipt JSON:")
    assert seen["text"].lstrip().startswith("{")
    assert app.activeModalWidget() is None
