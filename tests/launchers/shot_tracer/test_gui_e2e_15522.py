"""Headless native Shot Tracer click-through tests (Refs #15522).

Only optional OpenGL rendering is disabled; Qt controls, signals, flight
models and comparison-table formatting are real. No pytest-qt is required.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtWidgets
    from PyQt6.QtTest import QTest
except (ImportError, OSError) as exc:
    pytest.skip(f"Native PyQt6 is unavailable: {exc}", allow_module_level=True)

from src.launchers import _shot_tracer_gui as gui
from src.launchers import shot_tracer

pytestmark = pytest.mark.unit


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch) -> Iterator[gui.MultiModelShotTracerWindow]:
    """Launch the production window in its supported no-OpenGL mode."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(shot_tracer, "PYQTGRAPH_AVAILABLE", False)
    monkeypatch.setattr(gui, "PYQTGRAPH_AVAILABLE", False)
    widget = gui.get_dockable_ui()
    widget.show()
    app.processEvents()
    assert widget.isVisible()
    try:
        yield widget
    finally:
        widget.close()
        widget.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()


def _click(button: QtWidgets.QAbstractButton) -> None:
    # Checkbox layouts stretch beyond their indicator/label hit region.
    position = (
        QtCore.QPoint(8, button.height() // 2)
        if isinstance(button, QtWidgets.QCheckBox)
        else button.rect().center()
    )
    QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton, pos=position)


def _edit(spin: QtWidgets.QDoubleSpinBox, value: str) -> None:
    spin.setFocus()
    spin.selectAll()
    QTest.keyClicks(spin, value)
    QTest.keyClick(spin, QtCore.Qt.Key.Key_Tab)
    assert spin.value() == pytest.approx(float(value))


def _assert_results(widget: gui.MultiModelShotTracerWidget, count: int) -> None:
    """Assert solver output and each visible table value agree in display units."""
    table = widget.results_table
    assert table.rowCount() == len(widget.results) == count
    for row, (name, result) in enumerate(widget.results.items()):
        assert table.item(row, 0).text() == name
        positions = result.to_position_array()
        assert positions.shape[0] > 2
        assert np.isfinite(positions).all()
        assert result.carry_distance > 0
        assert result.max_height > 0
        assert result.flight_time > 0
        expected = (
            f"{result.carry_distance * 1.09361:.1f}",
            f"{result.max_height:.1f}",
            f"{result.flight_time:.2f}",
            f"{result.landing_angle:.1f}",
        )
        assert tuple(table.item(row, col).text() for col in range(1, 5)) == expected


def test_launch_compare_all_models_and_clear(window) -> None:
    """A first launch computes real trajectories and Clear removes the results."""
    widget = window.central_widget
    assert widget.results_table.rowCount() == 0
    assert all(box.isChecked() for box in widget.model_checkboxes.values())
    _click(widget.simulate_btn)
    _assert_results(widget, len(gui.FlightModelType))
    _click(widget.clear_btn)
    assert not widget.results
    assert widget.results_table.rowCount() == 0
    assert widget.imported_list.count() == 0


def test_preset_edit_select_and_rerun(window) -> None:
    """Club presets and typed edits feed a selected model; reruns replace rows."""
    widget = window.central_widget
    preset = next(
        b for b in widget.findChildren(QtWidgets.QPushButton) if b.text() == "PW"
    )
    _click(preset)
    assert (
        widget.speed_spin.value(),
        widget.angle_spin.value(),
        widget.spin_spin.value(),
    ) == (94.0, 23.0, 9000.0)
    boxes = list(widget.model_checkboxes.values())
    for box in boxes[1:]:
        _click(box)
    assert sum(box.isChecked() for box in boxes) == 1
    _click(widget.simulate_btn)
    _assert_results(widget, 1)
    previous = next(iter(widget.results.values())).carry_distance
    _edit(widget.speed_spin, "120")
    _edit(widget.azimuth_spin, "10")
    _click(widget.simulate_btn)
    _assert_results(widget, 1)
    result = next(iter(widget.results.values()))
    assert result.carry_distance > previous
    assert result.to_position_array()[-1, 1] > 0


def test_no_model_warning_and_recovery(window) -> None:
    """The real modal warning can be dismissed, then a model runs normally."""
    widget = window.central_widget
    for box in widget.model_checkboxes.values():
        _click(box)
    observed = []

    def dismiss_warning() -> None:
        dialog = QtWidgets.QApplication.activeModalWidget()
        if isinstance(dialog, QtWidgets.QMessageBox):
            observed.append((dialog.windowTitle(), dialog.text()))
            _click(dialog.button(QtWidgets.QMessageBox.StandardButton.Ok))

    timer = QtCore.QTimer(window)
    timer.setInterval(10)
    timer.timeout.connect(dismiss_warning)
    timer.start()
    try:
        _click(widget.simulate_btn)
    finally:
        timer.stop()
    assert observed == [("No Models", "Please select at least one model.")]
    assert not widget.results
    assert widget.results_table.rowCount() == 0
    _click(next(iter(widget.model_checkboxes.values())))
    _click(widget.simulate_btn)
    _assert_results(widget, 1)
