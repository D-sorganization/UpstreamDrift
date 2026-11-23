"""Headless QTest end-to-end tests for the swing-to-flight GUI (#15522).

``test_gui.py`` / ``test_async_actions.py`` call handlers directly or mock the
pipeline. These tests drive the real window the way a user does: QTest mouse
clicks on the club presets and "Run Full Pipeline", the real off-thread worker
and the real ``SwingBallFlightPipeline``, asserting on what is displayed.
"""

from __future__ import annotations

import os
import re
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, Qt, QThread  # noqa: E402
from PyQt6.QtGui import QStandardItemModel  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

from src.shared.python.physics.rust_kernel import is_rust_available  # noqa: E402
from src.shared.python.ui.units import UnitSystem  # noqa: E402
from src.tools.swing_flight_pipeline.gui import (  # noqa: E402
    SwingFlightWidget,
    SwingFlightWindow,
)

pytestmark = pytest.mark.unit

# The real flight simulator enforces Rust parity; CI builds the kernel.
requires_kernel = pytest.mark.skipif(
    not is_rust_available(), reason="upstream-physics Rust kernel not built"
)

# METRIC display reads "Carry: 230.0 m (251.5 yd)"; capture the metres value.
_CARRY = re.compile(r"Carry:\s+([\d.]+) m ")
_RUN = "Run Full Pipeline"


_APP: QApplication | None = None


def _ensure_qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # type: ignore[assignment]  # keep alive: GC'd app crashes Qt
    return app  # type: ignore[return-value]


def _pump_until(predicate, timeout_s: float = 30.0) -> bool:  # noqa: ANN001
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        QThread.msleep(5)
    QCoreApplication.processEvents()
    return predicate()


def _button(widget: SwingFlightWidget, text: str) -> QPushButton:
    for btn in widget.findChildren(QPushButton):
        if btn.text() == text:
            return btn
    raise AssertionError(f"no button labelled {text!r}")


def _click(widget: SwingFlightWidget, text: str) -> None:
    QTest.mouseClick(_button(widget, text), Qt.MouseButton.LeftButton)


def _run_and_read_carry(widget: SwingFlightWidget) -> float:
    """Click Run, wait for the worker, return the carry shown in the results."""
    _click(widget, _RUN)
    assert _pump_until(
        lambda: "Pipeline Complete" in widget._results_text.toPlainText()
    )
    assert _pump_until(lambda: not widget.action_bar.is_running)
    match = _CARRY.search(widget._results_text.toPlainText())
    assert match, widget._results_text.toPlainText()
    return float(match.group(1))


@pytest.fixture()
def widget():  # noqa: ANN201
    _ensure_qapp()
    w = SwingFlightWidget(unit_system=UnitSystem.METRIC)
    w.show()
    yield w
    w.cleanup()
    w.close()


def test_clicking_club_presets_stamps_speed_and_loft(widget: SwingFlightWidget) -> None:
    _click(widget, "7-Iron")
    assert widget._speed_spin.value() == pytest.approx(35.0)
    assert widget._loft_spin.value() == pytest.approx(34.0)

    _click(widget, "PW")
    assert widget._speed_spin.value() == pytest.approx(28.0)
    assert widget._loft_spin.value() == pytest.approx(46.0)

    _click(widget, "Driver")
    assert widget._speed_spin.value() == pytest.approx(50.0)
    assert widget._loft_spin.value() == pytest.approx(10.5)


def test_engine_combo_defaults_to_manual_and_disables_unavailable(
    widget: SwingFlightWidget,
) -> None:
    combo = widget._engine_combo
    assert combo.currentText() == "manual"
    model = combo.model()
    assert isinstance(model, QStandardItemModel)
    for index in range(combo.count()):
        provider = widget._providers[combo.itemText(index)]
        assert model.item(index).isEnabled() == provider.is_available()


def test_run_button_is_idle_with_placeholder_before_any_run(
    widget: SwingFlightWidget,
) -> None:
    assert _button(widget, _RUN).isEnabled()
    assert not widget.action_bar.is_running
    assert "click 'Run Full Pipeline'" in widget._results_text.toPlainText()


@requires_kernel
def test_run_click_executes_real_pipeline_and_renders_results(
    widget: SwingFlightWidget,
) -> None:
    _click(widget, "Driver")
    carry = _run_and_read_carry(widget)

    text = widget._results_text.toPlainText()
    assert "Engine: manual" in text
    assert "Flight Time:" in text
    assert 100.0 < carry < 450.0  # metres; 50 m/s driver swing, broad band
    assert _button(widget, _RUN).isEnabled()
    assert "Carry Distance:" in widget._carry_label.text()
    assert "Launch Speed:" in widget._launch_speed_label.text()


@requires_kernel
def test_changing_club_changes_displayed_carry(widget: SwingFlightWidget) -> None:
    _click(widget, "Driver")
    driver = _run_and_read_carry(widget)
    _click(widget, "PW")
    wedge = _run_and_read_carry(widget)
    assert wedge < driver * 0.75


@requires_kernel
def test_switching_units_after_a_run_rerenders_results(
    widget: SwingFlightWidget,
) -> None:
    _click(widget, "7-Iron")
    _run_and_read_carry(widget)
    metric_text = widget._results_text.toPlainText()

    widget.set_unit_system(UnitSystem.IMPERIAL)
    imperial_text = widget._results_text.toPlainText()

    assert "Pipeline Complete" in imperial_text
    assert imperial_text != metric_text
    # The spin box now shows mph, not m/s, for the same physical swing.
    assert widget._speed_spin.value() > 60.0


def test_standalone_window_launches_and_closes_cleanly() -> None:
    _ensure_qapp()
    window = SwingFlightWindow()
    window.show()
    QCoreApplication.processEvents()
    try:
        assert window.isVisible()
        assert window.windowTitle() == "Swing → Flight Pipeline"
        assert isinstance(window.centralWidget(), SwingFlightWidget)
        assert "run the pipeline" in window.statusBar().currentMessage()
    finally:
        window.close()
        QCoreApplication.processEvents()
    assert not window.isVisible()
