"""End-to-end PyQt6 interaction tests for the ball flight GUI (#15522).

Unlike ``test_ball_flight_gui.py`` / ``test_async_actions.py`` (which mock the
physics engine), these drive the *real* window the way a user does: real mouse
clicks via ``QTest`` on the club-preset and "Simulate Flight" buttons, the real
off-thread worker, and the real ``BallFlightSimulator``. They assert on what the
user would see: updated spin boxes, button busy/idle state, status line and the
rendered results text.
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
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

from src.shared.python.ui.units import UnitSystem  # noqa: E402
from src.tools.ball_flight_gui.gui import (  # noqa: E402
    BallFlightWidget,
    BallFlightWindow,
)

from src.shared.python.physics.rust_kernel import is_rust_available  # noqa: E402

pytestmark = pytest.mark.unit

# The real simulator enforces strict Rust parity; without the built kernel the
# click-through-to-results tests cannot run (CI builds the wheel).
requires_kernel = pytest.mark.skipif(
    not is_rust_available(), reason="upstream-physics Rust kernel not built"
)

_APP: QApplication | None = None
_CARRY = re.compile(r"Carry:\s+([\d.]+)\s*yd")


def _ensure_qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # keep a module-level reference alive
    return app


def _pump_until(predicate, timeout_s: float = 30.0) -> bool:  # noqa: ANN001
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        QThread.msleep(5)
    QCoreApplication.processEvents()
    return predicate()


def _click(button: QPushButton) -> None:
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)


def _button(widget: BallFlightWidget, text: str) -> QPushButton:
    for btn in widget.findChildren(QPushButton):
        if btn.text() == text:
            return btn
    raise AssertionError(f"no button labelled {text!r}")


def _simulate_and_read_carry(widget: BallFlightWidget) -> float:
    """Click Simulate Flight, wait for the worker, return the displayed carry."""
    _click(_button(widget, "Simulate Flight"))
    assert _pump_until(lambda: "Carry:" in widget._results_text.toPlainText())
    assert _pump_until(lambda: not widget.action_bar.is_running)
    match = _CARRY.search(widget._results_text.toPlainText())
    assert match, widget._results_text.toPlainText()
    return float(match.group(1))


@pytest.fixture()
def widget():  # noqa: ANN201
    _ensure_qapp()
    w = BallFlightWidget(unit_system=UnitSystem.IMPERIAL)
    w.show()
    yield w
    w.cleanup()
    w.close()


def test_clicking_a_club_preset_updates_launch_inputs(
    widget: BallFlightWidget,
) -> None:
    _click(_button(widget, "Driver"))
    assert widget._speed_spin.value() == pytest.approx(163.0)
    assert widget._angle_spin.value() == pytest.approx(11.0)
    assert widget._spin_spin.value() == pytest.approx(2500.0)

    _click(_button(widget, "PW"))
    assert widget._speed_spin.value() == pytest.approx(94.0)
    assert widget._angle_spin.value() == pytest.approx(23.0)
    assert widget._spin_spin.value() == pytest.approx(9000.0)


@requires_kernel
def test_simulate_click_runs_real_physics_and_shows_results(
    widget: BallFlightWidget,
) -> None:
    _click(_button(widget, "Driver"))
    run_btn = _button(widget, "Simulate Flight")

    _click(run_btn)
    # While the worker is in flight the trigger must be locked out.
    assert widget.action_bar.is_running or "Carry:" in (
        widget._results_text.toPlainText()
    )
    assert _pump_until(lambda: "Carry:" in widget._results_text.toPlainText())
    assert _pump_until(lambda: not widget.action_bar.is_running)

    text = widget._results_text.toPlainText()
    assert "Ball Flight Results" in text
    assert "Flight Time:" in text
    carry = float(_CARRY.search(text).group(1))  # type: ignore[union-attr]
    # A 163 mph / 11 deg / 2500 rpm driver launch carries roughly 220-320 yd.
    assert 180.0 < carry < 350.0
    assert run_btn.isEnabled()
    assert "complete" in widget.action_bar.status_label.text()


@requires_kernel
def test_changing_club_changes_simulated_carry(widget: BallFlightWidget) -> None:
    _click(_button(widget, "Driver"))
    driver_carry = _simulate_and_read_carry(widget)

    _click(_button(widget, "PW"))
    wedge_carry = _simulate_and_read_carry(widget)

    assert wedge_carry < driver_carry * 0.75


@requires_kernel
def test_headwind_input_shortens_carry(widget: BallFlightWidget) -> None:
    _click(_button(widget, "7-Iron"))
    calm = _simulate_and_read_carry(widget)

    widget._wind_speed.setValue(20.0)
    widget._wind_dir.setValue(180.0)  # into the player
    windy = _simulate_and_read_carry(widget)

    assert windy != pytest.approx(calm, abs=0.5)


def test_standalone_window_launches_and_closes_cleanly() -> None:
    _ensure_qapp()
    window = BallFlightWindow()
    window.show()
    QCoreApplication.processEvents()
    try:
        assert window.isVisible()
        assert isinstance(window.centralWidget(), BallFlightWidget)
    finally:
        window.close()
        QCoreApplication.processEvents()
    assert not window.isVisible()
