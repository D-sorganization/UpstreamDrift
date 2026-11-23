"""Native Qt input drives the engine-level Driven Pendulum Explorer (Refs #15522).

Real ``QTest`` clicks and key presses operate the real ``PendulumController``,
its ``QTimer`` and the real double/triple pendulum dynamics. Nothing in the
widget, the timer or the physics is mocked; only the Qt platform is offscreen.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from double_pendulum_model.ui.pendulum_pyqt_app import PendulumController
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QAbstractButton, QApplication, QCheckBox, QLineEdit
except (ImportError, OSError) as exc:
    pytest.skip(f"PyQt6 pendulum stack not loadable: {exc}", allow_module_level=True)

pytestmark = pytest.mark.unit

_SHOULDER_TORQUE = "Shoulder torque (N·m)"
_SHOULDER_OMEGA = "Shoulder ω polynomial"
_STEPS_TIMEOUT_MS = 5000


@pytest.fixture
def application() -> QApplication:
    """Keep QTest usable when pytest-qt is disabled by the diff runner."""
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    return app


@pytest.fixture
def window(application: QApplication) -> Iterator[PendulumController]:
    """Show a real controller and always stop its timer on teardown."""
    controller = PendulumController()
    controller.show()
    QTest.qWait(10)
    assert controller.isVisible()
    try:
        yield controller
    finally:
        controller.timer.stop()
        controller.close()


def _click(button: QAbstractButton) -> None:
    # Aim at the indicator: a check box's centre can be empty row space.
    spot = (
        QPoint(6, button.height() // 2) if isinstance(button, QCheckBox) else QPoint()
    )
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=spot)


def _retype(entry: QLineEdit, text: str) -> None:
    """Replace the field contents with real key events."""
    entry.setFocus()
    entry.selectAll()
    QTest.keyClick(entry, Qt.Key.Key_Delete)
    QTest.keyClicks(entry, text)


def _run_until(window: PendulumController, seconds: float) -> None:
    """Pump the real event loop until the QTimer has advanced simulated time."""
    waited = 0
    while window.time < seconds and waited < _STEPS_TIMEOUT_MS:
        QTest.qWait(10)
        waited += 10
    assert window.time >= seconds


def test_start_runs_real_dynamics_and_pause_freezes_state(
    window: PendulumController,
) -> None:
    """Start swings the double pendulum under gravity; Pause stops the clock."""
    assert not window.timer.isActive()
    assert "Time: 0.000" in window.status_label.text()
    initial_theta1 = window.state_double.theta1

    _click(window.start_button)
    assert window.timer.isActive()
    _run_until(window, 0.05)
    _click(window.stop_button)

    assert not window.timer.isActive()
    assert window.state_double.theta1 != pytest.approx(initial_theta1, abs=1e-9)
    frozen_time, frozen_theta1 = window.time, window.state_double.theta1
    QTest.qWait(60)
    assert window.time == frozen_time
    assert window.state_double.theta1 == frozen_theta1
    assert f"Time: {frozen_time:.3f} s" in window.status_label.text()


def test_gravity_checkbox_gates_motion_from_rest_with_zero_torque(
    window: PendulumController,
) -> None:
    """Zero torque and zero velocity only move when the gravity box is ticked."""
    window.state_double.omega1 = window.state_double.omega2 = 0.0
    _click(window.gravity_checkbox)
    assert not window.gravity_checkbox.isChecked()
    _click(window.start_button)
    _run_until(window, 0.05)
    _click(window.stop_button)
    assert window.state_double.omega1 == pytest.approx(0.0, abs=1e-9)
    assert window.state_double.omega2 == pytest.approx(0.0, abs=1e-9)

    _click(window.gravity_checkbox)
    assert window.gravity_checkbox.isChecked()
    _click(window.start_button)
    _run_until(window, 0.05)
    _click(window.stop_button)
    assert abs(window.state_double.omega1) > 1e-6


def test_reset_restores_initial_state_after_running(
    window: PendulumController,
) -> None:
    """Reset stops the timer, zeroes the clock and redraws the start pose."""
    _click(window.start_button)
    _run_until(window, 0.05)
    _click(window.reset_button)

    assert not window.timer.isActive()
    assert window.time == 0.0
    assert window.state_double.theta1 == pytest.approx(-0.5)
    assert window.state_double.theta2 == pytest.approx(-1.2)
    assert window.state_double.omega1 == 0.0
    assert "Time: 0.000" in window.status_label.text()


def test_model_selector_switches_status_to_triple_and_steps_it(
    window: PendulumController,
) -> None:
    """Choosing Triple via the combo adds the third joint to status and motion."""
    assert "θ3" not in window.status_label.text()
    window.model_selector.setFocus()
    QTest.keyClick(window.model_selector, Qt.Key.Key_Down)
    assert window.model_selector.currentText() == "Triple"

    _click(window.reset_button)
    assert "θ3" in window.status_label.text()
    initial_theta3 = window.state_triple.theta3
    _click(window.start_button)
    _run_until(window, 0.05)
    _click(window.stop_button)
    assert window.state_triple.theta3 != pytest.approx(initial_theta3, abs=1e-9)
    assert "θ3" in window.status_label.text()


def test_typed_torque_expression_is_validated_and_drives_the_shoulder(
    window: PendulumController,
) -> None:
    """Typing flags bad input, then a valid torque accelerates the shoulder."""
    entry = window.torque_inputs[_SHOULDER_TORQUE]
    _retype(entry, "import os")
    assert entry.styleSheet() != ""
    assert entry.toolTip() != ""

    _retype(entry, "5")
    assert entry.styleSheet() == ""
    assert entry.toolTip() == ""
    assert window._current_config().torque_expressions[0] == "5"

    _click(window.gravity_checkbox)
    _click(window.start_button)
    _run_until(window, 0.05)
    _click(window.stop_button)
    assert window.state_double.omega1 > 1e-6


def test_inverse_mode_applies_the_typed_velocity_polynomial(
    window: PendulumController,
) -> None:
    """Switching to inverse dynamics makes the shoulder follow the polynomial."""
    window.mode_selector.setFocus()
    QTest.keyClick(window.mode_selector, Qt.Key.Key_Down)
    assert window.mode_selector.currentIndex() == 1
    assert not window._current_config().forward_mode

    _retype(window.velocity_inputs[_SHOULDER_OMEGA], "2")
    _click(window.start_button)
    _run_until(window, 0.05)
    _click(window.stop_button)
    assert window.state_double.omega1 == pytest.approx(2.0, abs=5e-3)
