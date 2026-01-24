import typing

import pytest
from double_pendulum_model.physics.double_pendulum import DoublePendulumState
from double_pendulum_model.physics.triple_pendulum import TriplePendulumState
from double_pendulum_model.ui.pendulum_pyqt_app import PendulumController
from PyQt6 import QtWidgets

from src.shared.python.gui_utils import get_qapp


@pytest.fixture(scope="module")
def app() -> QtWidgets.QApplication:
    app = get_qapp()
    return typing.cast(QtWidgets.QApplication, app)


def test_status_update_double(app: QtWidgets.QApplication) -> None:  # noqa: ARG001
    controller = PendulumController()

    # We need to manually check if status_label exists because we are in TDD
    if not hasattr(controller, "status_label"):
        pytest.fail("PendulumController does not have status_label")

    state = DoublePendulumState(theta1=1.0, theta2=2.0, omega1=0.5, omega2=0.6)
    controller.time = 1.235
    controller._update_status(state)

    text = controller.status_label.text()
    assert "Time: 1.235 s" in text
    assert "θ1:   1.000" in text
    assert "θ2:   2.000" in text
    assert "ω1:   0.500" in text


def test_status_update_triple(app: QtWidgets.QApplication) -> None:  # noqa: ARG001
    controller = PendulumController()

    if not hasattr(controller, "status_label"):
        pytest.fail("PendulumController does not have status_label")

    state = TriplePendulumState(
        theta1=1.0, theta2=2.0, theta3=3.0, omega1=0.1, omega2=0.2, omega3=0.3
    )
    controller.time = 5.6789
    controller._update_status(state)

    text = controller.status_label.text()
    assert "Time: 5.679 s" in text
    assert "θ3:   3.000" in text
