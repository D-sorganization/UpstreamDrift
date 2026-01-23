import pytest
from double_pendulum_model.ui.pendulum_pyqt_app import PendulumController
from PyQt6.QtWidgets import QApplication


def test_smoke() -> None:
    qapp = QApplication.instance()
    if not qapp:
        qapp = QApplication(["test", "-platform", "offscreen"])

    try:
        window = PendulumController()
        assert window is not None
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Could not instantiate PendulumController: {e}")
