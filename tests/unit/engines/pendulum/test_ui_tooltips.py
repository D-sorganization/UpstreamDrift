from double_pendulum_model.ui.pendulum_pyqt_app import PendulumController
from PyQt6.QtWidgets import QApplication


def test_tooltips() -> None:
    qapp = QApplication.instance()
    if not qapp:
        qapp = QApplication(["test", "-platform", "offscreen"])

    window = PendulumController()

    # Check button tooltips
    assert window.start_button.toolTip() == "Start the simulation"
    assert window.stop_button.toolTip() == "Pause the simulation"
    assert window.reset_button.toolTip() == "Reset simulation to initial state"

    # Check input tooltips
    for entry in window.torque_inputs.values():
        assert "Constant value" in entry.toolTip()

    for entry in window.velocity_inputs.values():
        assert "Polynomial coefficients" in entry.toolTip()
