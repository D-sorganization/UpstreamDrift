import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDoubleSpinBox, QLabel, QSlider


class MockWindowFix:
    def __init__(self) -> None:
        self.actuator_constant_inputs = []
        self.actuator_sliders = []
        self.actuator_labels = []
        self.control_system_calls = 0

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(-100)
        self.slider.setMaximum(100)

        self.constant_input = QDoubleSpinBox()
        self.constant_input.setRange(-1000.0, 1000.0)
        self.constant_input.setSingleStep(1.0)
        self.constant_input.setDecimals(2)

        self.label = QLabel("0 Nm")

        self.actuator_sliders.append(self.slider)
        self.actuator_constant_inputs.append(self.constant_input)
        self.actuator_labels.append(self.label)

        actuator_index = 0
        self.slider.valueChanged.connect(
            lambda val, i=actuator_index: self.on_actuator_slider_changed(i, val)
        )
        self.constant_input.valueChanged.connect(
            lambda val, i=actuator_index: self.on_constant_value_changed(i, val)
        )

        self.slider_calls = 0
        self.spinbox_calls = 0

    def on_actuator_slider_changed(self, actuator_index: int, value: int) -> None:
        assert actuator_index is not None, "actuator_index must be provided"
        assert actuator_index >= 0, "actuator_index must be non-negative"
        assert isinstance(value, int), "value must be an int"
        self.slider_calls += 1
        print(f"Slider changed to {value}")

        if actuator_index < len(self.actuator_constant_inputs):
            spinbox = self.actuator_constant_inputs[actuator_index]
            # BLOCK SIGNALS FIX
            was_blocked = spinbox.blockSignals(True)
            spinbox.setValue(float(value))
            spinbox.blockSignals(was_blocked)

        self.control_system_calls += 1

    def on_constant_value_changed(self, actuator_index: int, value: float) -> None:
        assert actuator_index is not None, "actuator_index must be provided"
        assert actuator_index >= 0, "actuator_index must be non-negative"
        assert isinstance(value, float), "value must be a float"
        self.spinbox_calls += 1
        print(f"Spinbox changed to {value}")

        if actuator_index < len(self.actuator_sliders):
            slider = self.actuator_sliders[actuator_index]
            # BLOCK SIGNALS FIX
            was_blocked = slider.blockSignals(True)
            slider.setValue(int(value))
            slider.blockSignals(was_blocked)

        self.control_system_calls += 1


def run_test() -> None:
    _ = QApplication(sys.argv + ["-platform", "offscreen"])

    window = MockWindowFix()

    print("--- Test Fix: Setting Spinbox to 50.5 ---")

    constant_input = window.constant_input
    constant_input.setValue(50.5)

    final_spinbox_value = constant_input.value()
    slider = window.slider
    final_slider_value = slider.value()

    print(f"Final Spinbox Value: {final_spinbox_value}")
    print(f"Final Slider Value: {final_slider_value}")

    slider_calls = window.slider_calls
    spinbox_calls = window.spinbox_calls
    print(f"Total Callbacks: Slider={slider_calls}, Spinbox={spinbox_calls}")
    control_calls = window.control_system_calls
    print(f"Control System Updates: {control_calls}")

    if final_spinbox_value == 50.5:
        print("PASS: Precision kept.")
    else:
        print("FAIL: Precision lost!")

    if control_calls == 1:
        print("PASS: Single update.")
    else:
        print(f"FAIL: {control_calls} updates.")


if __name__ == "__main__":
    run_test()
