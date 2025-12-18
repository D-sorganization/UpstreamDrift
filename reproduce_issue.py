import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDoubleSpinBox, QLabel, QSlider


# Mocking the window class partially to reproduce the issue
class MockWindow:
    def __init__(self):
        self.actuator_constant_inputs = []
        self.actuator_sliders = []
        self.actuator_labels = []
        self.control_system_calls = 0

        # Create widgets
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(-100)
        self.slider.setMaximum(100)

        self.constant_input = QDoubleSpinBox()
        self.constant_input.setRange(-1000.0, 1000.0)
        self.constant_input.setSingleStep(1.0)
        self.constant_input.setDecimals(2)

        self.label = QLabel("0 Nm")

        # Add to lists as in the original code
        self.actuator_sliders.append(self.slider)
        self.actuator_constant_inputs.append(self.constant_input)
        self.actuator_labels.append(self.label)

        # Connect signals
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
        self.slider_calls += 1
        print(f"Slider changed to {value}")
        if actuator_index < len(self.actuator_constant_inputs):
            # This triggers on_constant_value_changed
            self.actuator_constant_inputs[actuator_index].setValue(float(value))

        # Simulate control system update
        self.control_system_calls += 1

    def on_constant_value_changed(self, actuator_index: int, value: float) -> None:
        self.spinbox_calls += 1
        print(f"Spinbox changed to {value}")
        # Update slider
        if actuator_index < len(self.actuator_sliders):
            # This triggers on_actuator_slider_changed
            self.actuator_sliders[actuator_index].setValue(int(value))

        # Simulate control system update
        self.control_system_calls += 1


def run_test():
    _ = QApplication(sys.argv + ["-platform", "offscreen"])

    window = MockWindow()

    print("--- Test 1: Setting Spinbox to 50.5 ---")
    # Setting spinbox to 50.5.
    # Expectation:
    # 1. Spinbox -> 50.5
    # 2. Slider -> 50 (int cast)
    # 3. Spinbox -> 50.0 (from slider)
    window.constant_input.setValue(50.5)

    final_spinbox_value = window.constant_input.value()
    final_slider_value = window.slider.value()

    print(f"Final Spinbox Value: {final_spinbox_value}")
    print(f"Final Slider Value: {final_slider_value}")
    print(
        f"Total Callbacks: Slider={window.slider_calls}, "
        f"Spinbox={window.spinbox_calls}"
    )
    print(f"Control System Updates: {window.control_system_calls}")

    if final_spinbox_value != 50.5:
        print("FAIL: Precision lost!")
    else:
        print("PASS: Precision kept.")

    if window.control_system_calls > 1:
        print("FAIL: Redundant updates detected!")


if __name__ == "__main__":
    run_test()
