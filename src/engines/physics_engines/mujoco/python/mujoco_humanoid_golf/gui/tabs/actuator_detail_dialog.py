"""ActuatorDetailDialog: on-demand editor for a single actuator's parameters."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PyQt6 import QtWidgets

from src.shared.python.signal_toolkit.polynomial_generator import (
    PolynomialGeneratorWidget,
)

from ...control_system import ControlSystem, ControlType


class ActuatorDetailDialog(QtWidgets.QDialog):
    """On-demand editor for actuator control parameters."""

    CONTROL_TYPE_LABELS = [
        "Constant",
        "Polynomial (6th order)",
        "Sine Wave",
        "Step Function",
    ]

    def __init__(
        self,
        *,
        control_system: ControlSystem,
        actuator_index: int,
        actuator_name: str,
        slider_sync: Callable[[float], None] | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Build the detail dialog for a single actuator."""
        super().__init__(parent)
        self.control_system = control_system
        self.actuator_index = actuator_index
        self.slider_sync = slider_sync
        self.setWindowTitle(f"Actuator Detail \u2014 {actuator_name}")
        self.setModal(True)
        self.resize(500, 540)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self.control = self.control_system.get_actuator_control(actuator_index)

        self._create_control_type_section(layout)
        self._create_constant_damping_section(layout)
        self._create_polynomial_section(layout)
        self._create_sine_wave_section(layout)
        self._create_step_function_section(layout)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close,
        )
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _create_control_type_section(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Create the control-type selector."""
        type_group = QtWidgets.QGroupBox("Control Type")
        type_layout = QtWidgets.QHBoxLayout(type_group)
        self.control_type_combo = QtWidgets.QComboBox()
        self.control_type_combo.addItems(self.CONTROL_TYPE_LABELS)
        current_type = self.control_system.get_control_type(self.actuator_index)
        type_map = {
            ControlType.CONSTANT: 0,
            ControlType.POLYNOMIAL: 1,
            ControlType.SINE_WAVE: 2,
            ControlType.STEP: 3,
        }
        self.control_type_combo.setCurrentIndex(type_map.get(current_type, 0))
        self.control_type_combo.currentIndexChanged.connect(
            self._on_control_type_changed
        )
        type_layout.addWidget(self.control_type_combo)
        layout.addWidget(type_group)

    def _create_constant_damping_section(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Create the constant value + damping inputs."""
        cd_group = QtWidgets.QGroupBox("Constant / Damping")
        cd_layout = QtWidgets.QFormLayout(cd_group)

        self.constant_spin = QtWidgets.QDoubleSpinBox()
        self.constant_spin.setRange(-1000, 1000)
        self.constant_spin.setSingleStep(1.0)
        const_val = self.control_system.get_constant_value(self.actuator_index)
        self.constant_spin.setValue(const_val if const_val is not None else 0.0)
        self.constant_spin.valueChanged.connect(self._on_constant_changed)
        cd_layout.addRow("Constant Value (Nm):", self.constant_spin)

        self.damping_spin = QtWidgets.QDoubleSpinBox()
        self.damping_spin.setRange(0, 100)
        damping_val = self.control_system.get_damping(self.actuator_index)
        self.damping_spin.setValue(damping_val if damping_val is not None else 0.0)
        self.damping_spin.valueChanged.connect(self._on_damping_changed)
        cd_layout.addRow("Damping:", self.damping_spin)

        layout.addWidget(cd_group)

    def _create_polynomial_section(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Embed the PolynomialGeneratorWidget for polynomial control."""
        self.poly_widget = PolynomialGeneratorWidget()
        self.poly_widget.polynomial_generated.connect(self._on_polynomial_generated)
        layout.addWidget(self.poly_widget)

    def _create_sine_wave_section(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Create amplitude, frequency, and phase inputs for sine-wave control."""
        sine_group = QtWidgets.QGroupBox("Sine Wave Parameters")
        sine_layout = QtWidgets.QFormLayout(sine_group)

        sine_params = self.control_system.get_sine_params(self.actuator_index)

        self.amplitude_spin = QtWidgets.QDoubleSpinBox()
        self.amplitude_spin.setRange(0, 1000)
        self.amplitude_spin.setValue(
            sine_params.get("amplitude", 0.0) if sine_params else 0.0
        )
        self.amplitude_spin.valueChanged.connect(
            lambda v: self._update_sine_params("amplitude", v)
        )
        sine_layout.addRow("Amplitude (Nm):", self.amplitude_spin)

        self.frequency_spin = QtWidgets.QDoubleSpinBox()
        self.frequency_spin.setRange(0.01, 100)
        self.frequency_spin.setSingleStep(0.1)
        self.frequency_spin.setValue(
            sine_params.get("frequency", 1.0) if sine_params else 1.0
        )
        self.frequency_spin.valueChanged.connect(
            lambda v: self._update_sine_params("frequency", v)
        )
        sine_layout.addRow("Frequency (Hz):", self.frequency_spin)

        self.phase_spin = QtWidgets.QDoubleSpinBox()
        self.phase_spin.setRange(-360, 360)
        self.phase_spin.setValue(sine_params.get("phase", 0.0) if sine_params else 0.0)
        self.phase_spin.valueChanged.connect(
            lambda v: self._update_sine_params("phase", v)
        )
        sine_layout.addRow("Phase (deg):", self.phase_spin)

        layout.addWidget(sine_group)

    def _create_step_function_section(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Create on-value, off-value, and toggle-time inputs for step control."""
        step_group = QtWidgets.QGroupBox("Step Function Parameters")
        step_layout = QtWidgets.QFormLayout(step_group)

        step_params = self.control_system.get_step_params(self.actuator_index)

        self.step_on_spin = QtWidgets.QDoubleSpinBox()
        self.step_on_spin.setRange(-1000, 1000)
        self.step_on_spin.setValue(
            step_params.get("on_value", 0.0) if step_params else 0.0
        )
        self.step_on_spin.valueChanged.connect(
            lambda v: self._update_step_params("on_value", v)
        )
        step_layout.addRow("On Value (Nm):", self.step_on_spin)

        self.step_off_spin = QtWidgets.QDoubleSpinBox()
        self.step_off_spin.setRange(-1000, 1000)
        self.step_off_spin.setValue(
            step_params.get("off_value", 0.0) if step_params else 0.0
        )
        self.step_off_spin.valueChanged.connect(
            lambda v: self._update_step_params("off_value", v)
        )
        step_layout.addRow("Off Value (Nm):", self.step_off_spin)

        self.step_time_spin = QtWidgets.QDoubleSpinBox()
        self.step_time_spin.setRange(0, 100)
        self.step_time_spin.setSingleStep(0.1)
        self.step_time_spin.setValue(
            step_params.get("toggle_time", 1.0) if step_params else 1.0
        )
        self.step_time_spin.valueChanged.connect(
            lambda v: self._update_step_params("toggle_time", v)
        )
        step_layout.addRow("Toggle Time (s):", self.step_time_spin)

        layout.addWidget(step_group)

    # ---- Internal slots ---------------------------------------------------

    def _on_control_type_changed(self, index: int) -> None:
        types = [
            ControlType.CONSTANT,
            ControlType.POLYNOMIAL,
            ControlType.SINE_WAVE,
            ControlType.STEP,
        ]
        if index < len(types):
            self.control_system.set_control_type(self.actuator_index, types[index])

    def _on_constant_changed(self, value: float) -> None:
        self.control_system.set_constant_value(self.actuator_index, value)
        if self.slider_sync is not None:
            self.slider_sync(value)

    def _on_damping_changed(self, value: float) -> None:
        self.control_system.set_damping(self.actuator_index, value)

    def _on_polynomial_generated(self, _joint_name: str, coefficients: list) -> None:
        self.control_system.set_polynomial_coeffs(
            self.actuator_index,
            np.asarray(coefficients, dtype=np.float64),
        )
        self.control_system.set_control_type(
            self.actuator_index,
            ControlType.POLYNOMIAL,
        )

    def _update_sine_params(self, key: str, value: float) -> None:
        params = self.control_system.get_sine_params(self.actuator_index) or {}
        params[key] = value
        self.control_system.set_sine_wave_params(
            self.actuator_index,
            params.get("amplitude", 0.0),
            params.get("frequency", 1.0),
            params.get("phase", 0.0),
        )

    def _update_step_params(self, key: str, value: float) -> None:
        params = self.control_system.get_step_params(self.actuator_index) or {}
        params[key] = value
        self.control_system.set_step_params(
            self.actuator_index,
            params.get("toggle_time", 0.0),
            params.get("on_value", 0.0),
        )
