"""Mixin providing actuator management for ControlsTab."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6 import QtCore, QtWidgets

from src.shared.python.theme.style_constants import Styles

from ...control_system import ControlType
from ...sim_widget import MuJoCoSimWidget
from .actuator_detail_dialog import ActuatorDetailDialog


class _ActuatorControlsMixin:
    """Mixin: actuator group creation, callbacks, and filter logic."""

    # Attribute declarations for type checking (set by ControlsTab)
    sim_widget: MuJoCoSimWidget
    actuator_layout: QtWidgets.QVBoxLayout
    actuator_groups: list[QtWidgets.QGroupBox]
    actuator_control_widgets: list[QtWidgets.QWidget]
    actuator_sliders: list[QtWidgets.QSlider]
    actuator_labels: list[QtWidgets.QLabel]
    actuator_control_types: list[QtWidgets.QComboBox]
    actuator_constant_inputs: list[QtWidgets.QDoubleSpinBox]
    actuator_polynomial_coeffs: list[list[QtWidgets.QDoubleSpinBox]]
    actuator_damping_inputs: list[QtWidgets.QDoubleSpinBox]
    _simplified_notice: QtWidgets.QLabel | None
    simplified_actuator_mode: bool
    SIMPLIFIED_ACTUATOR_THRESHOLD: int

    def _clear_actuator_controls(self) -> None:
        """Remove all existing actuator control widgets."""
        self.actuator_sliders.clear()
        self.actuator_labels.clear()
        self.actuator_control_types.clear()
        self.actuator_constant_inputs.clear()
        self.actuator_polynomial_coeffs.clear()
        self.actuator_damping_inputs.clear()

        for widget in self.actuator_control_widgets:
            self.actuator_layout.removeWidget(widget)
            widget.deleteLater()
        self.actuator_control_widgets.clear()
        self.actuator_groups.clear()

        if self._simplified_notice:
            self.actuator_layout.removeWidget(self._simplified_notice)
            self._simplified_notice.deleteLater()
            self._simplified_notice = None

    def _create_actuator_controls(self, actuator_names: list[str]) -> None:
        if not (actuator_names is not None):
            raise ValueError("actuator_names must be provided")
        groups = self._group_actuators(actuator_names)
        actuator_index = 0
        total = len(actuator_names)

        self.simplified_actuator_mode = total >= self.SIMPLIFIED_ACTUATOR_THRESHOLD

        if self.simplified_actuator_mode:
            self._simplified_notice = QtWidgets.QLabel(
                "Large musculoskeletal model detected. Showing simplified "
                "actuator controls."
            )
            self._simplified_notice.setStyleSheet(Styles.NOTICE_WARNING)
            self.actuator_layout.addWidget(self._simplified_notice)

        for group_name, actuators in groups.items():
            group_box = QtWidgets.QGroupBox(f"{group_name} ({len(actuators)})")
            group_box.setCheckable(True)
            group_box.setChecked(True)
            group_box.setProperty("actuator_names", actuators)

            content = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(content)
            layout.setContentsMargins(0, 0, 0, 0)

            for act_name in actuators:
                if self.simplified_actuator_mode:
                    w = self._create_simplified_actuator_row(actuator_index, act_name)
                else:
                    w = self._create_advanced_actuator_control(actuator_index, act_name)
                self.actuator_control_widgets.append(w)
                layout.addWidget(w)
                actuator_index += 1

            group_box.toggled.connect(content.setVisible)
            gl = QtWidgets.QVBoxLayout(group_box)
            gl.addWidget(content)

            self.actuator_groups.append(group_box)
            self.actuator_layout.addWidget(group_box)

        self.actuator_layout.addStretch(1)

    def _group_actuators(self, names: list[str]) -> dict[str, list[str]]:
        if not (names is not None):
            raise ValueError("names must be provided")
        groups: dict[str, list[str]] = {}
        for name in names:
            if "Shoulder" in name:
                key = "Shoulder"
            elif "Elbow" in name or "Forearm" in name:
                key = "Arm/Elbow"
            elif "Wrist" in name:
                key = "Wrist"
            elif "Spine" in name:
                key = "Spine/Torso"
            elif "Leg" in name or "Knee" in name or "Ankle" in name:
                key = "Legs"
            elif "Scap" in name:
                key = "Scapula"
            elif "Muscle" in name:
                key = "Muscles"
            else:
                key = "Other"

            if key not in groups:
                groups[key] = []
            groups[key].append(name)
        return groups

    def _create_simplified_actuator_row(
        self, index: int, name: str
    ) -> QtWidgets.QWidget:
        if not (index is not None):
            raise ValueError("index must be provided")
        container = QtWidgets.QFrame()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(4, 2, 4, 2)

        layout.addWidget(QtWidgets.QLabel(f"<b>{name}</b>"), stretch=2)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(-100, 100)
        slider.valueChanged.connect(
            lambda v, i=index: self.on_actuator_slider_changed(i, v)
        )
        self.actuator_sliders.append(slider)
        layout.addWidget(slider, stretch=4)

        label = QtWidgets.QLabel("0 Nm")
        label.setMinimumWidth(60)
        self.actuator_labels.append(label)
        layout.addWidget(label)

        detail_btn = QtWidgets.QPushButton("Edit...")
        detail_btn.setFixedWidth(50)
        detail_btn.clicked.connect(
            lambda _, i=index, n=name, s=slider: self.open_actuator_detail_dialog(
                i, n, s
            )
        )
        layout.addWidget(detail_btn)
        return container

    def _create_advanced_actuator_control(
        self, index: int, name: str
    ) -> QtWidgets.QWidget:
        if not (index is not None):
            raise ValueError("index must be provided")
        container = QtWidgets.QFrame()
        container.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        layout = QtWidgets.QVBoxLayout(container)

        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(QtWidgets.QLabel(f"<b>{name}</b>"))

        combo = QtWidgets.QComboBox()
        combo.addItems(["Constant", "Polynomial", "Sine Wave", "Step"])
        combo.currentIndexChanged.connect(
            lambda idx, i=index: self.on_control_type_changed(i, idx)
        )
        self.actuator_control_types.append(combo)
        hl.addWidget(QtWidgets.QLabel("Type:"))
        hl.addWidget(combo)
        layout.addLayout(hl)

        ql = QtWidgets.QHBoxLayout()
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(-100, 100)
        slider.valueChanged.connect(
            lambda v, i=index: self.on_actuator_slider_changed(i, v)
        )
        self.actuator_sliders.append(slider)

        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1000, 1000)
        spin.valueChanged.connect(
            lambda v, i=index: self.on_constant_value_changed(i, v)
        )
        self.actuator_constant_inputs.append(spin)

        label = QtWidgets.QLabel("0 Nm")
        self.actuator_labels.append(label)

        ql.addWidget(QtWidgets.QLabel("Value:"))
        ql.addWidget(slider)
        ql.addWidget(spin)
        ql.addWidget(label)
        layout.addLayout(ql)

        dl = QtWidgets.QHBoxLayout()
        d_spin = QtWidgets.QDoubleSpinBox()
        d_spin.setRange(0, 100)
        d_spin.valueChanged.connect(lambda v, i=index: self.on_damping_changed(i, v))
        self.actuator_damping_inputs.append(d_spin)
        dl.addWidget(QtWidgets.QLabel("Damping:"))
        dl.addWidget(d_spin)

        detail_btn = QtWidgets.QPushButton("Params...")
        detail_btn.clicked.connect(
            lambda _, i=index, n=name, s=slider: self.open_actuator_detail_dialog(
                i, n, s
            )
        )
        dl.addWidget(detail_btn)
        layout.addLayout(dl)
        return container

    def open_actuator_detail_dialog(
        self,
        actuator_index: int,
        actuator_name: str,
        slider: QtWidgets.QSlider | None = None,
    ) -> None:
        """Open a dialog with comprehensive controls for an actuator."""
        if not (actuator_index is not None):
            raise ValueError("actuator_index must be provided")
        control_system = self.sim_widget.get_control_system()
        if control_system is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Error",
                "Control system not initialized.",  # type: ignore[arg-type]
            )
            return

        slider_sync: Callable[[float], None] | None = None
        if slider is not None:

            def slider_sync_func(value: float) -> None:
                slider.blockSignals(True)
                slider.setValue(int(value))
                slider.blockSignals(False)
                if actuator_index < len(self.actuator_labels):
                    self.actuator_labels[actuator_index].setText(f"{value:.0f} Nm")

            slider_sync = slider_sync_func

        dialog = ActuatorDetailDialog(
            control_system=control_system,
            actuator_index=actuator_index,
            actuator_name=actuator_name,
            slider_sync=slider_sync,
            parent=self,  # type: ignore[arg-type]
        )
        dialog.exec()

    def on_actuator_filter_changed(self, text: str) -> None:
        """Filter visible actuator groups by search text."""
        if not (text is not None):
            raise ValueError("text must be provided")
        text = text.lower()
        for group in self.actuator_groups:
            group_name = group.title().lower()
            actuators = group.property("actuator_names") or []
            match = (text in group_name) or any(text in a.lower() for a in actuators)
            group.setVisible(match)

    def on_actuator_slider_changed(self, index: int, value: int) -> None:
        """Apply slider value change to actuator constant torque."""
        if not (index is not None):
            raise ValueError("index must be provided")
        if index < len(self.actuator_labels):
            self.actuator_labels[index].setText(f"{value} Nm")
        if index < len(self.actuator_constant_inputs):
            s = self.actuator_constant_inputs[index]
            s.blockSignals(True)
            s.setValue(float(value))
            s.blockSignals(False)
        cs = self.sim_widget.get_control_system()
        if cs:
            cs.set_constant_value(index, float(value))
            cs.set_control_type(index, ControlType.CONSTANT)

    def on_constant_value_changed(self, index: int, value: float) -> None:
        """Apply spinbox value change to actuator constant torque."""
        if not (index is not None):
            raise ValueError("index must be provided")
        if index < len(self.actuator_sliders):
            s = self.actuator_sliders[index]
            s.blockSignals(True)
            s.setValue(int(value))
            s.blockSignals(False)
        cs = self.sim_widget.get_control_system()
        if cs:
            cs.set_constant_value(index, value)
            cs.set_control_type(index, ControlType.CONSTANT)

    def on_damping_changed(self, index: int, value: float) -> None:
        """Update damping coefficient for an actuator."""
        if not (index is not None):
            raise ValueError("index must be provided")
        cs = self.sim_widget.get_control_system()
        if cs:
            cs.set_damping(index, value)

    def on_control_type_changed(self, index: int, type_idx: int) -> None:
        """Switch the control type for an actuator."""
        if not (index is not None):
            raise ValueError("index must be provided")
        cs = self.sim_widget.get_control_system()
        if cs:
            types = [
                ControlType.CONSTANT,
                ControlType.POLYNOMIAL,
                ControlType.SINE_WAVE,
                ControlType.STEP,
            ]
            if type_idx < len(types):
                cs.set_control_type(index, types[type_idx])
