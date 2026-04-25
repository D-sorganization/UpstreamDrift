"""Mixin providing kinematic joint control for ControlsTab."""

from __future__ import annotations

from typing import Any

from PyQt6 import QtCore, QtWidgets

from ...sim_widget import MuJoCoSimWidget


class _KinematicControlsMixin:
    """Mixin: kinematic joint sliders and spinboxes."""

    # Attribute declarations for type checking (set by ControlsTab)
    sim_widget: MuJoCoSimWidget
    joint_layout: QtWidgets.QVBoxLayout
    joint_widgets: dict[str, dict[str, QtWidgets.QWidget]]

    def _refresh_kinematic_controls(self) -> None:
        """Rebuild the kinematic joint controls."""
        while self.joint_layout.count():
            item = self.joint_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.joint_widgets = {}
        dof_info = self.sim_widget.get_dof_info()

        if not dof_info:
            self.joint_layout.addWidget(
                QtWidgets.QLabel("No controllable joints found.")
            )
            return

        for name, (min_val, max_val), current_val in dof_info:
            container = QtWidgets.QFrame()
            container.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            layout = QtWidgets.QVBoxLayout(container)

            header = QtWidgets.QHBoxLayout()
            header.addWidget(QtWidgets.QLabel(f"<b>{name}</b>"))
            val_label = QtWidgets.QLabel(f"{current_val:.3f}")
            header.addWidget(val_label, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
            layout.addLayout(header)

            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            steps = 1000
            slider.setRange(0, steps)

            range_span = max_val - min_val
            if range_span <= 0:
                range_span = 1.0

            norm_val = (current_val - min_val) / range_span
            slider_val = max(0, min(steps, int(norm_val * steps)))
            slider.setValue(slider_val)

            def _on_slider_change(
                v: int,
                n: str = name,
                mn: float = min_val,
                mx: float = max_val,
                lbl: Any = val_label,
            ) -> None:
                self._on_joint_slider_changed(n, v, mn, mx, lbl)

            slider.valueChanged.connect(_on_slider_change)
            layout.addWidget(slider)

            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(0.01)
            spin.setValue(current_val)

            def _on_spin_change(
                v: float,
                n: str = name,
                mn: float = min_val,
                mx: float = max_val,
                sl: Any = slider,
                lbl: Any = val_label,
            ) -> None:
                self._on_joint_spin_changed(n, v, mn, mx, sl, lbl)

            spin.valueChanged.connect(_on_spin_change)
            layout.addWidget(spin)

            self.joint_widgets[name] = {"slider": slider, "spin": spin}
            self.joint_layout.addWidget(container)

    def _on_joint_slider_changed(
        self,
        name: str,
        value_int: int,
        min_val: float,
        max_val: float,
        label: QtWidgets.QLabel,
    ) -> None:
        """Handle joint slider change."""
        if not (name is not None):
            raise ValueError("name must be provided")
        steps = 1000
        val = min_val + (value_int / steps) * (max_val - min_val)
        label.setText(f"{val:.3f}")
        self.sim_widget.set_joint_qpos(name, val)
        if hasattr(self, "joint_widgets") and name in self.joint_widgets:
            spin = self.joint_widgets[name]["spin"]
            if isinstance(spin, QtWidgets.QDoubleSpinBox):
                spin.blockSignals(True)
                spin.setValue(val)
                spin.blockSignals(False)

    def _on_joint_spin_changed(
        self,
        name: str,
        value: float,
        min_val: float,
        max_val: float,
        slider: QtWidgets.QSlider,
        label: QtWidgets.QLabel,
    ) -> None:
        """Handle joint spinbox change."""
        if not (name is not None):
            raise ValueError("name must be provided")
        self.sim_widget.set_joint_qpos(name, value)
        label.setText(f"{value:.3f}")

        steps = 1000
        range_span = max_val - min_val
        if range_span <= 0:
            range_span = 1.0

        norm_val = (value - min_val) / range_span
        slider_val = max(0, min(steps, int(norm_val * steps)))

        if isinstance(slider, QtWidgets.QSlider):
            slider.blockSignals(True)
            slider.setValue(slider_val)
            slider.blockSignals(False)
