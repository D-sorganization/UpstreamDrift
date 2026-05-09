"""Accordion of per-joint sliders, grouped by body region.

Each canonical joint (member of :data:`REFERENCE_GOLFER_FIELDS`) gets a
:class:`QDoubleSpinBox` + :class:`QSlider` pair inside a
:class:`QGroupBox` for its body region.  The widget emits a single
``angle_edited(name, degrees)`` signal whenever any spinbox changes; the
GUI wires this to :meth:`EngineController.set_pose` after applying the
delta to the current :class:`CanonicalPose`.

Note on units: this widget always presents angles in **degrees**
internally (matching the canonical convention).  The
:class:`UnitsBadge` reports the active engine's native convention; if
the user toggles "show radians" the widget swaps the spinbox suffix
and re-scales the displayed value but keeps the underlying canonical
state in degrees.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PyQt6 import QtCore, QtWidgets

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)
from src.tools.pose_studio.core import JOINT_REGION_LAYOUT

# Reasonable default range for a 1-DOF revolute golfer joint.  Engine
# adapters can report tighter limits via their JointSlot; if so the
# tighter range overrides this default.
_DEFAULT_DEG_RANGE: tuple[float, float] = (-180.0, 180.0)


class JointPanel(QtWidgets.QScrollArea):
    """Accordion of per-region joint sliders.

    Signals
    -------
    angle_edited(str, float)
        Emitted whenever the user edits a joint angle.  First argument
        is the canonical joint name (a member of
        :data:`REFERENCE_GOLFER_FIELDS`); the second is the new value
        in degrees.
    """

    angle_edited = QtCore.pyqtSignal(str, float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._show_radians: bool = False
        self._spinboxes: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._sliders: dict[str, QtWidgets.QSlider] = {}
        self._build_ui()

    # ---- construction --------------------------------------------------

    def _build_ui(self) -> None:
        container = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(container)
        outer.setContentsMargins(4, 4, 4, 4)

        for region, joints in JOINT_REGION_LAYOUT.items():
            group = QtWidgets.QGroupBox(region)
            group.setCheckable(True)
            group.setChecked(True)
            group.setToolTip(f"Toggle to collapse/expand the {region} joint group.")
            grid = QtWidgets.QGridLayout(group)
            for row, joint in enumerate(joints):
                self._add_joint_row(grid, row, joint)
            outer.addWidget(group)

        outer.addStretch(1)
        self.setWidget(container)

    def _add_joint_row(
        self,
        grid: QtWidgets.QGridLayout,
        row: int,
        joint: str,
    ) -> None:
        label = QtWidgets.QLabel(joint)
        label.setToolTip(f"Canonical joint name: {joint}")
        label.setMinimumWidth(160)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(int(_DEFAULT_DEG_RANGE[0] * 10))
        slider.setMaximum(int(_DEFAULT_DEG_RANGE[1] * 10))
        slider.setValue(0)
        slider.setToolTip(
            f"Drag to set {joint} angle in the canonical degrees convention. "
            f"Range {_DEFAULT_DEG_RANGE[0]:.0f} to {_DEFAULT_DEG_RANGE[1]:.0f} deg."
        )
        slider.valueChanged.connect(
            lambda value, name=joint: self._on_slider_changed(name, value)
        )

        spin = QtWidgets.QDoubleSpinBox()
        spin.setMinimum(_DEFAULT_DEG_RANGE[0])
        spin.setMaximum(_DEFAULT_DEG_RANGE[1])
        spin.setDecimals(2)
        spin.setSingleStep(0.5)
        spin.setSuffix(" deg")
        spin.setMinimumWidth(96)
        spin.setToolTip(
            f"Type a precise value for {joint} (degrees in the canonical convention)."
        )
        spin.valueChanged.connect(
            lambda value, name=joint: self._on_spinbox_changed(name, value)
        )

        grid.addWidget(label, row, 0)
        grid.addWidget(slider, row, 1)
        grid.addWidget(spin, row, 2)

        self._sliders[joint] = slider
        self._spinboxes[joint] = spin

    # ---- public surface ------------------------------------------------

    def set_angles(self, angles_deg: Mapping[str, float]) -> None:
        """Update every spinbox + slider from *angles_deg* without
        re-emitting ``angle_edited``."""
        for name in REFERENCE_GOLFER_FIELDS:
            value = float(angles_deg.get(name, 0.0))
            spin = self._spinboxes[name]
            slider = self._sliders[name]
            spin_blocker = QtCore.QSignalBlocker(spin)
            slider_blocker = QtCore.QSignalBlocker(slider)
            spin.setValue(self._to_display_value(value))
            slider.setValue(int(value * 10))
            del spin_blocker
            del slider_blocker

    def set_show_radians(self, show_radians: bool) -> None:
        """Switch the spinbox suffix and value scaling between deg/rad.

        Sliders always tick in tenths of a degree; only the spinbox
        display unit changes.
        """
        if not isinstance(show_radians, bool):
            raise TypeError(
                f"show_radians must be bool, got {type(show_radians).__name__}"
            )
        self._show_radians = show_radians
        suffix = " rad" if show_radians else " deg"
        for name, spin in self._spinboxes.items():
            slider = self._sliders[name]
            current_deg = slider.value() / 10.0
            blocker = QtCore.QSignalBlocker(spin)
            spin.setSuffix(suffix)
            if show_radians:
                spin.setMinimum(float(np.radians(_DEFAULT_DEG_RANGE[0])))
                spin.setMaximum(float(np.radians(_DEFAULT_DEG_RANGE[1])))
                spin.setDecimals(4)
                spin.setSingleStep(0.01)
                spin.setValue(float(np.radians(current_deg)))
            else:
                spin.setMinimum(_DEFAULT_DEG_RANGE[0])
                spin.setMaximum(_DEFAULT_DEG_RANGE[1])
                spin.setDecimals(2)
                spin.setSingleStep(0.5)
                spin.setValue(current_deg)
            del blocker

    def joint_widgets(self) -> dict[str, QtWidgets.QWidget]:
        """Return a flat dict of every joint's spinbox + slider, keyed
        ``"<joint>__spin"`` / ``"<joint>__slider"``.

        Used by the help-coverage test to assert tooltips.
        """
        out: dict[str, QtWidgets.QWidget] = {}
        for name, spin in self._spinboxes.items():
            out[f"{name}__spin"] = spin
        for name, slider in self._sliders.items():
            out[f"{name}__slider"] = slider
        return out

    # ---- internals -----------------------------------------------------

    def _to_display_value(self, deg: float) -> float:
        return float(np.radians(deg)) if self._show_radians else deg

    def _from_display_value(self, displayed: float) -> float:
        return float(np.degrees(displayed)) if self._show_radians else displayed

    def _on_slider_changed(self, name: str, value: int) -> None:
        deg = value / 10.0
        spin = self._spinboxes[name]
        blocker = QtCore.QSignalBlocker(spin)
        spin.setValue(self._to_display_value(deg))
        del blocker
        self.angle_edited.emit(name, deg)

    def _on_spinbox_changed(self, name: str, displayed_value: float) -> None:
        deg = self._from_display_value(displayed_value)
        slider = self._sliders[name]
        blocker = QtCore.QSignalBlocker(slider)
        slider.setValue(int(deg * 10))
        del blocker
        self.angle_edited.emit(name, deg)


__all__ = ["JointPanel"]
