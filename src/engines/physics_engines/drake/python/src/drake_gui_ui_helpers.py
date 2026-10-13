"""Shared Drake GUI widget helpers for UISetupMixin and DrakeUIMixin."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.shared.python.engine_core.engine_availability import PYQT6_AVAILABLE

if PYQT6_AVAILABLE:
    from PyQt6 import QtWidgets
else:
    QtWidgets = None  # type: ignore[misc, assignment]

INDUCED_SOURCE_PRESET_ITEMS = ("gravity", "velocity", "total")


def set_drake_ui_initialized(host: Any, initialized: bool) -> None:
    """Track whether Drake UI widgets are safe for visualization callbacks."""
    host._drake_ui_initialized = initialized


def install_force_torque_visibility_toggles(
    host: Any,
    ft_grid: Any,
    on_visualization_changed: Callable[[], None],
) -> None:
    """Create force/torque/gravity checkboxes and add them to ``ft_grid``."""
    if QtWidgets is None:
        raise RuntimeError("PyQt6 is required for Drake GUI widgets")
    host.chk_show_forces = QtWidgets.QCheckBox("Show Forces")
    host.chk_show_forces.toggled.connect(on_visualization_changed)
    host.chk_show_torques = QtWidgets.QCheckBox("Show Torques")
    host.chk_show_torques.toggled.connect(on_visualization_changed)
    host.chk_show_gravity = QtWidgets.QCheckBox("Show Gravity")
    host.chk_show_gravity.toggled.connect(on_visualization_changed)
    ft_grid.addWidget(host.chk_show_forces, 0, 0)
    ft_grid.addWidget(host.chk_show_torques, 0, 1)
    ft_grid.addWidget(host.chk_show_gravity, 0, 2)


def repopulate_induced_source_combo(
    combo: Any,
    plant: Any,
    joint_index_type: Any,
) -> None:
    """Refresh induced-acceleration source names without spurious signal emissions."""
    current_text = combo.currentText()
    was_blocked = combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItems(list(INDUCED_SOURCE_PRESET_ITEMS))
        for i in range(plant.num_joints()):
            joint = plant.get_joint(joint_index_type(i))
            if joint.num_velocities() == 1:
                combo.addItem(joint.name())
        if current_text:
            combo.setCurrentText(current_text)
    finally:
        combo.blockSignals(was_blocked)
