"""GUI registration for the canonical Movement Optimizer app."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "Movement Optimizer",
    "tool_name": "movement_optimizer",
    "description": (
        "Optimize barbell biomechanics trajectories with Lagrangian dynamics, "
        "swingset, and chain models"
    ),
    "category": "Optimization",
    "icon": "chart-line",
    "pyqt6": {
        "module": "movement_optimizer.gui.main_window",
        "class": "MainWindow",
        "dependencies": ["PyQt6", "numpy", "scipy"],
        "settings_app": "MovementOptimizer",
    },
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
