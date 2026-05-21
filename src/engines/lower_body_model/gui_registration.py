"""GUI registration for the Lower Body Model tool."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "Lower Body Model",
    "tool_name": "lower_body_model",
    "description": "Simulate and inspect lower-body MuJoCo kinematics and controls",
    "category": "Biomechanics",
    "icon": "accessibility",
    "pyqt6": {
        "module": "src.engines.lower_body_model.launch_pyqt6",
        "class": "ControlPanel",
        "dependencies": ["PyQt6", "mujoco", "numpy"],
        "settings_app": "LowerBodyModel",
    },
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
