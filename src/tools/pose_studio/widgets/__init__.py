"""PyQt6 widgets composed by the Pose Studio main window.

Each widget is a thin Qt shell over the pure-data controllers in
:mod:`src.tools.pose_studio.controllers`.  Importing this package
requires PyQt6.
"""

from __future__ import annotations

from src.tools.pose_studio.widgets.engine_picker import EnginePicker
from src.tools.pose_studio.widgets.joint_panel import JointPanel
from src.tools.pose_studio.widgets.units_badge import UnitsBadge
from src.tools.pose_studio.widgets.view_3d import View3D

__all__ = [
    "EnginePicker",
    "JointPanel",
    "UnitsBadge",
    "View3D",
]
