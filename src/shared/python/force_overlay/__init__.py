"""Engine-agnostic force and torque overlay pipeline (ADR-0052, #11285).

This package provides:
- Core force/torque contracts and schemas (contracts.py)
- Time-indexed immutable series with linear interpolation (series.py)

Headless import safe: this module and its core submodules never import GUI
or physics engine SDKs (PyQt6, matplotlib.pyplot, mujoco, pydrake,
pinocchio, opensim).
"""

from __future__ import annotations

from .contracts import (
    DEFAULT_OVERLAY_UNITS,
    ForceTorqueFrame,
    ForceTorqueProvider,
    OverlayWrench,
    WrenchKind,
    read_force_torque_frame,
)
from .conversions import (
    SegmentAxis,
    axial_loads_from_reactions,
    frame_with_axial_loads,
    joint_torque_wrench,
    move_wrench_point,
    world_wrench_from_local,
)
from .series import ForceTorqueSeries

__all__ = [
    "DEFAULT_OVERLAY_UNITS",
    "ForceTorqueFrame",
    "ForceTorqueProvider",
    "ForceTorqueSeries",
    "OverlayWrench",
    "SegmentAxis",
    "WrenchKind",
    "axial_loads_from_reactions",
    "frame_with_axial_loads",
    "joint_torque_wrench",
    "move_wrench_point",
    "read_force_torque_frame",
    "world_wrench_from_local",
]
