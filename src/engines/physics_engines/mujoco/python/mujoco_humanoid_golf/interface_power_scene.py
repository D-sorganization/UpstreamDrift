"""MuJoCo scene markers for Robertson-Winter interface transfer power (#15399)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.engines.physics_engines.mujoco.python.visual_layer import add_scene_marker
from src.shared.python.biomechanics.robertson_winter_interface_power import (
    InterfacePowerFrame,
)
from src.shared.python.body_part_viz.segment_net_power import interface_edge_label

__all__ = ["InterfacePowerDrawReceipt", "add_interface_power_glyphs_to_scene"]


@dataclass(frozen=True)
class InterfacePowerDrawReceipt:
    markers_drawn: int
    labels: tuple[str, ...]


def _joint_anchor(model: Any, data: Any, joint_index: int) -> np.ndarray:
    import mujoco

    joint = model.joint(joint_index)
    body = int(joint.bodyid[0])
    if body <= 0:
        return np.asarray(data.xpos[0], dtype=float)
    return np.asarray(data.xpos[body], dtype=float)


def add_interface_power_glyphs_to_scene(
    model: Any,
    data: Any,
    scene: Any,
    frame: InterfacePowerFrame | None,
) -> InterfacePowerDrawReceipt:
    """Short edge markers at parent/child joints; labels carry transfer wording."""
    if frame is None:
        return InterfacePowerDrawReceipt(0, ())
    labels: list[str] = []
    drawn = 0
    for joint in frame.joints:
        if abs(joint.power_on_child_w) <= 1e-6:
            continue
        pos = _joint_anchor(model, data, joint.joint_index)
        sign = 1.0 if joint.power_on_child_w >= 0.0 else -1.0
        rgba = (0.55, 0.2, 0.75, 0.9) if sign > 0 else (0.85, 0.45, 0.15, 0.9)
        radius = min(0.05, 0.02 + 0.00002 * abs(joint.power_on_child_w))
        add_scene_marker(scene, pos, radius, rgba)
        labels.append(interface_edge_label(joint))
        drawn += 1
    return InterfacePowerDrawReceipt(drawn, tuple(labels))