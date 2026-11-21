"""Anchor positions for angular-momentum arrows (group / body COM)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .mujoco_segment_angular_momentum import whole_body_com_kinematics
from .segment_angular_momentum import SegmentGroupCatalog

if TYPE_CHECKING:
    import mujoco


def body_com_positions(
    model: mujoco.MjModel, data: mujoco.MjData
) -> dict[str, tuple[float, float, float]]:
    """World-frame COM per named body (excludes world and unnamed bodies)."""
    out: dict[str, tuple[float, float, float]] = {}
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name
        if not name:
            continue
        pos = np.asarray(data.xipos[body_id], dtype=float)
        out[name] = (float(pos[0]), float(pos[1]), float(pos[2]))
    return out


def _mass_weighted_com(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    members: frozenset[str],
) -> tuple[float, float, float] | None:
    total_mass = 0.0
    weighted = np.zeros(3, dtype=float)
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name
        if name not in members:
            continue
        mass = float(model.body(body_id).mass[0])
        if mass <= 0.0:
            continue
        pos = np.asarray(data.xipos[body_id], dtype=float)
        weighted += mass * pos
        total_mass += mass
    if total_mass <= 0.0:
        return None
    com = weighted / total_mass
    return (float(com[0]), float(com[1]), float(com[2]))


def angular_momentum_anchors(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    catalog: SegmentGroupCatalog,
    *,
    root_body_id: int,
) -> dict[str, tuple[float, float, float]]:
    """Anchors for whole-body, five groups, and every named body."""
    anchors = body_com_positions(model, data)
    com, _ = whole_body_com_kinematics(
        model, data, np.asarray(data.qvel, dtype=float), root_body_id=root_body_id
    )
    anchors["whole_body"] = (float(com[0]), float(com[1]), float(com[2]))
    for group, members in catalog.as_mapping().items():
        group_com = _mass_weighted_com(model, data, members)
        if group_com is not None:
            anchors[group] = group_com
    return anchors
