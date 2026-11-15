"""MyoHand joint centers and axes in the grip hand-grip frame."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends.joint_mapping import (
    anatomical_mirror_for_side,
    grip_spec_to_myohand_joint_angles,
)
from src.shared.python.model_appearance.hand_backends.myohand import (
    _model_and_data,
    _pipeline_transform,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    MyoHandVisualSettings,
    default_myohand_settings,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualSide


def _world_to_grip(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    return points @ rot.T + trans


def export_myohand_skeleton(
    grip_spec: GripSpec,
    side: HandVisualSide,
    *,
    settings: MyoHandVisualSettings | None = None,
    joint_angles: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Joint origins and axes in the grip hand-grip frame for the skinned-human lane."""
    import mujoco

    from src.engines.physics_engines.myosuite.python.myohand_bilateral_scene import (
        MYOHAND_CANONICAL_JOINTS,
    )

    cfg = default_myohand_settings() if settings is None else settings
    targets = grip_spec_to_myohand_joint_angles(
        grip_spec,
        side,
        grip_radius_m=cfg.grip_radius_m,
        joint_angle_overrides=joint_angles,
    )
    model, data, raw = _model_and_data(targets, cfg)
    transform = _pipeline_transform(grip_spec, side, raw, cfg, joint_angles)
    mirror = anatomical_mirror_for_side(grip_spec, side)

    joints: dict[str, dict[str, list[float]]] = {}
    name_to_id = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): j
        for j in range(model.njnt)
    }
    for jname in MYOHAND_CANONICAL_JOINTS:
        jid = name_to_id.get(jname)
        if jid is None:
            continue
        origin = np.asarray(data.xanchor[jid], dtype=float).reshape(1, 3)
        axis = np.asarray(data.xaxis[jid], dtype=float).reshape(1, 3)
        origin_g = _world_to_grip(origin, transform)[0]
        axis_g = axis @ transform[:3, :3].T
        axis_g = axis_g[0] / max(float(np.linalg.norm(axis_g)), 1e-12)
        if mirror:
            origin_g[1] *= -1.0
            axis_g[1] *= -1.0
        joints[jname] = {
            "origin_m": [float(v) for v in origin_g],
            "axis_unit": [float(v) for v in axis_g],
        }

    return {
        "schema_version": 1,
        "backend": "myohand_skeleton",
        "research_only": True,
        "license_gate_issue": 13763,
        "grip_spec_sha256": grip_spec.spec_sha256,
        "handedness": grip_spec.handedness,
        "side": side,
        "joint_angles_rad": {k: float(v) for k, v in targets.items()},
        "joints": joints,
    }


def export_bilateral_myohand_skeleton(
    grip_spec: GripSpec,
    *,
    settings: MyoHandVisualSettings | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "backend": "myohand_skeleton",
        "grip_spec_sha256": grip_spec.spec_sha256,
        "lead": export_myohand_skeleton(grip_spec, "lead", settings=settings),
        "trail": export_myohand_skeleton(grip_spec, "trail", settings=settings),
    }


__all__ = ["export_bilateral_myohand_skeleton", "export_myohand_skeleton"]