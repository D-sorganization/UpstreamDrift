"""MyoHub MyoHand **bone** visual backend (anatomical skeleton reference, epic #15122).

Myo_sim provides STL **bone** meshes plus MuJoCo ``class="skin"`` capsule/ellipsoid
geoms (not a continuous skin surface). This backend renders bones only, in bone
colour, posed with the same curl-to-grip joint targets as the capsule reference.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
    resolved_shaft_roll_deg,
)
from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.hand_backends.joint_mapping import (
    anatomical_mirror_for_side,
    grip_spec_to_myohand_joint_angles,
)
from src.shared.python.model_appearance.hand_backends.mujoco_extract import (
    apply_joint_targets,
    load_myohand_model,
    transform_meshes,
)
from src.shared.python.model_appearance.hand_backends.registration import (
    posed_bones_in_model_frame,
    registration_to_grip_hand_mesh,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    MyoHandVisualSettings,
    default_myohand_settings,
)
from src.shared.python.model_appearance.hand_backends.types import (
    HandVisualMeshes,
    HandVisualSide,
)


def _check_grip_spec(spec: object) -> GripSpec:
    if not isinstance(spec, GripSpec):
        raise TypeError(f"grip_spec must be GripSpec, got {type(spec).__name__}")
    return spec


def _model_and_data(
    targets: dict[str, float],
    cfg: MyoHandVisualSettings,
) -> tuple[Any, Any, dict[str, Mesh]]:
    import mujoco

    model = load_myohand_model(cfg.model_xml_path)
    data = mujoco.MjData(model)
    apply_joint_targets(model, data, targets)
    raw = posed_bones_in_model_frame(
        model,
        data,
        allowed_bodies=frozenset(cfg.hand_body_names),
    )
    return model, data, raw


def _pipeline_transform(
    spec: GripSpec,
    side: HandVisualSide,
    raw_bones: dict[str, Mesh],
    cfg: MyoHandVisualSettings,
    joint_angles: Mapping[str, float] | None,
) -> np.ndarray:
    _ = joint_angles
    return registration_to_grip_hand_mesh(
        spec,
        side,
        raw_bones,
        cfg=cfg,
        grip_radius_m=cfg.grip_radius_m,
    )


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandVisualSide,
    joint_angles: Mapping[str, float] | None = None,
    *,
    settings: MyoHandVisualSettings | None = None,
) -> HandVisualMeshes:
    """Build posed MyoHand **bone** meshes in the grip hand-grip frame.

    Joint angles come from the shared curl-to-grip map (``pose_to_myohand_joint_targets``
    on :class:`GripSpec`); rigid placement Kabsch-aligns bones to the capsule hand
  mesh for that pose (contact reference, not a neutral calibrator).

    Research-only: see registry metadata and #13763 before production use.
    """
    spec = _check_grip_spec(grip_spec)
    if side not in ("lead", "trail"):
        raise ValueError(f"side must be 'lead' or 'trail', got {side!r}")
    cfg = default_myohand_settings() if settings is None else settings
    require(myohand_assets_ready(cfg), "myo_sim assets missing")

    targets = grip_spec_to_myohand_joint_angles(
        spec,
        side,
        grip_radius_m=cfg.grip_radius_m,
        joint_angle_overrides=joint_angles,
    )
    _model, _data, raw = _model_and_data(targets, cfg)

    transform = _pipeline_transform(spec, side, raw, cfg, joint_angles)
    mirror = anatomical_mirror_for_side(spec, side)
    placed = transform_meshes(raw, transform, mirror_y=mirror)

    roll_deg = resolved_shaft_roll_deg(spec, side)
    if abs(roll_deg) > 1e-12:
        frame = apply_shaft_roll(np.eye(4), APPEARANCE_GRIP_SHAFT_AXIS, roll_deg)
        placed = transform_meshes(placed, frame)

    return placed


def myohand_assets_ready(settings: MyoHandVisualSettings | None = None) -> bool:
    cfg = default_myohand_settings() if settings is None else settings
    return cfg.model_xml_path.is_file()


def penetration_depth_m(
    meshes: HandVisualMeshes,
    *,
    club_radius_m: float,
) -> float:
    if club_radius_m <= 0.0:
        raise ValueError("club_radius_m must be positive")
    if not meshes:
        return 0.0
    verts = np.vstack([m.vertices for m in meshes.values()])
    radial = np.linalg.norm(verts[:, :2], axis=1)
    return float(max(0.0, club_radius_m - radial.min()))


def finger_pair_penetration_m(
    meshes: HandVisualMeshes,
    body_a: str,
    body_b: str,
) -> float:
    if body_a not in meshes or body_b not in meshes:
        return float("inf")
    from scipy.spatial import cKDTree

    tree = cKDTree(meshes[body_b].vertices)
    dist, _ = tree.query(meshes[body_a].vertices, k=1)
    return float(-dist.min())


def mirror_y_vertices(mesh: Mesh) -> Mesh:
    verts = mesh.vertices.copy()
    verts[:, 1] *= -1.0
    return Mesh(verts, mesh.faces[:, ::-1].copy())


__all__ = [
    "_model_and_data",
    "_pipeline_transform",
    "build_hand_visual",
    "finger_pair_penetration_m",
    "mirror_y_vertices",
    "myohand_assets_ready",
    "penetration_depth_m",
]