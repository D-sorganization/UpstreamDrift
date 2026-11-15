"""Rigid registration of posed MyoHand bones into the grip hand-grip frame."""

from __future__ import annotations

import numpy as np

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.geometry import Mesh, concat_meshes
from src.shared.python.model_appearance.grip_hand_mesh import build_grip_hand_mesh
from src.shared.python.model_appearance.hand_backends.joint_mapping import (
    _hand_for_side,
    _mesh_params,
    grip_spec_to_myohand_joint_angles,
)
from src.shared.python.model_appearance.hand_backends.mujoco_extract import (
    anthropometric_scale_matrix,
    extract_hand_bone_meshes,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    MyoHandVisualSettings,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualSide


def kabsch_rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return ``4×4`` rigid transform mapping ``source`` rows onto ``target`` rows."""
    if source.shape != target.shape or source.shape[1] != 3:
        raise ValueError("source and target must be (N, 3) with the same N")
    if len(source) < 3:
        raise ValueError("need at least three points for Kabsch registration")
    c_s = source.mean(axis=0)
    c_t = target.mean(axis=0)
    a = source - c_s
    b = target - c_t
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt = vt.copy()
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    out = np.eye(4, dtype=float)
    out[:3, :3] = rot
    out[:3, 3] = c_t - c_s @ rot.T
    return out


def _subsample(verts: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(verts) <= count:
        return verts
    start = seed % len(verts)
    step = max(1, (len(verts) - 1) // max(count - 1, 1))
    idx = (start + step * np.arange(count)) % len(verts)
    return verts[idx]


def registration_to_grip_hand_mesh(
    spec: GripSpec,
    side: HandVisualSide,
    raw_bones: dict[str, Mesh],
    *,
    cfg: MyoHandVisualSettings,
    grip_radius_m: float,
    max_samples: int = 600,
) -> np.ndarray:
    """Kabsch-align posed bones to the curled capsule reference in grip-hand frame."""
    hand = _hand_for_side(side, spec)
    params = _mesh_params(spec)
    pose = spec.to_grip_pose()
    reference = build_grip_hand_mesh(
        hand,
        grip_radius_m=grip_radius_m,
        pose=pose,
        params=params,
    ).mesh

    scale = anthropometric_scale_matrix(
        spec.hand_size.length_m,
        spec.hand_size.breadth_m,
        reference_length_m=cfg.anthropometric_reference_hand_length_m,
        width_from_breadth=cfg.width_scale_from_breadth,
        exponent=cfg.anthropometric_scale_exponent,
    )

    bone = concat_meshes(*raw_bones.values()).vertices
    bone = bone @ scale[:3, :3].T + scale[:3, 3]
    # Shaft roll must not change Kabsch subsampling (roll is applied after registration).
    angle_targets = grip_spec_to_myohand_joint_angles(
        spec,
        side,
        grip_radius_m=grip_radius_m,
    )
    targets_tuple = tuple(sorted(angle_targets.items()))
    seed = hash((side, targets_tuple)) & 0xFFFF_FFFF
    return kabsch_rigid_transform(
        _subsample(bone, max_samples, seed),
        _subsample(reference.vertices, max_samples, seed + 1),
    )


def posed_bones_in_model_frame(
    model, data, *, allowed_bodies: frozenset[str]
) -> dict[str, Mesh]:
    return extract_hand_bone_meshes(model, data, allowed_bodies=allowed_bodies)


__all__ = [
    "kabsch_rigid_transform",
    "posed_bones_in_model_frame",
    "registration_to_grip_hand_mesh",
]
