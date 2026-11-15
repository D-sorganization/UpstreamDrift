"""MyoHub MyoHand anatomical mesh visual backend (epic #15122)."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    anthropometric_scale_matrix,
    apply_joint_targets,
    extract_hand_bone_meshes,
    load_myohand_model,
    transform_meshes,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    MyoHandVisualSettings,
    default_myohand_settings,
)
from src.shared.python.model_appearance.hand_backends.types import (
    HandVisualMeshes,
    HandVisualSide,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _check_grip_spec(spec: object) -> GripSpec:
    if not isinstance(spec, GripSpec):
        raise TypeError(f"grip_spec must be GripSpec, got {type(spec).__name__}")
    return spec


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandVisualSide,
    joint_angles: Mapping[str, float] | None = None,
    *,
    settings: MyoHandVisualSettings | None = None,
) -> HandVisualMeshes:
    """Build posed MyoHand bone meshes in the grip hand-grip frame.

    Preconditions:
        Pinned ``myo_sim`` assets exist (see :func:`myohand_assets_ready`).
    Postconditions:
        Returned meshes are in the same hand-grip frame as :mod:`grip_hand_mesh`
        (origin on the grip axis, ``z`` butt→head), keyed by carpal/phalanx body
        name. Visual only — not used for contact.
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

    import mujoco

    model = load_myohand_model(cfg.model_xml_path)
    data = mujoco.MjData(model)
    apply_joint_targets(model, data, targets)

    raw = extract_hand_bone_meshes(
        model,
        data,
        allowed_bodies=frozenset(cfg.hand_body_names),
    )

    scale = anthropometric_scale_matrix(
        spec.hand_size.length_m,
        spec.hand_size.breadth_m,
        reference_length_m=cfg.anthropometric_reference_hand_length_m,
        width_from_breadth=cfg.width_scale_from_breadth,
        exponent=cfg.anthropometric_scale_exponent,
    )
    mirror = anatomical_mirror_for_side(spec, side)
    reg = np.asarray(cfg.model_to_grip, dtype=float)
    reg = scale @ reg

    placed = transform_meshes(raw, reg, mirror_y=mirror)

    roll_deg = resolved_shaft_roll_deg(spec, side)
    if abs(roll_deg) > 1e-12:
        frame = np.eye(4)
        frame = apply_shaft_roll(frame, APPEARANCE_GRIP_SHAFT_AXIS, roll_deg)
        placed = transform_meshes(placed, frame)

    return placed


def myohand_assets_ready(settings: MyoHandVisualSettings | None = None) -> bool:
    """True when the pinned MyoHand MJCF is on disk."""
    cfg = default_myohand_settings() if settings is None else settings
    return cfg.model_xml_path.is_file()


def penetration_depth_m(
    meshes: HandVisualMeshes,
    *,
    club_radius_m: float,
) -> float:
    """Maximal grip-cylinder penetration (positive means inside the club)."""
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
    """Symmetric minimum distance between two part meshes (negative = overlap)."""
    if body_a not in meshes or body_b not in meshes:
        return float("inf")
    from scipy.spatial import cKDTree

    a = meshes[body_a].vertices
    b = meshes[body_b].vertices
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1)
    return float(-d.min())


def mirror_y_vertices(mesh: Mesh) -> Mesh:
    """Mirror mesh about grip-frame ``y`` (handedness test helper)."""
    verts = mesh.vertices.copy()
    verts[:, 1] *= -1.0
    faces = mesh.faces[:, ::-1].copy()
    return Mesh(verts, faces)


__all__ = [
    "build_hand_visual",
    "finger_pair_penetration_m",
    "mirror_y_vertices",
    "myohand_assets_ready",
    "penetration_depth_m",
]
