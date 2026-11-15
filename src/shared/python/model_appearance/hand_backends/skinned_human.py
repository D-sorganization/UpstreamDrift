"""Skinned MakeHuman-derived hand visual backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.grip_spec.shaft_roll import (
    apply_shaft_roll,
    resolved_shaft_roll_deg,
)
from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.hand_size import hand_mesh_scale
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.hand_backends.hand_asset import (
    load_hand_cc0_asset,
)
from src.shared.python.model_appearance.hand_backends.joint_angles import (
    HandJointAngles,
    HandSide,
    mirror_hand_joint_angles,
)
from src.shared.python.model_appearance.hand_backends.skeleton import (
    bone_poses_from_joint_angles,
)
from src.shared.python.model_appearance.hand_backends.skinning import (
    apply_palm_correctives,
    pose_vertices,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualResult

BACKEND_NAME = "skinned_human"
_SETTINGS_PATH = (
    Path(__file__).resolve().parents[5]
    / "scripts"
    / "config"
    / "skinned_human_hand.json"
)
_REPO = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class SkinnedHumanSettings:
    skin_rgb: tuple[float, float, float]
    palm_arch_gain: float
    grip_wrap_translation_m: float
    length_reference_m: float
    breadth_reference_m: float
    shaft_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)


@lru_cache(maxsize=1)
def default_settings() -> SkinnedHumanSettings:
    if not _SETTINGS_PATH.is_file():
        raise ValueError(f"missing settings: {_SETTINGS_PATH}")
    raw = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    rgb = tuple(float(x) for x in raw["skin_rgb"])
    return SkinnedHumanSettings(
        skin_rgb=(rgb[0], rgb[1], rgb[2]),
        palm_arch_gain=float(raw["palm_arch_gain"]),
        grip_wrap_translation_m=float(raw["grip_wrap_translation_m"]),
        length_reference_m=float(raw["length_reference_m"]),
        breadth_reference_m=float(raw["breadth_reference_m"]),
    )


def _scale_factors(spec: GripSpec) -> tuple[float, float]:
    settings = default_settings()
    length_scale = hand_mesh_scale(spec.hand_size.length_m) / hand_mesh_scale(
        settings.length_reference_m
    )
    breadth_scale = spec.hand_size.breadth_m / settings.breadth_reference_m
    uniform = (length_scale + breadth_scale) / 2.0
    return uniform, breadth_scale


def _clamp_grip_clearance(
    vertices: np.ndarray, grip_radius_m: float, pad_m: float
) -> np.ndarray:
    v = vertices.copy()
    rho = np.hypot(v[:, 0], v[:, 1])
    on_grip = (v[:, 0] < 0.04) & (rho < grip_radius_m + pad_m)
    for i in np.nonzero(on_grip)[0]:
        if rho[i] < 1e-9:
            continue
        scale = (grip_radius_m + pad_m) / rho[i]
        v[i, 0] *= scale
        v[i, 1] *= scale
    return v


def _place_on_grip(
    vertices: np.ndarray, grip_radius_m: float, settings: SkinnedHumanSettings
) -> np.ndarray:
    v = vertices.copy()
    palm_side = v[:, 0] < 0.0
    if not np.any(palm_side):
        palm_side = np.ones(len(v), dtype=bool)
    rho = np.hypot(v[palm_side, 0], v[palm_side, 1])
    min_rho = float(rho.min())
    if min_rho < 1e-9:
        return v
    target = grip_radius_m + settings.grip_wrap_translation_m
    v *= target / min_rho
    return v


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandSide,
    joint_angles: HandJointAngles,
    *,
    grip_radius_m: float,
    settings: SkinnedHumanSettings | None = None,
) -> HandVisualResult:
    """Return a skinned hand mesh in the hand-grip frame."""
    require(isinstance(grip_spec, GripSpec), "grip_spec must be GripSpec")
    require(side in ("lead", "trail"), "side must be lead or trail")
    require(
        isinstance(joint_angles, HandJointAngles),
        "joint_angles must be HandJointAngles",
    )
    if grip_radius_m <= 0.0 or not np.isfinite(grip_radius_m):
        raise ValueError("grip_radius_m must be positive and finite")
    settings = settings or default_settings()
    record = load_hand_cc0_asset()
    asset = record.asset
    angles = mirror_hand_joint_angles(joint_angles) if side == "trail" else joint_angles
    bone_poses = bone_poses_from_joint_angles(
        angles, bone_names=asset.bone_names, rest_poses=asset.rest_poses
    )
    verts = pose_vertices(asset, bone_poses)
    verts = apply_palm_correctives(
        verts,
        asset,
        arch_rad=angles.palm_arch_rad,
        arch_gain=settings.palm_arch_gain,
        grip_radius_m=grip_radius_m,
    )
    uniform, breadth = _scale_factors(grip_spec)
    verts[:, 0] *= breadth
    verts[:, 1] *= breadth
    verts[:, 2] *= uniform
    if side == "trail":
        verts[:, 1] *= -1.0
    verts = _place_on_grip(verts, grip_radius_m, settings)
    verts = _clamp_grip_clearance(
        verts, grip_radius_m, settings.grip_wrap_translation_m
    )
    roll_deg = resolved_shaft_roll_deg(grip_spec, side)
    axis = np.asarray(settings.shaft_axis, dtype=float)
    rot = apply_shaft_roll(np.eye(3), axis, roll_deg)[:3, :3]
    verts = verts @ rot.T
    return HandVisualResult(
        backend=BACKEND_NAME,
        side=side,
        mesh=Mesh(verts, asset.faces.copy()),
        skin_rgb=settings.skin_rgb,
    )


def register() -> None:
    from src.shared.python.model_appearance.hand_backends.registry import (
        HandVisualBackend,
        register_hand_visual_backend,
    )

    register_hand_visual_backend(
        HandVisualBackend(name=BACKEND_NAME, builder=build_hand_visual)
    )
