"""Linear blend skinning for hand assets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SkinnedHandAsset:
    rest_vertices: Array
    faces: NDArray[np.int64]
    joint_indices: NDArray[np.uint16]  # (V, 4)
    joint_weights: NDArray[np.float32]  # (V, 4)
    rest_poses: Array  # (B, 4, 4)
    bone_names: tuple[str, ...]
    palm_arch_vertex_ids: NDArray[np.int64]
    grip_bind_vertex_ids: NDArray[np.int64]


def _transform(name: str, mat: Array) -> Array:
    m = np.asarray(mat, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError(f"{name}: pose must be a finite 4x4 matrix")
    return m


def pose_vertices(asset: SkinnedHandAsset, bone_poses: dict[str, Array]) -> Array:
    missing = set(asset.bone_names) - set(bone_poses)
    if missing:
        raise ValueError(f"missing bone poses: {sorted(missing)}")
    posed = np.stack([_transform(name, bone_poses[name]) for name in asset.bone_names])
    rest_inv = np.linalg.inv(asset.rest_poses)
    homog = np.concatenate(
        [asset.rest_vertices, np.ones((len(asset.rest_vertices), 1))], axis=1
    )
    out = np.zeros_like(asset.rest_vertices)
    for k in range(asset.joint_indices.shape[1]):
        w = asset.joint_weights[:, k]
        if not np.any(w > 0):
            continue
        idx = asset.joint_indices[:, k]
        delta = posed[idx] @ rest_inv[idx]
        transformed = np.einsum("nij,nj->ni", delta, homog)[:, :3]
        out += w[:, None] * transformed
    return out


def apply_palm_correctives(
    vertices: Array,
    asset: SkinnedHandAsset,
    *,
    arch_rad: float,
    arch_gain: float,
    grip_radius_m: float,
) -> Array:
    """Cup the palm toward the grip using corrective offsets on arch-weighted verts."""
    if abs(arch_rad) < 1e-9 or len(asset.palm_arch_vertex_ids) == 0:
        return vertices
    v = vertices.copy()
    ids = asset.palm_arch_vertex_ids
    pts = v[ids]
    rho = np.hypot(pts[:, 0], pts[:, 1])
    w = np.clip((rho - grip_radius_m * 0.6) / (grip_radius_m * 0.5), 0.0, 1.0)
    radial = pts / np.maximum(rho[:, None], 1e-9)
    delta = -radial * (arch_rad * arch_gain * w)[:, None]
    v[ids] = pts + delta
    return v
