"""Extract posed bone meshes from a MuJoCo MyoHand model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.model_appearance.geometry import Mesh, concat_meshes


@lru_cache(maxsize=2)
def _cached_model(xml_path: str, mtime_ns: int) -> Any:
    import mujoco

    return mujoco.MjModel.from_xml_path(xml_path)


def load_myohand_model(xml_path: Path) -> Any:
    """Load (cached) ``MjModel`` for ``xml_path``."""
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)
    return _cached_model(str(xml_path.resolve()), xml_path.stat().st_mtime_ns)


def apply_joint_targets(model: Any, data: Any, targets: dict[str, float]) -> None:
    import mujoco

    data.qpos[:] = 0.0
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if not name or name not in targets:
            continue
        adr = model.jnt_qposadr[joint]
        data.qpos[adr] = float(targets[name])
    mujoco.mj_forward(model, data)


def extract_hand_bone_meshes(
    model: Any,
    data: Any,
    *,
    allowed_bodies: frozenset[str],
) -> dict[str, Mesh]:
    """World-frame bone meshes grouped by body name (visual ``type=mesh`` only)."""
    import mujoco

    out: dict[str, list[np.ndarray]] = {}
    faces: dict[str, list[np.ndarray]] = {}
    base_index: dict[str, int] = {}

    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body not in allowed_bodies:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        vadr = int(model.mesh_vertadr[mesh_id])
        vnum = int(model.mesh_vertnum[mesh_id])
        fadr = int(model.mesh_faceadr[mesh_id])
        fnum = int(model.mesh_facenum[mesh_id])
        local = np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=float)
        rot = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        world = local @ rot.T + np.asarray(data.geom_xpos[geom_id], dtype=float)
        tri = np.asarray(model.mesh_face[fadr : fadr + fnum], dtype=int).reshape(-1, 3)
        base = base_index.get(body, 0)
        base_index[body] = base + len(world)
        out.setdefault(body, []).append(world)
        faces.setdefault(body, []).append(tri + base)

    meshes: dict[str, Mesh] = {}
    for body, chunks in out.items():
        verts = np.vstack(chunks)
        merged_tri = np.vstack(faces[body])
        meshes[body] = Mesh(verts, merged_tri)
    return meshes


def transform_meshes(
    meshes: dict[str, Mesh],
    transform: np.ndarray,
    *,
    mirror_y: bool = False,
) -> dict[str, Mesh]:
    """Apply a rigid/affine ``4x4`` transform; flip winding whenever parity is odd."""
    t = np.asarray(transform, dtype=float)
    if t.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    rot = t[:3, :3]
    trans = t[:3, 3]
    flip_winding = (float(np.linalg.det(rot)) < 0.0) ^ bool(mirror_y)
    out: dict[str, Mesh] = {}
    for name, mesh in meshes.items():
        verts = mesh.vertices @ rot.T + trans
        if mirror_y:
            verts[:, 1] *= -1.0
        faces = mesh.faces[:, ::-1].copy() if flip_winding else mesh.faces
        out[name] = Mesh(verts, faces)
    return out


def anthropometric_scale_matrix(
    length_m: float,
    breadth_m: float,
    *,
    reference_length_m: float,
    width_from_breadth: bool,
    exponent: float,
) -> np.ndarray:
    """Uniform length scale with optional breadth axis stretch in grip frame."""
    if reference_length_m <= 0.0:
        raise ValueError("reference_length_m must be positive")
    s = (length_m / reference_length_m) ** exponent
    breadth_ref = reference_length_m * 0.449
    sy = (breadth_m / breadth_ref) if width_from_breadth else s
    return np.diag([s, sy, s, 1.0])


def merged_hand_mesh(meshes: dict[str, Mesh]) -> Mesh:
    if not meshes:
        raise ValueError("no meshes to merge")
    return concat_meshes(*meshes.values())


__all__ = [
    "anthropometric_scale_matrix",
    "apply_joint_targets",
    "extract_hand_bone_meshes",
    "load_myohand_model",
    "merged_hand_mesh",
    "transform_meshes",
]
