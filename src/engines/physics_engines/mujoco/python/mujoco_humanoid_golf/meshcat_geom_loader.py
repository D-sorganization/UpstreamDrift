"""Build MeshCat visual nodes and per-body colour bindings from a MuJoCo model."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

try:
    import meshcat.geometry as g
    import meshcat.transformations as tf
except ImportError:
    g = None
    tf = None


def _meshcat_shape(gtype: int, size: np.ndarray, rbound: float) -> Any | None:
    if g is None:
        return None
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return g.Sphere(radius=size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return g.Cylinder(height=size[1] * 2, radius=size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return g.Cylinder(height=size[1] * 2, radius=size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return g.Box(lengths=size * 2)
    if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
        return g.Box([20, 20, 0.01])
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        return g.Sphere(radius=rbound)
    return g.Sphere(radius=0.1)


def build_meshcat_color_bindings(
    vis: Any,
    model: mujoco.MjModel,
    *,
    rgba_to_int: Any,
) -> dict[str, dict[str, list[float]]]:
    """Create visual geometry under ``visuals`` and return segment colour bindings."""
    if vis is None or g is None or tf is None:
        return {}
    color_bindings: dict[str, dict[str, list[float]]] = {}
    for i in range(model.ngeom):
        gtype = model.geom_type[i]
        size = model.geom_size[i]
        rgba = model.geom_rgba[i]
        shape = _meshcat_shape(gtype, size, float(model.geom_rbound[i]))
        if shape is None:
            continue
        material = g.MeshPhongMaterial(color=rgba_to_int(rgba), opacity=rgba[3])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
        z_aligned = gtype in (
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
        )
        if z_aligned:
            vis["visuals"][name]["geometry"].set_object(shape, material)
            vis["visuals"][name]["geometry"].set_transform(
                tf.rotation_matrix(np.pi / 2, [1, 0, 0])
            )
        else:
            vis["visuals"][name].set_object(shape, material)
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i])
        if body:
            suffix = "/geometry" if z_aligned else ""
            path = f"visuals/{name}{suffix}/<object>"
            color_bindings.setdefault(body, {})[path] = [
                *(int(c * 255) / 255 for c in rgba[:3]),
                float(rgba[3]),
            ]
    return color_bindings


__all__ = ["build_meshcat_color_bindings"]
