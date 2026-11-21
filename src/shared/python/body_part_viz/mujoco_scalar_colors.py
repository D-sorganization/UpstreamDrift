"""Recolour MuJoCo scene geoms by generic per-segment scalars (#15396)."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from .mujoco_force_colors import unshaded_geom_ids
from .scalar_shading import ScalarColorScale


def apply_mujoco_scalar_scene_colors(
    model: Any,
    scene: Any,
    values: Mapping[str, float | None] | None,
    scale: ScalarColorScale,
    *,
    keep_geoms: Collection[int] = frozenset(),
) -> None:
    """Apply fixed-limit scalar shading after each native ``update_scene``."""
    import mujoco

    if not isinstance(model, mujoco.MjModel) or not isinstance(scene, mujoco.MjvScene):
        raise TypeError("model and scene must be native MuJoCo objects")
    if not isinstance(scale, ScalarColorScale):
        raise TypeError("scale must be ScalarColorScale")
    if values is not None and not isinstance(values, Mapping):
        raise TypeError("values must be a mapping or None")
    if not scale.enabled or values is None:
        return
    for index in range(scene.ngeom):
        geom = scene.geoms[index]
        if (
            geom.objtype != mujoco.mjtObj.mjOBJ_GEOM
            or not 0 <= geom.objid < model.ngeom
            or geom.objid in keep_geoms
        ):
            continue
        body = model.geom_bodyid[geom.objid]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        base = tuple(float(channel) for channel in geom.rgba[:3])
        rgb = scale.shade_rgb(values.get(name), base)
        if rgb is not None:
            geom.rgba[:3] = rgb


__all__ = ["apply_mujoco_scalar_scene_colors", "unshaded_geom_ids"]
