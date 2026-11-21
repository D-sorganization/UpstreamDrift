"""Shared MuJoCo scene-geom iteration for per-segment colour overlays."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, MutableSet
from typing import Any

def apply_mujoco_segment_scene_colors(
    model: Any,
    scene: Any,
    values: Mapping[str, float | None] | None,
    shade_rgb: Callable[
        [float | None, tuple[float, float, float]],
        tuple[float, float, float] | None,
    ],
    *,
    keep_geoms: Collection[int] = frozenset(),
    clipped_segments: MutableSet[str] | None = None,
    saturation_flag: Callable[[float | None], bool] | None = None,
) -> None:
    """Tint scene geoms by body name; optional ``clipped_segments`` collects saturated IDs."""
    import mujoco

    if not isinstance(model, mujoco.MjModel) or not isinstance(scene, mujoco.MjvScene):
        raise TypeError("model and scene must be native MuJoCo objects")
    if values is not None and not isinstance(values, Mapping):
        raise TypeError("values must be a mapping or None")
    if values is None:
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
        if not name:
            continue
        base = tuple(float(channel) for channel in geom.rgba[:3])
        sample = values.get(name)
        rgb = shade_rgb(sample, base)
        if rgb is not None:
            geom.rgba[:3] = rgb
            if (
                clipped_segments is not None
                and saturation_flag is not None
                and saturation_flag(sample)
            ):
                clipped_segments.add(name)


__all__ = ["apply_mujoco_segment_scene_colors"]
