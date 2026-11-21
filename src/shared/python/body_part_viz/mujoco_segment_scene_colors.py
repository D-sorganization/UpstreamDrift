"""Apply per-segment scalar shading on a native MuJoCo scene (#15395, #15399)."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import Any

from .scalar_shading import ScalarColorScale

__all__ = [
    "apply_mujoco_scalar_scene_colors",
    "apply_segment_body_geom_shading",
    "model_geom_base_rgb",
]

_RGB = tuple[float, float, float]
_ShadeFn = Callable[[float | None, _RGB], tuple[float, float, float] | None]


def model_geom_base_rgb(model: Any, geom_id: int) -> _RGB:
    """Model-authored RGB for one geom (stable under intermediate overlays)."""
    import mujoco

    if not isinstance(model, mujoco.MjModel):
        raise TypeError("model must be a native MuJoCo MjModel")
    if not 0 <= geom_id < model.ngeom:
        raise ValueError("geom_id out of range")
    rgba = model.geom_rgba[geom_id]
    return float(rgba[0]), float(rgba[1]), float(rgba[2])


def _iter_body_geoms(model: Any, scene: Any, keep_geoms: Collection[int]):
    import mujoco

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
        yield name, geom, model_geom_base_rgb(model, geom.objid)


def apply_segment_body_geom_shading(
    model: Any,
    scene: Any,
    values: Mapping[str, float | None],
    shade: _ShadeFn,
    *,
    keep_geoms: Collection[int],
    enabled: bool,
) -> None:
    if not enabled:
        return
    for name, geom, base in _iter_body_geoms(model, scene, keep_geoms):
        rgb = shade(values.get(name), base)
        if rgb is not None:
            geom.rgba[:3] = rgb


def apply_mujoco_scalar_scene_colors(
    model: Any,
    scene: Any,
    frame_values: Mapping[str, float | None] | None,
    scale: ScalarColorScale,
    *,
    keep_geoms: Collection[int] = frozenset(),
) -> None:
    """Recolour body geoms from a scalar map; next ``update_scene`` restores bases."""
    import mujoco

    if not isinstance(model, mujoco.MjModel) or not isinstance(scene, mujoco.MjvScene):
        raise TypeError("model and scene must be native MuJoCo objects")
    if not isinstance(scale, ScalarColorScale):
        raise TypeError("scale must be ScalarColorScale")
    if frame_values is not None and not isinstance(frame_values, Mapping):
        raise TypeError("frame_values must be a mapping or None")

    def shade(sample: float | None, base: _RGB):
        return scale.shade_rgb(sample, base)

    values = {} if frame_values is None else frame_values
    apply_segment_body_geom_shading(
        model, scene, values, shade, keep_geoms=keep_geoms, enabled=scale.enabled
    )
