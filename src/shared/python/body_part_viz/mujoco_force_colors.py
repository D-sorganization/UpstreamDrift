"""Color a freshly updated MuJoCo visual scene without changing model materials."""

from __future__ import annotations

from typing import Any

from .axial_loads import AxialLoadFrame
from .force_colors import ForceColorScale


def apply_mujoco_scene_colors(
    model: Any, scene: Any, frame: AxialLoadFrame | None, scale: ForceColorScale
) -> None:
    """Apply after each native update_scene; the next update restores base styling.

    Only scene RGB values change (blended with each geom's current color per
    the scale's opacity and blend mode), preserving opacity, geometry, overlays and the
    physics model. Callers must supply a fresh scene and a synchronous force frame.
    """
    import mujoco

    if not isinstance(model, mujoco.MjModel) or not isinstance(scene, mujoco.MjvScene):
        raise TypeError("model and scene must be native MuJoCo objects")
    if not isinstance(scale, ForceColorScale):
        raise TypeError("scale must be ForceColorScale")
    if frame is not None and not isinstance(frame, AxialLoadFrame):
        raise TypeError("frame must be AxialLoadFrame or None")
    if not scale.enabled or frame is None:
        return
    for index in range(scene.ngeom):
        geom = scene.geoms[index]
        if (
            geom.objtype != mujoco.mjtObj.mjOBJ_GEOM
            or not 0 <= geom.objid < model.ngeom
        ):
            continue
        body = model.geom_bodyid[geom.objid]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        base = tuple(float(channel) for channel in geom.rgba[:3])
        rgb = scale.shade_rgb(frame.values_n.get(name), base)
        if rgb is not None:
            geom.rgba[:3] = rgb
