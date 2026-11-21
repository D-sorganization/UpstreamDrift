"""Color a freshly updated MuJoCo visual scene without changing model materials."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from .axial_loads import AxialLoadFrame
from .force_colors import ForceColorScale
from .mujoco_segment_scene_colors import apply_mujoco_segment_scene_colors


def unshaded_geom_ids(model: Any, prefixes: Collection[str]) -> frozenset[int]:
    """Ids of the model geoms whose material name starts with one of ``prefixes``.

    These are the geoms that keep their own colour under segment shading (hair,
    eyes, brows, lips on the head body). Preconditions: ``model`` is a native
    MjModel and ``prefixes`` a collection of names, not one string
    (``TypeError`` otherwise). A geom without a material never matches.
    """
    import mujoco

    if not isinstance(model, mujoco.MjModel):
        raise TypeError("model must be a native MuJoCo MjModel")
    if isinstance(prefixes, str) or not all(isinstance(p, str) for p in prefixes):
        raise TypeError("prefixes must be a collection of material name prefixes")
    starts = tuple(prefixes)
    kept = set()
    for geom in range(model.ngeom if starts else 0):
        material = int(model.geom_matid[geom])
        if material < 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, material)
        if (name or "").startswith(starts):
            kept.add(geom)
    return frozenset(kept)


def apply_mujoco_scene_colors(
    model: Any,
    scene: Any,
    frame: AxialLoadFrame | None,
    scale: ForceColorScale,
    *,
    keep_geoms: Collection[int] = frozenset(),
) -> None:
    """Apply after each native update_scene; the next update restores base styling.

    Only scene RGB values change (blended with each geom's current color per
    the scale's opacity and blend mode), preserving opacity, geometry, overlays and the
    physics model. Callers must supply a fresh scene and a synchronous force frame.
    Model geoms in ``keep_geoms`` (see :func:`unshaded_geom_ids`) keep their colour.
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
    apply_mujoco_segment_scene_colors(
        model,
        scene,
        frame.values_n,
        lambda sample, base: scale.shade_rgb(sample, base),
        keep_geoms=keep_geoms,
    )
