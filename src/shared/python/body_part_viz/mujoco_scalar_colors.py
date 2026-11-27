"""Recolour MuJoCo scene geoms by generic per-segment scalars (#15396)."""

from __future__ import annotations

from collections.abc import Collection, Mapping, MutableSet
from typing import Any

from .mujoco_segment_scene_colors import apply_mujoco_segment_scene_colors
from .mujoco_force_colors import unshaded_geom_ids
from .scalar_shading import ScalarColorScale


def apply_mujoco_scalar_scene_colors(
    model: Any,
    scene: Any,
    values: Mapping[str, float | None] | None,
    scale: ScalarColorScale,
    *,
    keep_geoms: Collection[int] = frozenset(),
    clipped_segments: MutableSet[str] | None = None,
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

    def _saturated(sample: float | None) -> bool:
        return scale.evaluate(sample, "#000000").saturated

    apply_mujoco_segment_scene_colors(
        model,
        scene,
        values,
        lambda sample, base: scale.shade_rgb(sample, base),
        keep_geoms=keep_geoms,
        clipped_segments=clipped_segments,
        saturation_flag=_saturated,
    )


__all__ = ["apply_mujoco_scalar_scene_colors", "unshaded_geom_ids"]
