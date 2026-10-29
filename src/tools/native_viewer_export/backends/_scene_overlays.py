"""The per-frame overlay step shared by the MuJoCo-family render workers.

MuJoCo and MyoSuite both render through ``mujoco.Renderer``: after
``update_scene`` the segments are shaded by the staged axial loads and the 3D
force/torque glyphs are appended (shading first, so it recolours the model's
own geoms and never the glyph geoms).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyphs import (
    add_ellipsoids_to_scene,
    add_glyphs_to_scene,
)
from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.body_part_viz.mujoco_force_colors import (
    apply_mujoco_scene_colors,
)
from src.shared.python.force_overlay.glyphs import GlyphSet

__all__ = ["apply_scene_overlays"]


def apply_scene_overlays(
    model: Any,
    scene: Any,
    position: int,
    axial_frames: Sequence[AxialLoadFrame | None] | None,
    shading: ForceColorScale | None,
    glyph_sets: Sequence[GlyphSet] | None,
) -> None:
    """Shade the segments then add the glyphs of frame ``position``.

    Each part is skipped when its staged input is ``None``. Preconditions:
    ``position`` indexes the staged lists (``IndexError`` otherwise).
    """
    if axial_frames is not None and shading is not None:
        apply_mujoco_scene_colors(model, scene, axial_frames[position], shading)
    if glyph_sets is not None:
        frame = glyph_sets[position]
        add_glyphs_to_scene(scene, frame)
        if frame.ellipsoids:
            add_ellipsoids_to_scene(scene, frame.ellipsoids)
