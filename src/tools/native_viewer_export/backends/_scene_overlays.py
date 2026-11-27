"""The per-frame overlay step shared by the MuJoCo-family render workers.

MuJoCo and MyoSuite both render through ``mujoco.Renderer``: after
``update_scene`` the segments are shaded by the staged axial loads and the 3D
force/torque glyphs are appended (shading first, so it recolours the model's
own geoms and never the glyph geoms). The geoms a skinned model keeps
unshaded (hair, eyes, brows, lips) are passed as ``keep_geoms``; the skin
pieces themselves are coloured before ``update_scene`` (``_skin_shading``).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
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
from src.tools.native_viewer_export.arrow_style import head_span_shaft_radii

__all__ = ["apply_scene_overlays"]


def apply_scene_overlays(
    model: Any,
    scene: Any,
    position: int,
    axial_frames: Sequence[AxialLoadFrame | None] | None,
    shading: ForceColorScale | None,
    glyph_sets: Sequence[GlyphSet] | None,
    *,
    keep_geoms: Collection[int] = frozenset(),
) -> None:
    """Shade the segments then add the glyphs of frame ``position``.

    Each part is skipped when its staged input is ``None``. Model geoms in
    ``keep_geoms`` keep their colour; with none (the default) every geom of a
    loaded body is shaded. Preconditions: ``position`` indexes the staged
    lists (``IndexError`` otherwise).
    """
    if axial_frames is not None and shading is not None:
        apply_mujoco_scene_colors(
            model, scene, axial_frames[position], shading, keep_geoms=keep_geoms
        )
    if glyph_sets is not None:
        frame = glyph_sets[position]
        add_glyphs_to_scene(
            scene, frame, arrow_head_shaft_radii=head_span_shaft_radii()
        )
        if frame.ellipsoids:
            add_ellipsoids_to_scene(scene, frame.ellipsoids)
