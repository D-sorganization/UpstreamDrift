"""MuJoCo MjvScene glyph renderer for native viewers and offscreen renders (FTO-6, #11291).

Renders a GlyphSet into a MuJoCo MjvScene as true 3D geoms: shaded arrows,
arcs, and heads. Geoms are added directly into the pre-allocated scene.geoms
buffer and rendered by both the native viewer and mujoco.Renderer.

Minimum MuJoCo version: 2.3.0 (supports mjv_connector; legacy mjv_makeConnector
is supported as fallback).

Usage Example:
    >>> # After updating the scene and before rendering:
    >>> renderer.update_scene(data, camera)
    >>> receipt = add_glyphs_to_scene(renderer.scene, glyphs)
    >>> frame = renderer.render()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

from src.shared.python.force_overlay.contracts import ForceTorqueFrame
from src.shared.python.force_overlay.glyphs import (
    ForceGlyphStyle,
    GlyphSet,
    build_glyphs,
)

logger = logging.getLogger(__name__)

# Detect connector function once at module import
_HAS_MJV_CONNECTOR = hasattr(mujoco, "mjv_connector")
_HAS_MJV_MAKE_CONNECTOR = hasattr(mujoco, "mjv_makeConnector")

if not (_HAS_MJV_CONNECTOR or _HAS_MJV_MAKE_CONNECTOR):
    logger.warning(
        "MuJoCo installation lacks both mjv_connector and mjv_makeConnector."
    )


def _call_connector(
    geom: Any,
    geom_type: int,
    width: float,
    from_pt: npt.NDArray[np.float64],
    to_pt: npt.NDArray[np.float64],
) -> None:
    """Invoke the appropriate MuJoCo connector function based on runtime version."""
    if _HAS_MJV_CONNECTOR:
        mujoco.mjv_connector(geom, geom_type, width, from_pt, to_pt)
    elif _HAS_MJV_MAKE_CONNECTOR:
        try:
            mujoco.mjv_makeConnector(geom, geom_type, width, from_pt, to_pt)
        except TypeError:
            mujoco.mjv_makeConnector(
                geom,
                geom_type,
                width,
                float(from_pt[0]),
                float(from_pt[1]),
                float(from_pt[2]),
                float(to_pt[0]),
                float(to_pt[1]),
                float(to_pt[2]),
            )
    else:
        raise RuntimeError("MuJoCo installation lacks connector support.")


@dataclass(frozen=True)
class SceneGlyphReceipt:
    """Receipt summarizing geoms added to and dropped from an MjvScene."""

    added: int
    dropped: int


def segment_geom_count(glyphs: GlyphSet) -> int:
    """Return the total number of MjvGeom entries required to render glyphs.

    Hosts can size scene.maxgeom or warn before rendering.
    Each arrow requires 1 geom. Each torque arc requires (len(polyline) - 1)
    capsule connectors plus 1 arrow head connector.
    """
    count = len(glyphs.arrows)
    for arc in glyphs.torque_arcs:
        n_pts = len(arc.polyline_m)
        segments = max(0, n_pts - 1)
        count += segments + 1
    return count


def filter_frame_glyphs(
    frame: ForceTorqueFrame,
    *,
    show_forces: bool,
    show_torques: bool,
    force_scale: float = 0.001,
    torque_scale: float = 0.005,
) -> GlyphSet:
    """Build and filter a GlyphSet from a ForceTorqueFrame based on visibility toggles."""
    style = ForceGlyphStyle(
        force_scale_m_per_n=force_scale,
        torque_scale_m_per_nm=torque_scale,
    )
    glyphs = build_glyphs(frame, style)
    arrows = glyphs.arrows if show_forces else ()
    torque_arcs = glyphs.torque_arcs if show_torques else ()
    return GlyphSet(
        time_s=glyphs.time_s,
        arrows=arrows,
        torque_arcs=torque_arcs,
        legend=glyphs.legend,
    )


def _append_connector(
    scene: mujoco.MjvScene,
    geom_type: int,
    width: float,
    from_pt: npt.NDArray[np.float64],
    to_pt: npt.NDArray[np.float64],
    rgba: tuple[float, float, float, float],
) -> bool:
    """Write one connector geom to scene if capacity allows.

    Returns True if written, False if dropped due to buffer capacity.
    """
    if scene.ngeom >= scene.maxgeom:
        return False

    geom = scene.geoms[scene.ngeom]
    _call_connector(geom, geom_type, width, from_pt, to_pt)
    geom.rgba[:] = rgba
    scene.ngeom += 1
    return True


def add_glyphs_to_scene(
    scene: mujoco.MjvScene,
    glyphs: GlyphSet,
    *,
    arc_width_m: float = 0.006,
) -> SceneGlyphReceipt:
    """Draw a GlyphSet into a MuJoCo MjvScene as 3D arrows and torque arcs.

    Must be called after `mjv_updateScene` and before `mjr_render` or
    `mujoco.Renderer.render()`.

    Args:
        scene: Target MuJoCo scene whose pre-allocated geoms buffer is populated.
        glyphs: Deterministic GlyphSet containing arrows and torque arcs.
        arc_width_m: Diameter in meters of the torque arc polyline capsules.

    Returns:
        SceneGlyphReceipt recording the count of geoms added and dropped.

    Example:
        >>> renderer.update_scene(data, camera)
        >>> receipt = add_glyphs_to_scene(renderer.scene, glyphs)
        >>> frame = renderer.render()
    """
    if not isinstance(scene, mujoco.MjvScene):
        raise TypeError(f"scene must be a mujoco.MjvScene, got {type(scene).__name__}")
    if not isinstance(glyphs, GlyphSet):
        raise TypeError(f"glyphs must be a GlyphSet, got {type(glyphs).__name__}")
    if arc_width_m <= 0.0 or not np.isfinite(arc_width_m):
        raise ValueError(f"arc_width_m must be positive and finite, got {arc_width_m}")

    added = 0
    dropped = 0
    arrow_type = int(mujoco.mjtGeom.mjGEOM_ARROW)
    capsule_type = int(mujoco.mjtGeom.mjGEOM_CAPSULE)

    # 1. Force Arrows: 1 mjGEOM_ARROW per arrow
    for arrow in glyphs.arrows:
        from_pt = np.asarray(arrow.tail_m, dtype=np.float64)
        to_pt = np.asarray(arrow.tip_m, dtype=np.float64)
        width = 2.0 * float(arrow.shaft_radius_m)

        if _append_connector(scene, arrow_type, width, from_pt, to_pt, arrow.rgba):
            added += 1
        else:
            dropped += 1

    # 2. Torque Arcs: (N-1) mjGEOM_CAPSULE along polyline + 1 mjGEOM_ARROW head
    for arc in glyphs.torque_arcs:
        rgba = arc.rgba
        width = float(arc_width_m)
        polyline = arc.polyline_m
        n_pts = len(polyline)

        for i in range(n_pts - 1):
            from_pt = np.asarray(polyline[i], dtype=np.float64)
            to_pt = np.asarray(polyline[i + 1], dtype=np.float64)
            if _append_connector(scene, capsule_type, width, from_pt, to_pt, rgba):
                added += 1
            else:
                dropped += 1

        # Arrow head connector from head_base_m to head_tip_m
        from_head = np.asarray(arc.head_base_m, dtype=np.float64)
        to_head = np.asarray(arc.head_tip_m, dtype=np.float64)
        if _append_connector(scene, arrow_type, width, from_head, to_head, rgba):
            added += 1
        else:
            dropped += 1

    return SceneGlyphReceipt(added=added, dropped=dropped)
