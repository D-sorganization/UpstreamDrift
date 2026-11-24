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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

from src.shared.python.force_overlay.capacity_glyphs import (
    CapacityWireGlyph,
    LineSegmentGlyph,
    capacity_wire_segments,
)
from src.shared.python.force_overlay.glyphs import EllipsoidGlyph, GlyphSet

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


#: Length of the white marker arrow drawn past the tip of a clamped arrow (GCV-4).
CLAMPED_TIP_LENGTH_M = 0.1

#: Drawn length of an ``mjGEOM_ARROW`` as a fraction of its ``size[2]``.
#: ``mjv_connector`` writes the full tail-to-tip length to ``size[2]``, but the
#: renderer draws the arrow half that long (measured on MuJoCo 3.6 and 3.8: a
#: 2 m connector arrow ends level with a 1 m capsule), so arrows are stretched
#: by its inverse after the connector call (#11729).
ARROW_RENDER_LENGTH_FRACTION = 0.5


@dataclass(frozen=True)
class SceneGlyphReceipt:
    """Receipt summarizing geoms added to and dropped from an MjvScene."""

    added: int
    dropped: int


def segment_geom_count(
    glyphs: GlyphSet, *, arrow_head_shaft_radii: float | None = None
) -> int:
    """Return the total number of MjvGeom entries required to render glyphs.

    Hosts can size scene.maxgeom or warn before rendering.
    Each arrow requires 1 geom; with ``arrow_head_shaft_radii`` (see
    ``add_glyphs_to_scene``) an upper bound of 2 is counted per arrow. Each
    torque arc requires (len(polyline) - 1) capsule connectors plus 1 arrow
    head connector.
    """
    per_arrow = 1 if arrow_head_shaft_radii is None else 2
    count = per_arrow * len(glyphs.arrows)
    count += sum(1 for a in glyphs.arrows if a.clamped)
    for arc in glyphs.torque_arcs:
        n_pts = len(arc.polyline_m)
        segments = max(0, n_pts - 1)
        count += segments + 1
    return count


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
    # Slots past ngeom hold stale memory; mjr_render dereferences matid/dataid, so
    # an uninitialised slot segfaults the renderer (#11675). Reset it first.
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.array([0.01, 0.0, 0.0]),
        np.zeros(3),
        np.eye(3).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    _call_connector(geom, geom_type, width, from_pt, to_pt)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_ARROW):
        geom.size[2] /= ARROW_RENDER_LENGTH_FRACTION
    geom.rgba[:] = rgba
    scene.ngeom += 1
    return True


def _add_force_arrow(
    scene: mujoco.MjvScene, arrow: Any, head_shaft_radii: float | None
) -> tuple[int, int]:
    """Draw one force arrow; ``(added, dropped)`` geom counts.

    See :func:`add_glyphs_to_scene` for ``head_shaft_radii``. A clamped arrow
    gets a short white arrow continuing past its tip.
    """
    arrow_type = int(mujoco.mjtGeom.mjGEOM_ARROW)
    from_pt = np.asarray(arrow.tail_m, dtype=np.float64)
    to_pt = np.asarray(arrow.tip_m, dtype=np.float64)
    width = 2.0 * float(arrow.shaft_radius_m)
    pieces = []
    head_from = from_pt
    length = float(np.linalg.norm(to_pt - from_pt))
    span = None if head_shaft_radii is None else head_shaft_radii * width / 2.0
    if span is not None and length > span:
        head_from = to_pt - (to_pt - from_pt) / length * span
        capsule = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        pieces.append((capsule, from_pt, head_from, arrow.rgba))
    pieces.append((arrow_type, head_from, to_pt, arrow.rgba))
    if arrow.clamped and length > 0.0:
        extra = to_pt + (to_pt - from_pt) / length * CLAMPED_TIP_LENGTH_M
        pieces.append((arrow_type, to_pt, extra, (1.0, 1.0, 1.0, arrow.rgba[3])))
    added = sum(
        bool(_append_connector(scene, kind, width, a, b, rgba))
        for kind, a, b, rgba in pieces
    )
    return added, len(pieces) - added


def add_glyphs_to_scene(
    scene: mujoco.MjvScene,
    glyphs: GlyphSet,
    *,
    arc_width_m: float = 0.006,
    arrow_head_shaft_radii: float | None = None,
) -> SceneGlyphReceipt:
    """Draw a GlyphSet into a MuJoCo MjvScene as 3D arrows and torque arcs.

    Must be called after `mjv_updateScene` and before `mjr_render` or
    `mujoco.Renderer.render()`.

    Args:
        scene: Target MuJoCo scene whose pre-allocated geoms buffer is populated.
        glyphs: Deterministic GlyphSet containing arrows and torque arcs.
        arc_width_m: Diameter in meters of the torque arc polyline capsules.
        arrow_head_shaft_radii: When set, the head length of each force arrow
            is ``span = arrow_head_shaft_radii * shaft_radius_m``; an arrow
            longer than ``span`` is drawn as a capsule shaft up to ``span``
            before the tip plus an arrow geom of the same width over the last
            ``span`` metres, so the head keeps one size instead of growing
            into a long spike. ``None`` (the default) draws one arrow geom.

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
    if arrow_head_shaft_radii is not None and not (
        np.isfinite(arrow_head_shaft_radii) and arrow_head_shaft_radii > 0.0
    ):
        raise ValueError(
            "arrow head length in shaft radii must be positive and finite, "
            f"got {arrow_head_shaft_radii}"
        )

    added = 0
    dropped = 0
    arrow_type = int(mujoco.mjtGeom.mjGEOM_ARROW)
    capsule_type = int(mujoco.mjtGeom.mjGEOM_CAPSULE)

    # 1. Force Arrows: an arrow geom (with a capsule shaft for a fixed head)
    for arrow in glyphs.arrows:
        ok, bad = _add_force_arrow(scene, arrow, arrow_head_shaft_radii)
        added, dropped = added + ok, dropped + bad

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


def add_capacity_wire_to_scene(
    scene: mujoco.MjvScene,
    wireframes: Sequence[CapacityWireGlyph] | Sequence[LineSegmentGlyph],
    *,
    rim_samples: int = 32,
) -> SceneGlyphReceipt:
    """Draw capacity ellipsoids as line primitives (CAP-5b, #14045)."""
    if not isinstance(scene, mujoco.MjvScene):
        raise TypeError(f"scene must be a mujoco.MjvScene, got {type(scene).__name__}")
    segments: list[LineSegmentGlyph] = []
    for item in wireframes:
        if isinstance(item, CapacityWireGlyph):
            segments.extend(capacity_wire_segments(item, rim_samples=rim_samples))
        elif isinstance(item, LineSegmentGlyph):
            segments.append(item)
        else:
            raise TypeError(
                "wireframes must be CapacityWireGlyph or LineSegmentGlyph, "
                f"got {type(item).__name__}"
            )
    capsule_type = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    added = dropped = 0
    for seg in segments:
        from_pt = np.asarray(seg.start, dtype=np.float64)
        to_pt = np.asarray(seg.end, dtype=np.float64)
        width = 2.0 * float(seg.width_m)
        if _append_connector(scene, capsule_type, width, from_pt, to_pt, seg.rgba):
            added += 1
        else:
            dropped += 1
    return SceneGlyphReceipt(added=added, dropped=dropped)


def add_ellipsoids_to_scene(
    scene: mujoco.MjvScene, ellipsoids: Sequence[EllipsoidGlyph]
) -> SceneGlyphReceipt:
    """Draw oriented ellipsoids as ``mjGEOM_ELLIPSOID`` geoms (#14003).

    ``size`` is the glyph radii and ``mat`` is the row-major flatten of the
    rotation matrix whose columns are the ellipsoid axes (MuJoCo's geom
    ``xmat`` convention). Geoms beyond ``scene.maxgeom`` are dropped and
    counted, never written. Call after ``mjv_updateScene``.
    """
    if not isinstance(scene, mujoco.MjvScene):
        raise TypeError(f"scene must be a mujoco.MjvScene, got {type(scene).__name__}")
    added = dropped = 0
    for glyph in ellipsoids:
        if not isinstance(glyph, EllipsoidGlyph):
            raise TypeError(
                f"ellipsoids must be EllipsoidGlyph, got {type(glyph).__name__}"
            )
        if scene.ngeom >= scene.maxgeom:
            dropped += 1
            continue
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
            np.asarray(glyph.radii, dtype=np.float64),
            np.asarray(glyph.centre, dtype=np.float64),
            np.asarray(glyph.axes, dtype=np.float64).ravel(),
            np.asarray(glyph.rgba, dtype=np.float32),
        )
        scene.ngeom += 1
        added += 1
    return SceneGlyphReceipt(added=added, dropped=dropped)


def add_ball_to_scene(
    scene: mujoco.MjvScene,
    position_m: npt.ArrayLike,
    radius_m: float,
    rgba: tuple[float, float, float, float] = (0.97, 0.97, 0.95, 1.0),
) -> bool:
    """Draw a decorative sphere (the golf ball) into a MuJoCo render scene.

    The sphere lives only in ``scene`` (the render buffer): it is not part of
    the model, so mass, bodies and dynamics are unchanged (GCV-13, #11719).
    Returns ``False`` when the scene buffer is full.

    Preconditions: ``position_m`` is a finite 3-vector and ``radius_m`` is
    positive and finite (``ValueError``).
    """
    if not isinstance(scene, mujoco.MjvScene):
        raise TypeError(f"scene must be a mujoco.MjvScene, got {type(scene).__name__}")
    pos = np.asarray(position_m, dtype=np.float64)
    if pos.shape != (3,) or not np.isfinite(pos).all():
        raise ValueError("position_m must be a finite 3-vector")
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError(f"radius_m must be positive and finite, got {radius_m}")
    if scene.ngeom >= scene.maxgeom:
        return False
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([radius_m, 0.0, 0.0]),
        pos,
        np.eye(3).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return True
