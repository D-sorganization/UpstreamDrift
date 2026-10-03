# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Reusable force/torque vector overlay for the motion canvases.

Pure Qt drawing helpers shared by the swingset and chain :class:`MotionCanvas`.
Vectors are supplied in **world** coordinates; both endpoints are projected
through the canvas's own projector so the screen Y-flip is handled
automatically -- no pixel deltas are computed in world space.

Design Principles:
    DBC -- drawing entry points validate their preconditions (ValueError).
    LoD -- no physics imports; operates only on geometry + a projector.
    DRY -- a single arrow primitive serves every force type.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QPainter, QPen

Projector = Callable[[tuple[float, float]], QPointF]

_ARROWHEAD_ANGLE_RAD = math.radians(28.0)
_MIN_SHAFT_PX = 1.5


@dataclass(frozen=True)
class VectorStyle:
    """Visual style for a drawn vector."""

    color: QColor
    width: int = 2
    head_px: float = 9.0
    label: str = ""


@dataclass(frozen=True)
class ForceArrow:
    """A single force arrow in world coordinates.

    ``vector_m`` is the world-frame displacement to add to ``origin_m`` (after
    scaling) to obtain the arrow tip. Its magnitude drives auto-scaling.
    """

    origin_m: tuple[float, float]
    vector_m: tuple[float, float]
    style: VectorStyle


@dataclass(frozen=True)
class TorqueArc:
    """A signed torque indicator drawn as a curved arrow at a joint."""

    center_m: tuple[float, float]
    magnitude_nm: float
    style: VectorStyle
    radius_px: float = 16.0


@dataclass(frozen=True)
class ComMarker:
    """A centre-of-mass marker (filled dot + crosshair)."""

    point_m: tuple[float, float]
    style: VectorStyle
    radius_px: float = 5.0


@dataclass(frozen=True)
class OverlayScene:
    """A bundle of overlay primitives toggled together by the canvas."""

    arrows: tuple[ForceArrow, ...] = ()
    torque_arcs: tuple[TorqueArc, ...] = ()
    com_markers: tuple[ComMarker, ...] = field(default=())


def _magnitude(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def auto_scale_factor(arrows: Sequence[ForceArrow], target_world_len: float) -> float:
    """Return a scale so the largest arrow spans about ``target_world_len``.

    Returns ``1.0`` when there are no arrows or all arrows are zero-length.

    Preconditions:
        ``target_world_len`` is positive.
    """
    if target_world_len <= 0.0:
        raise ValueError("target_world_len must be positive")
    largest = max((_magnitude(arrow.vector_m) for arrow in arrows), default=0.0)
    if largest <= 0.0:
        return 1.0
    return target_world_len / largest


def arrows_to_force_torque_frame(arrows: Sequence[ForceArrow], scale: float) -> Any:
    """Convert movement optimizer ForceArrows to a ForceTorqueFrame."""
    from src.shared.python.force_overlay.contracts import (
        ForceTorqueFrame,
        OverlayWrench,
        WrenchKind,
    )

    wrenches = []
    for i, arrow in enumerate(arrows):
        vx, vy = arrow.vector_m
        if math.hypot(vx, vy) <= 0.0:
            continue
        ox, oy = arrow.origin_m
        wrenches.append(
            OverlayWrench(
                kind=WrenchKind.EXTERNAL,
                label=f"arrow:{i}",
                body="canvas",
                point_m=(ox, oy, 0.0),
                force_n=(vx * scale, vy * scale, 0.0),
                torque_nm=None,
                source="vector_overlay",
            )
        )
    return ForceTorqueFrame(time_s=0.0, engine="vector_overlay", wrenches=tuple(wrenches))


def draw_force_arrows(
    painter: QPainter,
    projector: Projector,
    arrows: Sequence[ForceArrow],
    *,
    scale: float,
) -> None:
    """Draw force arrows by projecting origin and scaled tip through ``projector``.

    Preconditions:
        ``scale`` is positive and finite.
    """
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be a positive, finite number")
    from src.shared.python.force_overlay.contracts import WrenchKind
    from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, build_glyphs
    from src.shared.python.force_overlay.renderers.qpainter_glyphs import (
        draw_glyphs_2d,
    )

    frame = arrows_to_force_torque_frame(arrows, scale)
    if not frame.wrenches:
        return
    palette: dict[str, str] = (
        {WrenchKind.EXTERNAL.value: str(arrows[0].style.color.name())} if arrows else {}
    )
    style = ForceGlyphStyle(
        force_scale_m_per_n=1.0,
        min_length_m=1e-4,
        max_length_m=1e5,
        palette=palette,
    )
    glyphs = build_glyphs(frame, style=style)
    draw_glyphs_2d(
        painter,
        projector,
        glyphs,
        px_width=float(arrows[0].style.width) if arrows else 2.0,
        halo=False,
    )


def draw_torque_arcs(
    painter: QPainter,
    projector: Projector,
    arcs: Sequence[TorqueArc],
    *,
    reference_nm: float,
) -> None:
    """Draw signed torque indicators as curved arrows.

    Sweep angle scales with ``magnitude_nm / reference_nm`` (clamped) and the
    sweep direction encodes the torque sign.

    Preconditions:
        ``reference_nm`` is positive.
    """
    if reference_nm <= 0.0:
        raise ValueError("reference_nm must be positive")
    for arc in arcs:
        center = projector(arc.center_m)
        pen = QPen(arc.style.color, arc.style.width)
        painter.setPen(pen)
        fraction = max(-1.0, min(1.0, arc.magnitude_nm / reference_nm))
        span_deg = 270.0 * fraction
        rect_x = center.x() - arc.radius_px
        rect_y = center.y() - arc.radius_px
        diameter = 2.0 * arc.radius_px
        # drawArc angles are in 1/16th degrees, starting at 3 o'clock.
        painter.drawArc(
            round(rect_x),
            round(rect_y),
            round(diameter),
            round(diameter),
            0,
            round(span_deg * 16),
        )


def draw_com_markers(
    painter: QPainter,
    projector: Projector,
    markers: Sequence[ComMarker],
) -> None:
    """Draw centre-of-mass markers as a filled dot with a crosshair."""
    for marker in markers:
        center = projector(marker.point_m)
        pen = QPen(marker.style.color, marker.style.width)
        painter.setPen(pen)
        painter.setBrush(marker.style.color)
        radius = marker.radius_px
        painter.drawEllipse(center, radius, radius)
        reach = radius * 2.2
        painter.drawLine(
            QPointF(center.x() - reach, center.y()),
            QPointF(center.x() + reach, center.y()),
        )
        painter.drawLine(
            QPointF(center.x(), center.y() - reach),
            QPointF(center.x(), center.y() + reach),
        )


def draw_overlay_scene(
    painter: QPainter,
    projector: Projector,
    scene: OverlayScene,
    *,
    arrow_scale: float,
    torque_reference_nm: float,
) -> None:
    """Draw every primitive in ``scene`` (DRY entry point for the canvas).

    Preconditions:
        ``arrow_scale`` is positive/finite and ``torque_reference_nm`` positive.
    """
    if scene.arrows:
        draw_force_arrows(painter, projector, scene.arrows, scale=arrow_scale)
    if scene.torque_arcs:
        draw_torque_arcs(painter, projector, scene.torque_arcs, reference_nm=torque_reference_nm)
    if scene.com_markers:
        draw_com_markers(painter, projector, scene.com_markers)
