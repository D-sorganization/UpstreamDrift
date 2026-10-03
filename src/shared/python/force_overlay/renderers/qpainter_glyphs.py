"""Canonical QPainter 2D force and torque glyph renderer (ADR-0052, #11292)."""

from __future__ import annotations

import math
from typing import Any, Sequence

from src.shared.python.force_overlay.glyphs import ArrowGlyph, GlyphSet, TorqueArcGlyph

__all__ = ["draw_glyphs_2d"]


def _project_point(project: Any, pt: Sequence[float]) -> Any:
    """Project a 2D or 3D coordinate through project callable into a QPointF."""
    from PyQt6.QtCore import QPointF

    try:
        res = project(pt)
    except TypeError:
        try:
            res = project((pt[0], pt[1]))
        except TypeError:
            res = project(pt[0], pt[1])

    if isinstance(res, QPointF):
        return res
    return QPointF(float(res[0]), float(res[1]))


def _to_qcolor(rgba: tuple[float, ...]) -> Any:
    """Convert normalized RGBA tuple to PyQt6 QColor."""
    from PyQt6.QtGui import QColor

    return QColor(
        int(rgba[0] * 255),
        int(rgba[1] * 255),
        int(rgba[2] * 255),
        int(rgba[3] * 255) if len(rgba) > 3 else 255,
    )


def _draw_single_arrow_2d(
    painter: Any,
    project: Any,
    arrow: ArrowGlyph,
    px_width: float,
    halo: bool,
    halo_color: Any,
) -> None:
    """Render a single 2D force arrow using QPainter."""
    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QBrush, QPen, QPolygonF

    p_tail: QPointF = _project_point(project, arrow.tail_m)
    p_base: QPointF = _project_point(project, arrow.head_base_m)
    p_tip: QPointF = _project_point(project, arrow.tip_m)

    dx = p_tip.x() - p_base.x()
    dy = p_tip.y() - p_base.y()
    head_len = math.hypot(dx, dy)
    if head_len < 1e-4:
        dx = p_tip.x() - p_tail.x()
        dy = p_tip.y() - p_tail.y()
        head_len = max(1.0, math.hypot(dx, dy))

    ux, uy = dx / head_len, dy / head_len
    perp_x, perp_y = -uy, ux

    head_w = max(3.0, px_width * 2.2)
    w1 = QPointF(p_base.x() + head_w * perp_x, p_base.y() + head_w * perp_y)
    w2 = QPointF(p_base.x() - head_w * perp_x, p_base.y() - head_w * perp_y)
    tri = QPolygonF([p_tip, w1, w2])
    fg_col = _to_qcolor(arrow.rgba)

    if halo:
        h_pen = QPen(halo_color, px_width * 1.6)
        painter.setPen(h_pen)
        painter.drawLine(p_tail, p_base)
        painter.setBrush(QBrush(halo_color))
        h_w = head_w * 1.4
        hw1 = QPointF(p_base.x() + h_w * perp_x, p_base.y() + h_w * perp_y)
        hw2 = QPointF(p_base.x() - h_w * perp_x, p_base.y() - h_w * perp_y)
        painter.drawPolygon(QPolygonF([p_tip, hw1, hw2]))

    fg_pen = QPen(fg_col, px_width)
    painter.setPen(fg_pen)
    painter.drawLine(p_tail, p_base)
    painter.setBrush(QBrush(fg_col))
    painter.drawPolygon(tri)


def _draw_single_torque_arc_2d(
    painter: Any,
    project: Any,
    arc: TorqueArcGlyph,
    px_width: float,
    halo: bool,
    halo_color: Any,
) -> None:
    """Render a single 2D torque arc using QPainter."""
    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QBrush, QPen, QPolygonF

    pts = [_project_point(project, pt) for pt in arc.polyline_m]
    fg_col = _to_qcolor(arc.rgba)
    if len(pts) >= 2:
        if halo:
            h_pen = QPen(halo_color, px_width * 1.6)
            painter.setPen(h_pen)
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

        fg_pen = QPen(fg_col, px_width)
        painter.setPen(fg_pen)
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])

    p_base = _project_point(project, arc.head_base_m)
    p_tip = _project_point(project, arc.head_tip_m)
    dx = p_tip.x() - p_base.x()
    dy = p_tip.y() - p_base.y()
    h_len = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / h_len, dy / h_len
    perp_x, perp_y = -uy, ux

    head_w = max(3.0, px_width * 2.2)
    w1 = QPointF(p_base.x() + head_w * perp_x, p_base.y() + head_w * perp_y)
    w2 = QPointF(p_base.x() - head_w * perp_x, p_base.y() - head_w * perp_y)
    tri = QPolygonF([p_tip, w1, w2])

    if halo:
        painter.setPen(QPen(halo_color, 1))
        painter.setBrush(QBrush(halo_color))
        h_w = head_w * 1.4
        hw1 = QPointF(p_base.x() + h_w * perp_x, p_base.y() + h_w * perp_y)
        hw2 = QPointF(p_base.x() - h_w * perp_x, p_base.y() - h_w * perp_y)
        painter.drawPolygon(QPolygonF([p_tip, hw1, hw2]))

    painter.setPen(QPen(fg_col, 1))
    painter.setBrush(QBrush(fg_col))
    painter.drawPolygon(tri)


def draw_glyphs_2d(
    painter: Any,
    project: Any,
    glyphs: GlyphSet,
    *,
    px_width: float = 2.0,
    halo: bool = True,
) -> None:
    """Render 2D force arrows and torque arcs using QPainter.

    Parameters
    ----------
    painter
        Active QPainter instance.
    project
        Callable converting world coordinates to screen pixel coordinates (QPointF or (x, y)).
    glyphs
        Deterministic glyph set to draw.
    px_width
        Base stroke width for vector shafts.
    halo
        If True, renders dark anti-aliased underlays behind strokes.
    """
    if painter is None:
        raise ValueError("painter must not be None")
    if not isinstance(glyphs, GlyphSet):
        raise TypeError(f"glyphs must be GlyphSet, got {type(glyphs)}")

    from PyQt6.QtGui import QColor, QPainter

    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    halo_color = QColor(32, 32, 32, 160)

    for arrow in glyphs.arrows:
        _draw_single_arrow_2d(painter, project, arrow, px_width, halo, halo_color)
    for arc in glyphs.torque_arcs:
        _draw_single_torque_arc_2d(painter, project, arc, px_width, halo, halo_color)
