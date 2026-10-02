"""Image-space landmark review without modifying stored source evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPixmap

from src.shared.python.theme.tool_stylesheet import get_tool_colors


def landmark_overlay(
    original: QPixmap,
    landmarks: Mapping[str, Mapping[str, Any]],
    *,
    pixel_coordinates: bool = False,
    native_model: bool = False,
) -> QPixmap:
    """Draw normalized XY on a detached image; missing landmarks draw nothing."""
    if original.isNull():
        raise ValueError("Source image must be decoded before landmark review")
    marked = original.copy()
    colors = get_tool_colors()
    painter = QPainter(marked)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    try:
        for point in landmarks.values():
            role = "text_secondary" if point["visibility"] is None else "text_primary"
            color = QColor(colors[role])
            painter.setPen(color)
            painter.setBrush(Qt.BrushStyle.NoBrush if native_model else color)
            painter.drawEllipse(
                QPointF(
                    float(point["x"]) * (1 if pixel_coordinates else marked.width()),
                    float(point["y"]) * (1 if pixel_coordinates else marked.height()),
                ),
                7 if native_model else 4,
                7 if native_model else 4,
            )
    finally:
        painter.end()
    return marked
