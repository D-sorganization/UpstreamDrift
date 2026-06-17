# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Packaged Qt icon helpers for Movement Optimizer."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

from ..rendering import Palette


def movement_optimizer_icon_path() -> Path:
    """Return the packaged launcher icon path.

    Preconditions:
        ``src/movement_optimizer/assets/project_map.svg`` is included with the app.
    """

    path = Path(__file__).resolve().parent.parent / "assets" / "project_map.svg"
    if not path.exists():
        raise FileNotFoundError(f"Movement Optimizer icon is missing: {path}")
    return path


def movement_optimizer_icon() -> QIcon:
    """Build a Qt icon matching the UpstreamDrift Movement Optimizer tile."""

    icon = QIcon(str(movement_optimizer_icon_path()))
    if not icon.isNull():
        return icon
    return QIcon(_fallback_project_map_pixmap())


def _fallback_project_map_pixmap() -> QPixmap:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    blue = QColor(Palette.BLUE)
    pale = QColor(Palette.ACCENT2)
    painter.setPen(QPen(blue, 2))
    for coordinate in (16, 32, 48):
        painter.drawLine(8, coordinate, 56, coordinate)
        painter.drawLine(coordinate, 8, coordinate, 56)
    painter.setPen(QPen(blue, 3))
    painter.drawLine(21, 21, 32, 32)
    painter.drawLine(43, 21, 32, 32)
    painter.drawLine(32, 32, 21, 43)
    painter.drawLine(32, 32, 43, 43)
    painter.setPen(Qt.PenStyle.NoPen)
    for x, y, radius, color in (
        (21, 21, 5, blue),
        (43, 21, 4, pale),
        (32, 32, 6, blue),
        (21, 43, 4, pale),
        (43, 43, 5, blue),
    ):
        painter.setBrush(color)
        painter.drawEllipse(x - radius, y - radius, 2 * radius, 2 * radius)
    painter.end()
    return pixmap
