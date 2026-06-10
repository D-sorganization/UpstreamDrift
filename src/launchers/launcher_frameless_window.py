"""Frameless-window resize support for the UpstreamDrift launcher."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, cast

from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, Qt
from PyQt6.QtWidgets import QMainWindow


class ResizeEdge(IntEnum):
    """Resize edge identifiers preserved from the legacy launcher filter."""

    LEFT = 10
    RIGHT = 11
    TOP = 12
    TOP_LEFT = 13
    TOP_RIGHT = 14
    BOTTOM = 15
    BOTTOM_LEFT = 16
    BOTTOM_RIGHT = 17


def resize_edge_at(
    x: int,
    y: int,
    *,
    width: int,
    height: int,
    border: int = 8,
) -> ResizeEdge | None:
    """Return the resize edge for a point inside the frameless window."""
    if x < 0 or y < 0 or x > width or y > height:
        return None
    if x < border and y < border:
        return ResizeEdge.TOP_LEFT
    if x > width - border and y < border:
        return ResizeEdge.TOP_RIGHT
    if x < border and y > height - border:
        return ResizeEdge.BOTTOM_LEFT
    if x > width - border and y > height - border:
        return ResizeEdge.BOTTOM_RIGHT
    if x < border:
        return ResizeEdge.LEFT
    if x > width - border:
        return ResizeEdge.RIGHT
    if y < border:
        return ResizeEdge.TOP
    if y > height - border:
        return ResizeEdge.BOTTOM
    return None


def cursor_shape_for_edge(edge: ResizeEdge) -> Qt.CursorShape:
    """Map a resize edge to the cursor shape used by the legacy launcher."""
    if edge in (ResizeEdge.TOP_LEFT, ResizeEdge.BOTTOM_RIGHT):
        return Qt.CursorShape.SizeFDiagCursor
    if edge in (ResizeEdge.TOP_RIGHT, ResizeEdge.BOTTOM_LEFT):
        return Qt.CursorShape.SizeBDiagCursor
    if edge in (ResizeEdge.LEFT, ResizeEdge.RIGHT):
        return Qt.CursorShape.SizeHorCursor
    return Qt.CursorShape.SizeVerCursor


def resized_geometry(
    start_geometry: QRect,
    edge: ResizeEdge,
    delta: QPoint,
    *,
    minimum_width: int,
    minimum_height: int,
) -> QRect | None:
    """Return resized geometry, or None when the minimum size would be violated."""
    rect = QRect(start_geometry)
    if edge in (ResizeEdge.LEFT, ResizeEdge.TOP_LEFT, ResizeEdge.BOTTOM_LEFT):
        rect.setLeft(rect.left() + delta.x())
    if edge in (ResizeEdge.RIGHT, ResizeEdge.TOP_RIGHT, ResizeEdge.BOTTOM_RIGHT):
        rect.setRight(rect.right() + delta.x())
    if edge in (ResizeEdge.TOP, ResizeEdge.TOP_LEFT, ResizeEdge.TOP_RIGHT):
        rect.setTop(rect.top() + delta.y())
    if edge in (ResizeEdge.BOTTOM, ResizeEdge.BOTTOM_LEFT, ResizeEdge.BOTTOM_RIGHT):
        rect.setBottom(rect.bottom() + delta.y())
    if rect.width() < minimum_width or rect.height() < minimum_height:
        return None
    return rect


class FramelessResizeFilter(QObject):
    """Application-level event filter for resizing the frameless launcher window."""

    def __init__(self, window: QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self._resizing = False
        self._resize_edge: ResizeEdge | None = None
        self._start_pos: QPoint | None = None
        self._start_geo: QRect | None = None

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        if event is None:
            return False
        if event.type() not in (
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.HoverMove,
        ):
            return super().eventFilter(obj, event)

        gpos = self._global_position(event)
        if gpos is None:
            return super().eventFilter(obj, event)

        if self._resizing:
            return self._handle_resize_drag(obj, event, gpos)
        return self._handle_edge_hover_or_press(obj, event, gpos)

    def _global_position(self, event: QEvent) -> QPoint | None:
        if hasattr(event, "globalPosition"):
            return cast(Any, event).globalPosition().toPoint()
        if hasattr(event, "globalPos"):
            return cast(Any, event).globalPos()
        return None

    def _handle_edge_hover_or_press(
        self,
        obj: QObject | None,
        event: QEvent,
        gpos: QPoint,
    ) -> bool:
        local_pos = self.window.mapFromGlobal(gpos)
        edge = resize_edge_at(
            local_pos.x(),
            local_pos.y(),
            width=self.window.width(),
            height=self.window.height(),
        )
        if edge is None:
            if self.window.cursor().shape() != Qt.CursorShape.ArrowCursor:
                self.window.setCursor(Qt.CursorShape.ArrowCursor)
            return super().eventFilter(obj, event)

        if (
            event.type() == QEvent.Type.MouseButtonPress
            and cast(Any, event).button() == Qt.MouseButton.LeftButton
        ):
            self._resizing = True
            self._resize_edge = edge
            self._start_pos = gpos
            self._start_geo = self.window.geometry()
            return True

        if event.type() in (QEvent.Type.HoverMove, QEvent.Type.MouseMove):
            self.window.setCursor(cursor_shape_for_edge(edge))
            return True
        return super().eventFilter(obj, event)

    def _handle_resize_drag(
        self,
        obj: QObject | None,
        event: QEvent,
        gpos: QPoint,
    ) -> bool:
        if event.type() == QEvent.Type.MouseButtonRelease:
            self._resizing = False
            self._resize_edge = None
            self._start_pos = None
            self._start_geo = None
            self.window.setCursor(Qt.CursorShape.ArrowCursor)
            return True

        if event.type() != QEvent.Type.MouseMove:
            return super().eventFilter(obj, event)
        if (
            self._start_geo is None
            or self._start_pos is None
            or self._resize_edge is None
        ):
            return False

        rect = resized_geometry(
            self._start_geo,
            self._resize_edge,
            gpos - self._start_pos,
            minimum_width=self.window.minimumWidth(),
            minimum_height=self.window.minimumHeight(),
        )
        if rect is not None:
            self.window.setGeometry(rect)
        return True
