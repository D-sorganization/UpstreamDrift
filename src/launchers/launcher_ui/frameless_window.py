"""Frameless launcher window helpers."""

from __future__ import annotations

from typing import Any, cast

from PyQt6.QtCore import QEvent, QObject, QRect, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow


class FramelessResizeFilter(QObject):
    def __init__(self, window: QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self._resizing = False
        self._resize_edge = 0
        self._start_pos = None
        self._start_geo: QRect | None = None

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        if event is None:
            return False
        typed_event = cast(Any, event)
        if event.type() in (
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.HoverMove,
        ):
            if hasattr(event, "globalPosition"):
                gpos = event.globalPosition().toPoint()
            elif hasattr(event, "globalPos"):
                gpos = event.globalPos()
            else:
                return super().eventFilter(obj, event)

            local_pos = self.window.mapFromGlobal(gpos)
            x, y = local_pos.x(), local_pos.y()
            w, h = self.window.width(), self.window.height()
            border = 8

            if not self._resizing:
                if 0 <= x <= w and 0 <= y <= h:
                    edge = 0
                    if x < border and y < border:
                        edge = 13
                    elif x > w - border and y < border:
                        edge = 14
                    elif x < border and y > h - border:
                        edge = 16
                    elif x > w - border and y > h - border:
                        edge = 17
                    elif x < border:
                        edge = 10
                    elif x > w - border:
                        edge = 11
                    elif y < border:
                        edge = 12
                    elif y > h - border:
                        edge = 15

                    if edge != 0:
                        if (
                            event.type() == QEvent.Type.MouseButtonPress
                            and typed_event.button() == Qt.MouseButton.LeftButton
                        ):
                            self._resizing = True
                            self._resize_edge = edge
                            self._start_pos = gpos
                            self._start_geo = self.window.geometry()
                            return True
                        if event.type() in (
                            QEvent.Type.HoverMove,
                            QEvent.Type.MouseMove,
                        ):
                            if edge in (13, 17):
                                self.window.setCursor(Qt.CursorShape.SizeFDiagCursor)
                            elif edge in (14, 16):
                                self.window.setCursor(Qt.CursorShape.SizeBDiagCursor)
                            elif edge in (10, 11):
                                self.window.setCursor(Qt.CursorShape.SizeHorCursor)
                            elif edge in (12, 15):
                                self.window.setCursor(Qt.CursorShape.SizeVerCursor)
                            return True
                    else:
                        if self.window.cursor().shape() != Qt.CursorShape.ArrowCursor:
                            self.window.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                if event.type() == QEvent.Type.MouseMove:
                    if self._start_geo is None:
                        return False
                    delta = gpos - self._start_pos
                    rect = QRect(self._start_geo)
                    if self._resize_edge in (10, 13, 16):
                        rect.setLeft(rect.left() + delta.x())
                    if self._resize_edge in (11, 14, 17):
                        rect.setRight(rect.right() + delta.x())
                    if self._resize_edge in (12, 13, 14):
                        rect.setTop(rect.top() + delta.y())
                    if self._resize_edge in (15, 16, 17):
                        rect.setBottom(rect.bottom() + delta.y())

                    if (
                        rect.width() >= self.window.minimumWidth()
                        and rect.height() >= self.window.minimumHeight()
                    ):
                        self.window.setGeometry(rect)
                    return True
                if event.type() == QEvent.Type.MouseButtonRelease:
                    self._resizing = False
                    self.window.setCursor(Qt.CursorShape.ArrowCursor)
                    return True
        return super().eventFilter(obj, event)


def apply_frameless_window_chrome(window: QMainWindow) -> None:
    """Apply the launcher frameless-window flags and translucent background."""
    window.setWindowFlags(
        Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowMinimizeButtonHint
    )
    window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


def install_frameless_resize_filter(window: QMainWindow) -> FramelessResizeFilter:
    """Install the application-level resize filter and return it for ownership."""
    resize_filter = FramelessResizeFilter(window)
    app = QApplication.instance()
    if app is not None:
        app.installEventFilter(resize_filter)
    return resize_filter


def configure_frameless_window(window: QMainWindow) -> FramelessResizeFilter:
    """Apply frameless chrome and install the resize filter."""
    apply_frameless_window_chrome(window)
    return install_frameless_resize_filter(window)
