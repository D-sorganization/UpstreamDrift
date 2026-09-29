from __future__ import annotations

import sys

import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication, QMainWindow

from src.launchers.launcher_ui.frameless_window import (
    FramelessResizeFilter,
    configure_frameless_window,
)


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def _mouse_event(
    event_type: QEvent.Type,
    global_pos: QPoint,
    *,
    button: Qt.MouseButton = Qt.MouseButton.NoButton,
    buttons: Qt.MouseButton = Qt.MouseButton.NoButton,
) -> QMouseEvent:
    return QMouseEvent(
        event_type,
        QPointF(global_pos),
        QPointF(global_pos),
        button,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )


def test_configure_frameless_window_applies_chrome_and_keeps_filter_owned(
    qapp: QApplication,
) -> None:
    window = QMainWindow()

    resize_filter = configure_frameless_window(window)

    assert isinstance(resize_filter, FramelessResizeFilter)
    assert resize_filter.parent() is window
    assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert window.windowFlags() & Qt.WindowType.WindowMinimizeButtonHint
    assert window.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


def test_resize_filter_resizes_from_right_edge(qapp: QApplication) -> None:
    window = QMainWindow()
    window.setGeometry(100, 100, 100, 100)
    window.setMinimumSize(50, 50)
    resize_filter = FramelessResizeFilter(window)
    right_edge = window.mapToGlobal(QPoint(99, 50))

    hover = _mouse_event(QEvent.Type.MouseMove, right_edge)
    assert resize_filter.eventFilter(window, hover) is True
    assert window.cursor().shape() == Qt.CursorShape.SizeHorCursor

    press = _mouse_event(
        QEvent.Type.MouseButtonPress,
        right_edge,
        button=Qt.MouseButton.LeftButton,
        buttons=Qt.MouseButton.LeftButton,
    )
    assert resize_filter.eventFilter(window, press) is True

    drag = _mouse_event(
        QEvent.Type.MouseMove,
        right_edge + QPoint(20, 0),
        buttons=Qt.MouseButton.LeftButton,
    )
    assert resize_filter.eventFilter(window, drag) is True
    assert window.geometry().width() == 120

    release = _mouse_event(QEvent.Type.MouseButtonRelease, right_edge + QPoint(20, 0))
    assert resize_filter.eventFilter(window, release) is True
    assert window.cursor().shape() == Qt.CursorShape.ArrowCursor
