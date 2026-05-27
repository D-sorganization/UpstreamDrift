from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QMouseEvent
from unittest.mock import MagicMock
from src.launchers.custom_title_bar import CustomTitleBar


def test_custom_title_bar_signals(qapp):
    """Test that window control buttons emit correct signals."""
    title_bar = CustomTitleBar()

    minimize_mock = MagicMock()
    maximize_mock = MagicMock()
    close_mock = MagicMock()

    title_bar.minimize_requested.connect(minimize_mock)
    title_bar.maximize_requested.connect(maximize_mock)
    title_bar.close_requested.connect(close_mock)

    title_bar._minimize_window()
    minimize_mock.assert_called_once()

    title_bar._maximize_window()
    maximize_mock.assert_called_once()

    title_bar._close_window()
    close_mock.assert_called_once()


def test_custom_title_bar_mouse_events(qapp):
    """Test dragging functionality emits move_requested."""
    title_bar = CustomTitleBar()

    # Create a parent window so frameGeometry() is valid
    from PyQt6.QtWidgets import QWidget

    parent = QWidget()
    title_bar.setParent(parent)

    move_mock = MagicMock()
    title_bar.move_requested.connect(move_mock)

    # Press event
    press_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(10, 10),
        QPointF(10, 10),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    title_bar.mousePressEvent(press_event)

    # Move event
    move_event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(20, 20),
        QPointF(20, 20),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    title_bar.mouseMoveEvent(move_event)

    move_mock.assert_called_once()
