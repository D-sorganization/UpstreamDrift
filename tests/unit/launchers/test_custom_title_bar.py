from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QMouseEvent
from unittest.mock import MagicMock
from src.launchers.custom_title_bar import CustomTitleBar


def test_custom_title_bar_signals(qapp):
    """Test that window control buttons emit correct signals."""
    title_bar = CustomTitleBar()

    # Mock the emit methods
    title_bar.minimize_requested.emit = MagicMock()
    title_bar.maximize_requested.emit = MagicMock()
    title_bar.close_requested.emit = MagicMock()

    title_bar.minimize_requested.emit.reset_mock()
    title_bar._minimize_window()
    title_bar.minimize_requested.emit.assert_called_once()

    title_bar.maximize_requested.emit.reset_mock()
    title_bar._maximize_window()
    title_bar.maximize_requested.emit.assert_called_once()

    title_bar.close_requested.emit.reset_mock()
    title_bar._close_window()
    title_bar.close_requested.emit.assert_called_once()


def test_custom_title_bar_mouse_events(qapp):
    """Test dragging functionality emits move_requested."""
    title_bar = CustomTitleBar()

    # Create a parent window so frameGeometry() is valid
    from PyQt6.QtWidgets import QWidget

    parent = QWidget()
    title_bar.setParent(parent)

    title_bar.move_requested.emit = MagicMock()

    # Press event
    press_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPoint(10, 10),
        QPoint(10, 10),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    title_bar.mousePressEvent(press_event)

    # Move event
    move_event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPoint(20, 20),
        QPoint(20, 20),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    title_bar.mouseMoveEvent(move_event)

    title_bar.move_requested.emit.assert_called_once()
