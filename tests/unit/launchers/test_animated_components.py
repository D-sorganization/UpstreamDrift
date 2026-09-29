from PyQt6.QtGui import QColor, QEnterEvent
from PyQt6.QtCore import QEvent, QPointF
from src.launchers.animated_components import AnimatedButton


def test_animated_button_initialization(qapp):
    """Test that AnimatedButton initializes with default colors and creates animation object."""
    btn = AnimatedButton("Test")
    assert btn.text() == "Test"
    assert btn._base_color == QColor("#0A84FF")
    assert btn._hover_color == QColor("#0077E6")
    assert hasattr(btn, "_hover_anim")
    assert btn._hover_anim.duration() == 150


def test_animated_button_hover_events(qapp):
    """Test that enterEvent and leaveEvent trigger color transitions."""
    btn = AnimatedButton()

    # Simulate hover enter
    event_enter = QEnterEvent(QPointF(0, 0), QPointF(0, 0), QPointF(0, 0))
    btn.enterEvent(event_enter)
    # The animation should be running or finished
    assert btn._hover_anim.endValue() == btn._hover_color

    # Simulate hover leave
    event_leave = QEvent(QEvent.Type.Leave)
    btn.leaveEvent(event_leave)
    assert btn._hover_anim.endValue() == btn._base_color


def test_animated_button_color_changed(qapp):
    """Test the internal _on_color_changed method updates stylesheet."""
    btn = AnimatedButton()
    test_color = QColor("#ff0000")
    btn._on_color_changed(test_color)
    assert btn._current_color == test_color
    assert test_color.name() in btn.styleSheet()
