"""Unit tests for the hardened toast notification system (Issue #8900)."""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent, Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from src.shared.python.ui.toast import Toast, ToastManager, ToastType  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def qt_app() -> Any:
    """Fixture providing a QApplication instance in offscreen mode."""
    try:
        app = QApplication.instance() or QApplication(sys.argv[:1])
    except (ImportError, OSError) as e:
        pytest.skip(f"PyQt6 runtime unavailable: {e}")
    yield app


def test_toast_type_icons_and_accessible_names(qt_app: Any) -> None:
    """Verify each toast type renders a distinct icon and accessible type prefix (WCAG 1.4.1)."""
    cases = [
        (ToastType.SUCCESS, "Success", "\u2713"),
        (ToastType.ERROR, "Error", "\u2717"),
        (ToastType.WARNING, "Warning", "\u26a0"),
        (ToastType.INFO, "Info", "\u2139"),
    ]

    for toast_type, expected_prefix, expected_icon in cases:
        toast = Toast("Test message", toast_type=toast_type)
        assert toast.icon_label.text() == expected_icon
        assert toast.icon_label.accessibleName() == f"{expected_prefix} icon"
        expected_acc_name = f"{expected_prefix}: Test message"
        assert toast.accessibleName() == expected_acc_name
        assert toast.label.accessibleName() == expected_acc_name
        toast.deleteLater()


def test_toast_window_flags_and_cursor(qt_app: Any) -> None:
    """Verify toasts do not stay on top of other applications and indicate clickability."""
    toast = Toast("Flags test")
    flags = toast.windowFlags()

    assert flags & Qt.WindowType.WindowStaysOnTopHint == 0, (
        "WindowStaysOnTopHint must NOT be set"
    )
    assert flags & Qt.WindowType.Tool != 0, "Tool flag must be set"
    assert flags & Qt.WindowType.FramelessWindowHint != 0, (
        "FramelessWindowHint must be set"
    )
    assert toast.cursor().shape() == Qt.CursorShape.PointingHandCursor, (
        "Cursor must indicate click-to-dismiss"
    )
    toast.deleteLater()


def test_toast_mouse_press_dismisses(qt_app: Any) -> None:
    """Verify clicking anywhere on the toast initiates dismissal."""
    toast = Toast("Click to dismiss")
    assert not toast._is_dismissing

    toast.mousePressEvent(None)
    assert toast._is_dismissing, "Toast must transition to dismissing on mouse press"
    toast.deleteLater()


def test_toast_hover_pauses_and_resumes_timer(qt_app: Any) -> None:
    """Verify mouse hover pauses the auto-dismiss timer and leave resumes it."""
    toast = Toast("Hover test", duration=4000)
    toast.show_animated()

    assert toast._dismiss_timer is not None
    assert toast._dismiss_timer.isActive(), "Dismiss timer must initially be active"

    # Hover in -> timer pauses
    toast.enterEvent(None)
    assert not toast._dismiss_timer.isActive(), "Timer must be paused while hovered"
    assert toast._remaining_time is not None and toast._remaining_time > 0

    saved_time = toast._remaining_time

    # Hover out -> timer resumes
    toast.leaveEvent(None)
    assert toast._dismiss_timer.isActive(), "Timer must resume after mouse leaves"
    assert toast._remaining_time is None

    toast.dismiss()
    toast.deleteLater()


def test_toast_manager_caps_visible_toasts(qt_app: Any) -> None:
    """Verify ToastManager caps visible toasts to MAX_VISIBLE_TOASTS (default 4)."""
    window = QMainWindow()
    window.setGeometry(100, 100, 800, 600)
    manager = ToastManager(window)

    assert manager.max_visible_toasts == 4

    toasts = [manager.show_info(f"Toast {i}") for i in range(6)]

    assert len(manager.active_toasts) == 4, "Must not exceed 4 active toasts"
    assert toasts[0]._is_dismissing, "Oldest toast must be dismissed"
    assert toasts[1]._is_dismissing, "Second oldest toast must be dismissed"
    assert manager.active_toasts == toasts[2:], "Remaining toasts must be [2, 3, 4, 5]"

    manager.dismiss_all()
    window.deleteLater()


def test_toast_manager_custom_cap(qt_app: Any) -> None:
    """Verify ToastManager respects a custom max_visible_toasts limit."""
    window = QMainWindow()
    window.setGeometry(100, 100, 800, 600)
    manager = ToastManager(window)
    manager.max_visible_toasts = 2

    toasts = [manager.show_info(f"Toast {i}") for i in range(4)]

    assert len(manager.active_toasts) == 2
    assert toasts[0]._is_dismissing
    assert toasts[1]._is_dismissing
    assert manager.active_toasts == toasts[2:]

    manager.dismiss_all()
    window.deleteLater()


def test_toast_manager_window_move_and_resize(qt_app: Any) -> None:
    """Verify toasts follow the parent window when moved or resized."""
    window = QMainWindow()
    window.setGeometry(100, 100, 800, 600)
    manager = ToastManager(window)

    t1 = manager.show_info("Toast 1")
    initial_x = t1.x()
    initial_y = t1.y()

    # Move parent window
    window.setGeometry(200, 250, 800, 600)
    manager.reposition_all()

    assert t1.x() == initial_x + 100
    assert t1.y() == initial_y + 150

    # Resize parent window
    window.setGeometry(200, 250, 1000, 700)
    manager.reposition_all()

    assert t1.x() == initial_x + 100 + 200
    assert t1.y() == initial_y + 150 + 100

    manager.dismiss_all()
    window.deleteLater()


def test_toast_manager_window_deactivation_hides_toasts(qt_app: Any) -> None:
    """Verify toasts hide when window deactivates and reshow when window activates."""
    window = QMainWindow()
    window.setGeometry(100, 100, 800, 600)
    window.show()

    manager = ToastManager(window)
    toast = manager.show_info("Deactivate test")

    assert toast.isVisible()

    # Deactivate event
    manager.eventFilter(window, QEvent(QEvent.Type.WindowDeactivate))
    assert not toast.isVisible(), "Toast must hide on window deactivation"
    assert toast._hidden_by_deactivate

    # Reactivate event
    manager.eventFilter(window, QEvent(QEvent.Type.WindowActivate))
    assert toast.isVisible(), "Toast must reappear on window activation"
    assert not toast._hidden_by_deactivate

    manager.dismiss_all()
    window.close()
    window.deleteLater()


def test_toast_manager_dismiss_all(qt_app: Any) -> None:
    """Verify dismiss_all dismisses all active toasts and clears the list."""
    window = QMainWindow()
    window.setGeometry(100, 100, 800, 600)
    manager = ToastManager(window)

    t1 = manager.show_success("Success")
    t2 = manager.show_error("Error")
    t3 = manager.show_warning("Warning")

    assert len(manager.active_toasts) == 3
    manager.dismiss_all()

    assert len(manager.active_toasts) == 0
    assert t1._is_dismissing
    assert t2._is_dismissing
    assert t3._is_dismissing

    window.deleteLater()


def test_toast_manager_validation(qt_app: Any) -> None:
    """Verify argument validation on Toast and ToastManager."""
    with pytest.raises(ValueError, match="parent must be provided"):
        ToastManager(None)

    with pytest.raises(ValueError, match="message must be provided"):
        Toast(None)  # type: ignore[arg-type]
