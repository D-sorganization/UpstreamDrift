"""Toast notification system for Golf Modeling Suite.

Provides non-blocking, auto-dismissing notifications that appear
at the bottom-right of the application window.

Usage:
    from shared.python.ui.toast import ToastManager, ToastType

    # Create manager attached to main window
    toast_manager = ToastManager(main_window)

    # Show notifications
    toast_manager.show_success("Operation completed!")
    toast_manager.show_error("Something went wrong")
    toast_manager.show_warning("Check your settings")
    toast_manager.show_info("Did you know...")
"""

from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QPropertyAnimation, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QLabel, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QMainWindow

# Import theme colors if available
try:
    from shared.python.theme import Colors, Sizes, Weights, get_qfont

    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False


class ToastType(Enum):
    """Types of toast notifications."""

    SUCCESS = auto()
    ERROR = auto()
    WARNING = auto()
    INFO = auto()


class Toast(QWidget):
    """Individual toast notification widget.

    A rounded, styled notification that fades in, stays visible,
    then fades out automatically.
    """

    # Toast dimensions
    MIN_WIDTH = 280
    MAX_WIDTH = 400
    PADDING = 16
    BORDER_RADIUS = 10

    # Animation durations (ms)
    FADE_IN_DURATION = 200
    FADE_OUT_DURATION = 300

    def __init__(
        self,
        message: str,
        toast_type: ToastType = ToastType.INFO,
        duration: int = 4000,
        parent: QWidget | None = None,
    ) -> None:
        """Create a toast notification.

        Args:
            message: Text to display
            toast_type: Type of notification (affects color/icon)
            duration: How long to show (ms), 0 for persistent
            parent: Parent widget
        """
        super().__init__(parent)
        self.message = message
        self.toast_type = toast_type
        self.duration = duration

        # Set up widget
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # Create opacity effect for fade animations
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self.opacity_effect)

        # Set up layout
        self._setup_ui()

        # Animations
        self._fade_in_anim: QPropertyAnimation | None = None
        self._fade_out_anim: QPropertyAnimation | None = None
        self._dismiss_timer: QTimer | None = None

    def _setup_ui(self) -> None:
        """Set up the toast UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            self.PADDING, self.PADDING, self.PADDING, self.PADDING
        )

        # Message label
        self.label = QLabel(self.message)
        self.label.setWordWrap(True)
        self.label.setMinimumWidth(self.MIN_WIDTH - 2 * self.PADDING)
        self.label.setMaximumWidth(self.MAX_WIDTH - 2 * self.PADDING)

        # Style based on theme availability
        if THEME_AVAILABLE:
            font = get_qfont(size=Sizes.SM, weight=Weights.MEDIUM)
        else:
            font = QFont("Segoe UI", 9)
            font.setWeight(QFont.Weight.Medium)
        self.label.setFont(font)
        self.label.setStyleSheet("color: #FFFFFF; background: transparent;")

        layout.addWidget(self.label)

        # Adjust size to content
        self.adjustSize()

    def _get_background_color(self) -> QColor:
        """Get background color based on toast type."""
        if THEME_AVAILABLE:
            colors = {
                ToastType.SUCCESS: Colors.SUCCESS,
                ToastType.ERROR: Colors.ERROR,
                ToastType.WARNING: Colors.WARNING,
                ToastType.INFO: Colors.BG_ELEVATED,
            }
        else:
            colors = {
                ToastType.SUCCESS: "#30D158",
                ToastType.ERROR: "#FF375F",
                ToastType.WARNING: "#FF9F0A",
                ToastType.INFO: "#2D2D2D",
            }
        return QColor(colors.get(self.toast_type, colors[ToastType.INFO]))

    def _get_icon(self) -> str:
        """Get icon character for toast type."""
        icons = {
            ToastType.SUCCESS: "\u2713",  # Checkmark
            ToastType.ERROR: "\u2717",  # X mark
            ToastType.WARNING: "\u26a0",  # Warning triangle
            ToastType.INFO: "\u2139",  # Info circle
        }
        return icons.get(self.toast_type, "")

    def paintEvent(self, event: Any) -> None:
        """Custom paint for rounded background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw rounded rectangle background
        path = QPainterPath()
        path.addRoundedRect(
            0.0,
            0.0,
            float(self.width()),
            float(self.height()),
            self.BORDER_RADIUS,
            self.BORDER_RADIUS,
        )

        painter.fillPath(path, self._get_background_color())
        painter.end()

    def show_animated(self) -> None:
        """Show the toast with fade-in animation."""
        self.show()

        # Fade in
        self._fade_in_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self._fade_in_anim.setDuration(self.FADE_IN_DURATION)
        self._fade_in_anim.setStartValue(0.0)
        self._fade_in_anim.setEndValue(1.0)
        self._fade_in_anim.start()

        # Set up auto-dismiss if duration > 0
        if self.duration > 0:
            self._dismiss_timer = QTimer(self)
            self._dismiss_timer.setSingleShot(True)
            self._dismiss_timer.timeout.connect(self.dismiss)
            self._dismiss_timer.start(self.duration)

    def dismiss(self) -> None:
        """Dismiss the toast with fade-out animation."""
        if self._dismiss_timer:
            self._dismiss_timer.stop()

        self._fade_out_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self._fade_out_anim.setDuration(self.FADE_OUT_DURATION)
        self._fade_out_anim.setStartValue(1.0)
        self._fade_out_anim.setEndValue(0.0)
        self._fade_out_anim.finished.connect(self._on_fade_out_finished)
        self._fade_out_anim.start()

    def _on_fade_out_finished(self) -> None:
        """Handle fade-out completion."""
        self.hide()
        self.deleteLater()


class ToastManager:
    """Manages toast notifications for an application window.

    Handles positioning, stacking, and lifecycle of toast notifications.
    """

    # Spacing between toasts
    TOAST_SPACING = 10
    MARGIN_RIGHT = 20
    MARGIN_BOTTOM = 20

    def __init__(self, parent: "QMainWindow") -> None:
        """Create a toast manager.

        Args:
            parent: Main window to attach toasts to
        """
        self.parent = parent
        self.active_toasts: list[Toast] = []

    def _calculate_position(self, toast: Toast) -> tuple[int, int]:
        """Calculate position for a new toast.

        Args:
            toast: Toast to position

        Returns:
            Tuple of (x, y) coordinates
        """
        parent_rect = self.parent.geometry()

        # Start from bottom-right
        x = parent_rect.right() - toast.width() - self.MARGIN_RIGHT
        y = parent_rect.bottom() - toast.height() - self.MARGIN_BOTTOM

        # Stack above existing toasts
        for existing in self.active_toasts:
            if existing.isVisible():
                y -= existing.height() + self.TOAST_SPACING

        return x, y

    def _show_toast(self, message: str, toast_type: ToastType, duration: int) -> Toast:
        """Create and show a toast notification.

        Args:
            message: Text to display
            toast_type: Type of notification
            duration: Display duration in ms

        Returns:
            The created Toast widget
        """
        toast = Toast(message, toast_type, duration, None)

        # Position the toast
        x, y = self._calculate_position(toast)
        toast.move(x, y)

        # Track the toast
        self.active_toasts.append(toast)

        # Clean up when dismissed
        def on_destroyed() -> None:
            if toast in self.active_toasts:
                self.active_toasts.remove(toast)

        toast.destroyed.connect(on_destroyed)

        # Show with animation
        toast.show_animated()

        return toast

    def show_success(self, message: str, duration: int = 3000) -> Toast:
        """Show a success notification.

        Args:
            message: Success message
            duration: Display duration in ms

        Returns:
            The created Toast
        """
        return self._show_toast(message, ToastType.SUCCESS, duration)

    def show_error(self, message: str, duration: int = 5000) -> Toast:
        """Show an error notification.

        Args:
            message: Error message
            duration: Display duration in ms (longer for errors)

        Returns:
            The created Toast
        """
        return self._show_toast(message, ToastType.ERROR, duration)

    def show_warning(self, message: str, duration: int = 4000) -> Toast:
        """Show a warning notification.

        Args:
            message: Warning message
            duration: Display duration in ms

        Returns:
            The created Toast
        """
        return self._show_toast(message, ToastType.WARNING, duration)

    def show_info(self, message: str, duration: int = 3500) -> Toast:
        """Show an info notification.

        Args:
            message: Info message
            duration: Display duration in ms

        Returns:
            The created Toast
        """
        return self._show_toast(message, ToastType.INFO, duration)

    def dismiss_all(self) -> None:
        """Dismiss all active toasts."""
        for toast in self.active_toasts[:]:
            toast.dismiss()


__all__ = ["Toast", "ToastManager", "ToastType"]
