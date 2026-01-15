"""Loading-aware button component for Golf Modeling Suite.

Provides a button that shows a loading spinner and disables interaction
during async operations.

Usage:
    from shared.python.ui.loading_button import LoadingButton

    btn = LoadingButton("Launch Model")
    btn.clicked.connect(on_click)

    # When starting operation
    btn.set_loading(True, "Launching...")

    # When complete
    btn.set_loading(False)
"""

from PyQt6.QtCore import QPropertyAnimation, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QWidget

# Import theme if available
try:
    from shared.python.theme import Colors, Sizes, Weights, get_qfont

    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False


class LoadingSpinner(QWidget):
    """Animated loading spinner widget."""

    def __init__(self, size: int = 16, parent: QWidget | None = None) -> None:
        """Create a loading spinner.

        Args:
            size: Diameter of the spinner in pixels
            parent: Parent widget
        """
        super().__init__(parent)
        self._size = size
        self._angle = 0
        self._spinning = False

        self.setFixedSize(size, size)

        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)

    def _rotate(self) -> None:
        """Rotate the spinner."""
        self._angle = (self._angle + 30) % 360
        self.update()

    def start(self) -> None:
        """Start the spinner animation."""
        self._spinning = True
        self._timer.start(50)  # 20fps
        self.show()

    def stop(self) -> None:
        """Stop the spinner animation."""
        self._spinning = False
        self._timer.stop()
        self.hide()

    def paintEvent(self, event) -> None:
        """Paint the spinner."""
        if not self._spinning:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Center point
        center = self._size // 2

        # Draw arc
        pen = QPen()
        pen.setWidth(2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        # Gradient effect using multiple arcs
        if THEME_AVAILABLE:
            color = QColor(Colors.TEXT_PRIMARY)
        else:
            color = QColor("#FFFFFF")

        # Draw fading tail
        for i in range(8):
            alpha = int(255 * (8 - i) / 8)
            color.setAlpha(alpha)
            pen.setColor(color)
            painter.setPen(pen)

            start_angle = (self._angle - i * 30) * 16  # Qt uses 1/16th degrees
            painter.drawArc(
                2, 2, self._size - 4, self._size - 4, start_angle, 25 * 16
            )

        painter.end()


class LoadingButton(QPushButton):
    """Button with loading state indicator.

    Shows a spinner and optional loading text during async operations.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        """Create a loading button.

        Args:
            text: Button text
            parent: Parent widget
        """
        super().__init__(text, parent)
        self._original_text = text
        self._loading = False

        # Create spinner (initially hidden)
        self._spinner = LoadingSpinner(14, self)
        self._spinner.hide()

        # Style
        self._apply_style()

        # Ensure minimum size for spinner
        self.setMinimumHeight(32)

    def _apply_style(self) -> None:
        """Apply button styling."""
        if THEME_AVAILABLE:
            self.setFont(get_qfont(size=Sizes.BASE, weight=Weights.SEMIBOLD))
        else:
            font = QFont("Segoe UI", 10)
            font.setWeight(QFont.Weight.DemiBold)
            self.setFont(font)

    def set_loading(self, loading: bool, text: str | None = None) -> None:
        """Set the loading state.

        Args:
            loading: Whether to show loading state
            text: Optional loading text (uses "Loading..." if None)
        """
        self._loading = loading

        if loading:
            self.setEnabled(False)
            self.setText(text or "Loading...")
            self._position_spinner()
            self._spinner.start()
        else:
            self.setEnabled(True)
            self.setText(self._original_text)
            self._spinner.stop()

    def _position_spinner(self) -> None:
        """Position the spinner within the button."""
        # Position spinner to the left of text
        margin = 8
        y = (self.height() - self._spinner.height()) // 2
        self._spinner.move(margin, y)

    def resizeEvent(self, event) -> None:
        """Handle resize - reposition spinner."""
        super().resizeEvent(event)
        if self._loading:
            self._position_spinner()

    def is_loading(self) -> bool:
        """Check if button is in loading state."""
        return self._loading

    def set_text(self, text: str) -> None:
        """Set the button text (and update original text).

        Args:
            text: New button text
        """
        self._original_text = text
        if not self._loading:
            self.setText(text)


class IconLoadingButton(QWidget):
    """Button with icon and loading state.

    Shows icon on left, text in center, and can display loading spinner.
    """

    def __init__(
        self,
        text: str = "",
        icon: str = "",
        parent: QWidget | None = None,
    ) -> None:
        """Create an icon loading button.

        Args:
            text: Button text
            icon: Icon character/emoji
            parent: Parent widget
        """
        super().__init__(parent)
        self._original_text = text
        self._icon = icon
        self._loading = False

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the button UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Spinner (hidden initially)
        self._spinner = LoadingSpinner(14, self)
        self._spinner.hide()
        layout.addWidget(self._spinner)

        # Button
        self._button = QPushButton(f"{self._icon}  {self._original_text}")
        self._button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        if THEME_AVAILABLE:
            self._button.setFont(get_qfont(size=Sizes.BASE, weight=Weights.SEMIBOLD))

        layout.addWidget(self._button)

        # Forward click signal
        self._button.clicked.connect(self._on_clicked)

    def _on_clicked(self) -> None:
        """Handle button click."""
        if not self._loading:
            # Emit custom clicked signal (could use pyqtSignal)
            pass

    def set_loading(self, loading: bool, text: str | None = None) -> None:
        """Set loading state."""
        self._loading = loading

        if loading:
            self._button.setEnabled(False)
            self._button.setText(text or "Loading...")
            self._spinner.start()
        else:
            self._button.setEnabled(True)
            self._button.setText(f"{self._icon}  {self._original_text}")
            self._spinner.stop()


__all__ = ["IconLoadingButton", "LoadingButton", "LoadingSpinner"]
