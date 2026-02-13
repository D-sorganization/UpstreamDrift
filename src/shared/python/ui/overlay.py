"""Overlay widget for displaying controls and information on top of other widgets."""

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles

logger = get_logger(__name__)


class OverlayWidget(QWidget):
    """A transparent overlay widget that sits on top of a parent widget."""

    def __init__(self, parent: QWidget) -> None:
        """Initialize the overlay widget.

        Args:
            parent: The widget to overlay.
        """
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setMouseTracking(True)

        # Semi-transparent background color (mostly transparent)
        self.background_color = QColor(0, 0, 0, 100)  # Dark overlay
        self.overlay_enabled = False

        self._setup_ui()

        # Track parent resize
        if parent:
            parent.installEventFilter(self)
            self.resize(parent.size())

    def _setup_ui(self) -> None:
        """Set up the overlay UI."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # Top bar
        top_bar = QHBoxLayout()

        self.status_label = QLabel("Run Overlay Active")
        self.status_label.setStyleSheet(Styles.OVERLAY_STATUS)
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum
        )
        top_bar.addWidget(self.status_label)
        top_bar.addStretch()

        # Close button
        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.setStyleSheet(Styles.OVERLAY_CLOSE_BTN)
        self.close_btn.clicked.connect(self.hide_overlay)
        top_bar.addWidget(self.close_btn)

        self.main_layout.addLayout(top_bar)
        self.main_layout.addStretch()

        # Bottom controls
        bottom_bar = QHBoxLayout()

        # Example controls
        self.rec_btn = QPushButton("● REC")
        self.rec_btn.setStyleSheet(Styles.OVERLAY_REC_BTN)
        self.rec_btn.setCheckable(True)
        bottom_bar.addWidget(self.rec_btn)

        self.pause_btn = QPushButton("⏸ PAUSE")
        self.pause_btn.setStyleSheet(Styles.OVERLAY_PAUSE_BTN)
        bottom_bar.addWidget(self.pause_btn)

        bottom_bar.addStretch()
        self.main_layout.addLayout(bottom_bar)

        # Initially hidden
        self.setVisible(False)

    def eventFilter(self, obj: object, event: Any) -> bool:  # type: ignore
        """Handle resize events from parent."""
        if obj == self.parent() and event.type() == event.Type.Resize:
            self.resize(event.size())
        return super().eventFilter(obj, event)

    def paintEvent(self, event: Any) -> None:
        """Paint the overlay background."""
        if not self.overlay_enabled:
            return

        painter = QPainter(self)
        painter.fillRect(self.rect(), self.background_color)

    def show_overlay(self) -> None:
        """Show the overlay."""
        self.overlay_enabled = True
        self.setVisible(True)
        self.raise_()
        logger.info("Overlay shown")

    def hide_overlay(self) -> None:
        """Hide the overlay."""
        self.overlay_enabled = False
        self.setVisible(False)
        logger.info("Overlay hidden")

    def toggle(self) -> None:
        """Toggle overlay visibility."""
        if self.isVisible():
            self.hide_overlay()
        else:
            self.show_overlay()
