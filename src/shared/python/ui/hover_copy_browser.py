"""Custom QTextBrowser with hover copy to clipboard capability."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QEnterEvent, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication, QPushButton, QTextBrowser, QWidget

__all__ = ["HoverCopyTextBrowser"]


class HoverCopyTextBrowser(QTextBrowser):
    """QTextBrowser subclass that overlays a copy to clipboard button on hover."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.copy_btn = QPushButton(self)
        self.copy_btn.hide()
        self.copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(30, 32, 38, 0.85);
                border: 1px solid #495057;
                border-radius: 4px;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: rgba(59, 91, 219, 0.9);
                border-color: #3b5bdb;
            }
        """)

        # SVG for overlapping sheets (copy icon)
        svg_data = (
            b'<svg viewBox="0 0 24 24" width="16" height="16" stroke="#ffffff" '
            b'stroke-width="2" fill="none" stroke-linecap="round" '
            b'stroke-linejoin="round">\n'
            b'  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>\n'
            b'  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 '
            b'2 2v1"></path>\n'
            b"</svg>"
        )

        try:
            renderer = QSvgRenderer(svg_data)
            pixmap = QPixmap(16, 16)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            self.copy_btn.setIcon(QIcon(pixmap))
        except Exception:  # noqa: BLE001
            self.copy_btn.setText("\ud83d\udccb")

        self.copy_btn.setToolTip("Copy to clipboard")
        self.copy_btn.clicked.connect(self.copy_all_text)

        # Ensure hover events are tracked
        self.setMouseTracking(True)

    def copy_all_text(self) -> None:
        """Copy entire text contents to the clipboard."""
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(self.toPlainText())
            self.copy_btn.setToolTip("Copied!")

            orig_text = self.copy_btn.text()
            orig_icon = self.copy_btn.icon()

            # Show checkmark to indicate success
            self.copy_btn.setIcon(QIcon())
            self.copy_btn.setText("\u2713")
            self.copy_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(45, 164, 78, 0.9);
                    border: 1px solid #2da44e;
                    border-radius: 4px;
                    padding: 4px;
                    color: white;
                    font-weight: bold;
                }
            """)

            def restore() -> None:
                self.copy_btn.setToolTip("Copy to clipboard")
                self.copy_btn.setText(orig_text)
                self.copy_btn.setIcon(orig_icon)
                self.copy_btn.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(30, 32, 38, 0.85);
                        border: 1px solid #495057;
                        border-radius: 4px;
                        padding: 4px;
                    }
                    QPushButton:hover {
                        background-color: rgba(59, 91, 219, 0.9);
                        border-color: #3b5bdb;
                    }
                """)

            QTimer.singleShot(2000, restore)

    def enterEvent(self, event: QEnterEvent | None) -> None:
        """Position and show copy button on mouse hover enter."""
        self.position_copy_button()
        self.copy_btn.show()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent | None) -> None:
        """Hide copy button when mouse leaves.

        Does not hide if hover is directly on the button.
        """
        if not self.copy_btn.underMouse():
            self.copy_btn.hide()
        super().leaveEvent(event)

    def resizeEvent(self, event: Any) -> None:
        """Update copy button position on resize."""
        super().resizeEvent(event)
        self.position_copy_button()

    def position_copy_button(self) -> None:
        """Determine position of the button dynamically based on scrollbars."""
        vsb = self.verticalScrollBar()
        scrollbar_w = vsb.width() if (vsb is not None and vsb.isVisible()) else 0
        btn_w = 28
        btn_h = 28
        margin = 8
        self.copy_btn.setGeometry(
            self.width() - btn_w - margin - scrollbar_w,
            margin,
            btn_w,
            btn_h,
        )
