"""Message transcript widgets.

Provides MessageWidget (single message bubble with markdown rendering) and
build_message_area() for constructing the scrollable conversation history.
"""

from __future__ import annotations

from datetime import datetime, timezone

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.theme.style_constants import Styles


class MessageWidget(QFrame):
    """Widget displaying a single message in the conversation."""

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: datetime | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize message widget.

        Args:
            role: Message role (user, assistant, system).
            content: Message content.
            timestamp: When the message was created.
            parent: Parent widget.
        """
        if role is None:
            raise ValueError("role must be provided")
        super().__init__(parent)
        self._role = role
        self._content = content
        self._timestamp = timestamp or datetime.now(timezone.utc)
        self._setup_ui()
        self._apply_style()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        header = QHBoxLayout()

        role_label = QLabel(self._get_role_display())
        role_label.setStyleSheet(Styles.TEXT_LABEL_BOLD_WHITE)
        header.addWidget(role_label)

        header.addStretch()

        time_label = QLabel(self._timestamp.strftime("%H:%M"))
        time_label.setStyleSheet(Styles.TEXT_MUTED)
        header.addWidget(time_label)

        layout.addLayout(header)

        self._content_label = QTextEdit()
        self._content_label.setReadOnly(True)
        self._content_label.setFrameShape(QFrame.Shape.NoFrame)
        self._content_label.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._content_label.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._content_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self._content_label.setMarkdown(self._content)
        self._content_label.setStyleSheet(Styles.TEXT_CONTENT_TRANSPARENT)

        doc = self._content_label.document()
        if doc is not None:
            doc.contentsChanged.connect(self._adjust_height)
        self._adjust_height()

        layout.addWidget(self._content_label)

    def _get_role_display(self) -> str:
        """Get display name for role."""
        role_map = {
            "user": "You",
            "assistant": "AI Assistant",
            "system": "System",
            "tool": "Tool Result",
        }
        return role_map.get(self._role, self._role.title())

    def _apply_style(self) -> None:
        """Apply styling based on role and current theme."""
        self.refresh_theme()

    def refresh_theme(self) -> None:
        """Refresh colors from ThemeManager."""
        try:
            from src.shared.python.theme.theme_manager import get_theme_manager

            colors = get_theme_manager().get_current_colors()

            def _get(key, fallback):
                if isinstance(colors, dict):
                    return colors.get(key, fallback)
                return getattr(colors, key, fallback)

            bg_alt = _get("bg_elevated", _get("group_bg", "#2d2d2d"))
            bg_secondary = _get("bg_highlight", _get("input_bg", "#252526"))
            text_primary = _get("text_primary", _get("text", "#e0e0e0"))
        except ImportError:
            bg_alt = "#2d2d2d"
            bg_secondary = "#252526"
            text_primary = "#e0e0e0"

        bg = bg_alt if self._role == "user" else bg_secondary
        self.setStyleSheet(
            f"MessageWidget {{ background-color: {bg}; border-radius: 6px; }}"
        )
        self._content_label.setStyleSheet(
            f"color: {text_primary}; background: transparent; border: none;"
        )

    def _adjust_height(self) -> None:
        """Adjust height to fit content."""
        doc = self._content_label.document()
        if doc is not None:
            doc_height = doc.size().height()
            self._content_label.setFixedHeight(int(doc_height) + 10)

    def append_content(self, text: str) -> None:
        """Append content to the message (for streaming).

        Args:
            text: Text to append.
        """
        if text is None:
            raise ValueError("text must be provided")
        self._content += text
        self._content_label.setMarkdown(self._content)

    def set_content(self, text: str) -> None:
        """Set message content.

        Args:
            text: New content.
        """
        if text is None:
            raise ValueError("text must be provided")
        self._content = text
        self._content_label.setMarkdown(self._content)

    def get_content(self) -> str:
        """Get current content."""
        return self._content


def build_message_area() -> tuple[QScrollArea, QWidget, QVBoxLayout]:
    """Build the scrollable message area container.

    Returns:
        A 3-tuple of (scroll_area, message_container, message_layout).
    """
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setStyleSheet("""
        QScrollArea {
            background-color: #1e1e1e;
            border: none;
        }
        QScrollBar:vertical {
            background: #1e1e1e;
            width: 10px;
            margin: 0px 0px 0px 0px;
        }
        QScrollBar::handle:vertical {
            background: #424242;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            background: none;
        }
    """)

    container = QWidget()
    container.setStyleSheet(Styles.CONTAINER_DARK)
    msg_layout = QVBoxLayout(container)
    msg_layout.setContentsMargins(8, 8, 8, 8)
    msg_layout.setSpacing(8)
    msg_layout.addStretch()

    scroll.setWidget(container)
    return scroll, container, msg_layout
