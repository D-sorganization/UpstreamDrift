# ruff: noqa: E501
"""Chat message bubble widget — visual rendering of a single message.

Extracted from the monolithic ``_chat_dock_widget_qt`` module so the
parent file fits in the repo's 1500-line budget. Public name
:class:`ChatMessageBubble` is re-exported from the original module path
for backwards compatibility (Law of Demeter — external consumers must
not import directly from ``_qt``).
"""

from __future__ import annotations

from typing import Literal, cast

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .._theme_protocol import ThemeProviderProtocol
from .styling import get_theme_colors


class ChatMessageBubble(QFrame):
    """Compact message bubble for chat display."""

    def __init__(
        self,
        role: str,
        content: str,
        accent_color: str = "#FF8800",
        parent: QWidget | None = None,
        theme_provider: ThemeProviderProtocol | None = None,
        agent_label: str | None = None,
    ) -> None:
        """Construct a chat-message bubble.

        Args:
            role: ``"user"`` or any other value (treated as assistant).
            content: Initial markdown / plain-text body.
            accent_color: Accent for the user role label.
            parent: Optional Qt parent.
            theme_provider: Theme colours source.
            agent_label: Optional override for the assistant-side role
                label. Typical caller passes ``"Agent (llama3.1:8b)"``
                or ``"Agent (gpt-4o)"`` so users see which model produced
                a given turn. Falls back to ``"Agent"`` when ``None``.

        DbC precondition: ``role`` is a non-None string.
        DbC postcondition: ``self._role_label.text()`` reflects the
            computed label so call sites and tests can assert it.
        """
        if role is None:
            raise ValueError("role must be provided")
        super().__init__(parent)
        self._role = role
        self._content = content

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        colors = get_theme_colors(theme_provider)
        text_primary = colors.get("text", "#e0e0e0")
        bg_alt = colors.get("group_bg", "#2d2d2d")
        bg_secondary = colors.get("input_bg", "#252526")

        # Role label — show the agent's model name when available so users
        # can tell which provider produced each turn. Format: ``Agent (model_name)``;
        # falls back to plain ``Agent`` when the caller didn't supply a model.
        user_style = f"font-size: 10px; font-weight: bold; color: {accent_color};"
        ai_color = colors.get("accent", "#58a6ff")
        ai_style = f"font-size: 10px; font-weight: bold; color: {ai_color};"
        if role == "user":
            label_text = "You"
        elif agent_label:
            label_text = agent_label
        else:
            label_text = "Agent"
        role_label = QLabel(label_text)
        self._role_label = role_label
        role_label.setStyleSheet(user_style if role == "user" else ai_style)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.addWidget(role_label)
        header_row.addStretch()

        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setToolTip(
            "Copy message to clipboard. Use the dropdown to pick mode."
        )
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.setStyleSheet(
            "QPushButton { background-color: transparent; "
            f"color: {colors.get('text_secondary', '#888')}; "
            "border: none; font-size: 10px; padding: 0px; }"
            f"QPushButton:hover {{ color: {colors.get('text', '#e0e0e0')}; }}"
        )
        # Tools issue #2735: per-message copy mode dropdown.
        copy_menu = QMenu(self)
        for label, mode in (
            ("Raw text", "raw_text"),
            ("Markdown", "markdown"),
            ("Code only", "code_only"),
            ("JSON", "json"),
        ):
            act = copy_menu.addAction(label)
            if act is not None:
                act.triggered.connect(
                    lambda _checked=False, m=mode: self._copy_to_clipboard(m)
                )
        self._copy_btn.setMenu(copy_menu)
        # Direct click defaults to raw_text via the menu's first action.
        self._copy_btn.clicked.connect(lambda: self._copy_to_clipboard("raw_text"))
        header_row.addWidget(self._copy_btn)

        layout.addLayout(header_row)

        # Content
        self._content_label = QLabel(content)
        self._content_label.setWordWrap(True)
        self._content_label.setTextFormat(Qt.TextFormat.PlainText)
        self._content_label.setStyleSheet(f"color: {text_primary}; font-size: 12px;")
        layout.addWidget(self._content_label)

        bg = bg_alt if role == "user" else bg_secondary
        self.setStyleSheet(
            f"ChatMessageBubble {{ background-color: {bg}; border-radius: 6px; }}"
        )

    def set_content(self, text: str) -> None:
        """Replace the content text."""
        if text is None:
            raise ValueError("text must be provided")
        self._content = text
        self._content_label.setText(text)

    def append_content(self, text: str) -> None:
        """Append text to existing content."""
        if text is None:
            raise ValueError("text must be provided")
        self._content += text
        self._content_label.setText(self._content)

    def _copy_to_clipboard(self, mode: str = "raw_text") -> None:
        """Copy this bubble's content via the shared ``MessageClipboardCopier``.

        Tools issue #2735. The copier is constructed lazily because it
        pulls in :class:`QApplication` and is only meaningful when a Qt
        application is running.
        """
        from ..export import MessageClipboardCopier
        from ..service_base import ChatMessage

        try:
            copier = MessageClipboardCopier.from_qt_application()
        except RuntimeError:
            # Fall back to direct clipboard call when no QApplication exists.
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(self._content)
            return
        msg = ChatMessage(role=self._role, content=self._content)
        try:
            copier.copy_message(
                msg, cast("Literal['raw_text', 'markdown', 'code_only', 'json']", mode)
            )
        except ValueError:
            # Unknown mode -- fall back to raw_text.
            copier.copy_message(msg, "raw_text")
