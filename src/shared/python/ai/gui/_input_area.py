"""Input area widget for the AI assistant panel.

Owns the QPlainTextEdit, expertise label, and Send button. Emits a
``send_requested(str)`` signal whenever the user submits a non-empty
message either by pressing Enter or clicking Send.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from src.shared.python.ai.gui.assistant_widgets import ChatInput
from src.shared.python.theme.style_constants import Styles


class InputArea(QFrame):
    """Compose box + Send button + verbosity indicator."""

    send_requested = pyqtSignal(str)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            """
            QFrame {
                background-color: #1e1e1e;
                border-top: 1px solid #3c3c3c;
            }
            """
        )
        layout = QVBoxLayout(self)

        self.input_edit = ChatInput()
        self.input_edit.setPlaceholderText(
            "Type your message here... (Enter to send, Shift+Enter for new line)"
        )
        self.input_edit.setMaximumHeight(100)
        self.input_edit.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: #252526;
                color: #e0e0e0;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 8px;
            }
            QPlainTextEdit:focus { border: 1px solid #FF8800; }
            """
        )
        self.input_edit.submit_requested.connect(self._emit_send)
        layout.addWidget(self.input_edit)

        button_layout = QHBoxLayout()
        self.expertise_label = QLabel("Verbosity: Verbose")
        self.expertise_label.setStyleSheet(Styles.TEXT_MUTED)
        button_layout.addWidget(self.expertise_label)
        button_layout.addStretch()

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._emit_send)
        self.send_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #FF8800;
                color: black;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #cc6d00; }
            QPushButton:disabled {
                background-color: #444444;
                color: #888888;
            }
            """
        )
        button_layout.addWidget(self.send_btn)
        layout.addLayout(button_layout)

        # Ctrl+Enter to send, in addition to plain Enter from ChatInput.
        shortcut = QShortcut(QKeySequence("Ctrl+Return"), self.input_edit)
        shortcut.activated.connect(self._emit_send)

    def _emit_send(self) -> None:
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        self.input_edit.clear()
        self.send_requested.emit(text)

    def set_busy(self, busy: bool) -> None:
        """Disable Send while a request is in flight."""
        self.send_btn.setEnabled(not busy)

    def set_expertise_text(self, text: str) -> None:
        self.expertise_label.setText(text)

    def apply_theme(self, colors: dict) -> None:
        bg_primary = colors["bg_primary"]
        bg_alt = colors["bg_alt"]
        text_primary = colors["text_primary"]
        text_muted = colors["text_muted"]
        border = colors["border"]
        accent = colors["accent"]
        button_hover = colors.get("button_hover", accent)

        self.setStyleSheet(
            f"QFrame {{ background-color: {bg_primary}; "
            f"border-top: 1px solid {border}; }}"
        )
        self.input_edit.setStyleSheet(
            f"""
            QPlainTextEdit {{
                background-color: {bg_alt};
                color: {text_primary};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 8px;
            }}
            QPlainTextEdit:focus {{ border: 1px solid {accent}; }}
            """
        )
        self.send_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {accent};
                color: black;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {button_hover}; }}
            QPushButton:disabled {{
                background-color: {border};
                color: {text_muted};
            }}
            """
        )
        self.expertise_label.setStyleSheet(f"color: {text_muted};")
