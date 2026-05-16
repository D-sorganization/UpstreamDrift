"""Composer widgets for message input.

Provides ChatInput (QPlainTextEdit that emits submit_requested on Enter)
and build_input_area() that assembles the full input row with a send button.
"""

from __future__ import annotations

from PyQt6 import QtGui
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.theme.style_constants import Styles


class ChatInput(QPlainTextEdit):
    """Custom input widget handling Send vs Newline."""

    submit_requested = pyqtSignal()

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        """Handle key press events."""
        if event is None:
            return
        if (
            event.key() == Qt.Key.Key_Return
            and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            event.accept()
            self.submit_requested.emit()
        else:
            super().keyPressEvent(event)


def build_input_area() -> tuple[QWidget, ChatInput, QPushButton, QLabel]:
    """Build the message input area frame.

    Returns:
        A 4-tuple of (container_widget, chat_input, send_button, expertise_label).
    """
    widget = QFrame()
    widget.setStyleSheet("""
        QFrame {
            background-color: #1e1e1e;
            border-top: 1px solid #3c3c3c;
        }
        """)

    layout = QVBoxLayout(widget)

    input_edit = ChatInput()
    input_edit.setPlaceholderText(
        "Type your message here... (Enter to send, Shift+Enter for new line)"
    )
    input_edit.setMaximumHeight(100)
    input_edit.setStyleSheet("""
        QPlainTextEdit {
            background-color: #252526;
            color: #e0e0e0;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 8px;
        }
        QPlainTextEdit:focus {
            border: 1px solid #FF8800;
        }
    """)
    layout.addWidget(input_edit)

    button_layout = QHBoxLayout()

    expertise_label = QLabel("Verbosity: Verbose")
    expertise_label.setStyleSheet(Styles.TEXT_MUTED)
    button_layout.addWidget(expertise_label)

    button_layout.addStretch()

    send_btn = QPushButton("Send")
    send_btn.setStyleSheet("""
        QPushButton {
            background-color: #FF8800;
            color: black;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #cc6d00;
        }
        QPushButton:disabled {
            background-color: #444444;
            color: #888888;
        }
        """)
    button_layout.addWidget(send_btn)

    layout.addLayout(button_layout)

    return widget, input_edit, send_btn, expertise_label
