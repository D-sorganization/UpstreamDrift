"""Lightweight AI Chat dock widget for embedding in any PyQt6 application.

Connects to a FastAPI server's WebSocket chat endpoint and provides a
minimal streaming chat interface. Shares conversation context across
all windows via a common session ID persisted to disk.

This widget is fully portable — it depends only on PyQt6 and json,
with no application-specific imports.

Usage::

    from chat import ChatDockWidget

    dock = ChatDockWidget(
        app_context="gasification",
        app_name="integrated_process_simulator",
        accent_color="#3498db",
        parent=main_window,
    )
    main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtWebSockets import QWebSocket
from PyQt6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

_DEFAULT_SERVER = "ws://127.0.0.1:8000"


def _session_file_path(app_name: str) -> Path:
    """Return the path to the shared session ID file for an application."""
    return Path.home() / f".{app_name}" / "active_chat_session.txt"


def _read_shared_session_id(path: Path) -> str | None:
    """Read the active session ID from a shared file."""
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except (PermissionError, OSError):
        pass
    return None


def _write_shared_session_id(session_id: str, path: Path) -> None:
    """Write the active session ID to the shared file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(session_id, encoding="utf-8")
    except (PermissionError, OSError):
        pass


class ChatMessageBubble(QFrame):
    """Compact message bubble for chat display."""

    def __init__(
        self,
        role: str,
        content: str,
        accent_color: str = "#FF8800",
        parent: QWidget | None = None,
    ) -> None:
        assert role is not None, "role must be provided"
        assert role is not None, "role must be provided"
        super().__init__(parent)
        self._role = role
        self._content = content

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        # Role label
        user_style = f"font-size: 10px; font-weight: bold; color: {accent_color};"
        ai_style = "font-size: 10px; font-weight: bold; color: #58a6ff;"
        role_label = QLabel("You" if role == "user" else "AI")
        role_label.setStyleSheet(user_style if role == "user" else ai_style)
        layout.addWidget(role_label)

        # Content
        self._content_label = QLabel(content)
        self._content_label.setWordWrap(True)
        self._content_label.setTextFormat(Qt.TextFormat.PlainText)
        self._content_label.setStyleSheet("color: #e0e0e0; font-size: 12px;")
        layout.addWidget(self._content_label)

        bg = "#2d2d2d" if role == "user" else "#252526"
        self.setStyleSheet(
            f"ChatMessageBubble {{ background-color: {bg}; border-radius: 6px; }}"
        )

    def set_content(self, text: str) -> None:
        """Replace the content text."""
        assert text is not None, "text must be provided"
        assert text is not None, "text must be provided"
        self._content = text
        self._content_label.setText(text)

    def append_content(self, text: str) -> None:
        """Append text to existing content."""
        assert text is not None, "text must be provided"
        assert text is not None, "text must be provided"
        self._content += text
        self._content_label.setText(self._content)


class ChatDockWidget(QDockWidget):
    """Lightweight chat dock widget that connects to a FastAPI chat server.

    Uses QWebSocket for real-time streaming. All instances share the same
    conversation session via a file-persisted session ID.

    Args:
        app_context: Name of the module/context this widget is embedded in.
        app_name: Application identifier for session file storage.
        server_url: WebSocket server base URL.
        session_id: Explicit session ID (None = use shared or create new).
        ws_path_template: WebSocket path template with ``{session_id}`` placeholder.
        placeholder_text: Placeholder text for the input field.
        accent_color: Primary accent color for styling.
        parent: Parent widget.
    """

    # Class-level session for in-process sharing
    _shared_session_id: str | None = None

    def __init__(
        self,
        app_context: str = "unknown",
        app_name: str = "shared_chat",
        server_url: str = _DEFAULT_SERVER,
        session_id: str | None = None,
        ws_path_template: str = "/api/ws/chat/{session_id}",
        placeholder_text: str = "Ask a question...",
        accent_color: str = "#FF8800",
        parent: QWidget | None = None,
    ) -> None:
        assert app_context is not None, "app_context must be provided"
        assert app_context is not None, "app_context must be provided"
        super().__init__("AI Chat", parent)
        self._app_context = app_context
        self._app_name = app_name
        self._server_url = server_url.rstrip("/")
        self._ws_path_template = ws_path_template
        self._accent_color = accent_color
        self._placeholder_text = placeholder_text
        self._is_streaming = False
        self._current_bubble: ChatMessageBubble | None = None
        self._socket: QWebSocket | None = None
        self._session_file = _session_file_path(app_name)
        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setSingleShot(True)
        self._reconnect_timer.timeout.connect(self._connect)

        # Resolve session ID: explicit > class-level > file > "new"
        if session_id:
            ChatDockWidget._shared_session_id = session_id
        elif not ChatDockWidget._shared_session_id:
            ChatDockWidget._shared_session_id = _read_shared_session_id(
                self._session_file
            )

        self._setup_ui()
        # Delay connection slightly so the parent window can finish setup
        QTimer.singleShot(500, self._connect)

    def _setup_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Status bar
        self._status_label = QLabel("Connecting...")
        self._status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self._status_label)

        # Message scroll area
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._message_container = QWidget()
        self._message_layout = QVBoxLayout(self._message_container)
        self._message_layout.setContentsMargins(2, 2, 2, 2)
        self._message_layout.setSpacing(4)
        self._message_layout.addStretch()
        self._scroll_area.setWidget(self._message_container)
        layout.addWidget(self._scroll_area, stretch=1)

        # Input row
        input_row = QHBoxLayout()
        self._input_edit = QPlainTextEdit()
        self._input_edit.setMaximumHeight(50)
        self._input_edit.setPlaceholderText(self._placeholder_text)
        self._input_edit.setStyleSheet(
            "QPlainTextEdit {"
            "  background-color: #2d2d2d; color: #e0e0e0;"
            "  border: 1px solid #444; border-radius: 4px;"
            "  font-size: 12px; padding: 4px;"
            "}"
        )
        input_row.addWidget(self._input_edit, stretch=1)

        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(55)
        self._send_btn.setStyleSheet(
            "QPushButton {"
            f"  background-color: {self._accent_color}; color: black;"
            "  border-radius: 4px; font-weight: bold; padding: 4px;"
            "}"
            "QPushButton:hover { background-color: #ffaa33; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self._send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self._send_btn)

        layout.addLayout(input_row)
        self.setWidget(container)

        # Dock widget styling
        self.setStyleSheet(
            "QDockWidget { background-color: #1e1e1e; color: #e0e0e0; }"
            "QDockWidget::title {"
            f"  background-color: {self._accent_color}; color: black;"
            "  padding: 6px; font-weight: bold;"
            "}"
        )
        self._scroll_area.setStyleSheet(
            "QScrollArea { background-color: #1e1e1e; border: none; }"
        )
        self._message_container.setStyleSheet("background-color: #1e1e1e;")

    # ── WebSocket connection ─────────────────────────────────────────

    def _connect(self) -> None:
        """Establish WebSocket connection to the chat server."""
        if self._socket is not None:
            self._socket.close()
            self._socket.deleteLater()

        self._socket = QWebSocket()
        self._socket.connected.connect(self._on_connected)
        self._socket.disconnected.connect(self._on_disconnected)
        self._socket.textMessageReceived.connect(self._on_message)

        sid = ChatDockWidget._shared_session_id or "new"
        path = self._ws_path_template.replace("{session_id}", sid)
        url = QUrl(f"{self._server_url}{path}")
        self._status_label.setText("Connecting...")
        self._socket.open(url)

    def _on_connected(self) -> None:
        self._status_label.setText("Connected")
        self._status_label.setStyleSheet("color: #3fb950; font-size: 10px;")

    def _on_disconnected(self) -> None:
        self._status_label.setText("Disconnected - retrying in 3s...")
        self._status_label.setStyleSheet("color: #f85149; font-size: 10px;")
        self._is_streaming = False
        self._send_btn.setEnabled(True)
        # Auto-reconnect
        self._reconnect_timer.start(3000)

    def _on_message(self, raw: str) -> None:
        """Handle incoming WebSocket message."""
        assert raw is not None, "raw must be provided"
        assert raw is not None, "raw must be provided"
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        msg_type = data.get("type")

        if msg_type == "session_info":
            sid = data.get("session_id", "")
            ChatDockWidget._shared_session_id = sid
            _write_shared_session_id(sid, self._session_file)
            # Request history to populate UI
            self._send_ws({"action": "history"})

        elif msg_type == "chunk":
            content = data.get("content", "")
            if self._current_bubble:
                self._current_bubble.append_content(content)
                self._scroll_to_bottom()

        elif msg_type == "complete":
            self._is_streaming = False
            self._send_btn.setEnabled(True)
            self._current_bubble = None
            sid = data.get("session_id")
            if sid:
                ChatDockWidget._shared_session_id = sid
                _write_shared_session_id(sid, self._session_file)

        elif msg_type == "session_created":
            sid = data.get("session_id", "")
            ChatDockWidget._shared_session_id = sid
            _write_shared_session_id(sid, self._session_file)

        elif msg_type == "history":
            self._populate_history(data.get("messages", []))

        elif msg_type == "error":
            detail = data.get("detail", "Unknown error")
            self._status_label.setText(f"Error: {detail}")
            self._is_streaming = False
            self._send_btn.setEnabled(True)

    # ── UI actions ───────────────────────────────────────────────────

    def _on_send(self) -> None:
        text = self._input_edit.toPlainText().strip()
        if not text or self._is_streaming:
            return

        self._input_edit.clear()
        self._add_bubble("user", text)

        self._is_streaming = True
        self._send_btn.setEnabled(False)
        self._current_bubble = self._add_bubble("assistant", "")

        self._send_ws(
            {
                "action": "send",
                "message": text,
                "app_context": self._app_context,
            }
        )

    def _send_ws(self, payload: dict) -> None:
        """Send JSON payload over WebSocket."""
        if self._socket and self._socket.isValid():
            self._socket.sendTextMessage(json.dumps(payload))

    def _add_bubble(self, role: str, content: str) -> ChatMessageBubble:
        """Add a message bubble to the scroll area."""
        assert role is not None, "role must be provided"
        assert role is not None, "role must be provided"
        bubble = ChatMessageBubble(role, content, accent_color=self._accent_color)
        # Insert before the stretch item at the end
        count = self._message_layout.count()
        self._message_layout.insertWidget(count - 1, bubble)
        self._scroll_to_bottom()
        return bubble

    def _populate_history(self, messages: list[dict]) -> None:
        """Clear and rebuild message bubbles from history."""
        # Remove existing bubbles (keep the stretch)
        assert messages is not None, "messages must be provided"
        assert messages is not None, "messages must be provided"
        while self._message_layout.count() > 1:
            item = self._message_layout.takeAt(0)
            widget = item.widget() if item else None
            if widget is not None:
                widget.deleteLater()

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                self._add_bubble(role, content)

    def _scroll_to_bottom(self) -> None:
        """Scroll message area to bottom."""
        QTimer.singleShot(
            10,
            lambda: (
                scrollbar.setValue(scrollbar.maximum())
                if (scrollbar := self._scroll_area.verticalScrollBar()) is not None
                else None
            ),
        )

    # ── Cleanup ──────────────────────────────────────────────────────

    def closeEvent(self, event: Any) -> None:
        """Clean up WebSocket on close."""
        self._reconnect_timer.stop()
        if self._socket:
            self._socket.close()
        super().closeEvent(event)
