"""Lightweight AI Chat dock widget for embedding in engine GUIs.

Connects to the FastAPI server's /ws/chat/ WebSocket endpoint
and provides a minimal chat interface. Shares conversation context
across all engine windows via a common session ID.

Usage:
    from src.shared.python.ai.gui.chat_dock_widget import ChatDockWidget

    dock = ChatDockWidget(engine_context="mujoco", parent=main_window)
    main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
"""

from __future__ import annotations

import json
from pathlib import Path

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

_SESSION_FILE = Path.home() / ".golf_modeling_suite" / "active_chat_session.txt"
_DEFAULT_SERVER = "ws://127.0.0.1:8000"


def _read_shared_session_id() -> str | None:
    """Read the active session ID from the shared file."""
    try:
        if _SESSION_FILE.exists():
            text = _SESSION_FILE.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception:
        pass
    return None


def _write_shared_session_id(session_id: str) -> None:
    """Write the active session ID to the shared file."""
    try:
        _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SESSION_FILE.write_text(session_id, encoding="utf-8")
    except Exception:
        pass


class ChatMessageBubble(QFrame):
    """Compact message bubble for chat display."""

    def __init__(self, role: str, content: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._role = role
        self._content = content

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        # Role label
        role_label = QLabel("You" if role == "user" else "AI")
        role_label.setStyleSheet(
            "font-size: 10px; font-weight: bold; color: #FF8800;"
            if role == "user"
            else "font-size: 10px; font-weight: bold; color: #58a6ff;"
        )
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
        self._content = text
        self._content_label.setText(text)

    def append_content(self, text: str) -> None:
        """Append text to existing content."""
        self._content += text
        self._content_label.setText(self._content)


class ChatDockWidget(QDockWidget):
    """Lightweight chat dock widget that connects to the FastAPI chat server.

    Uses QWebSocket for real-time streaming. All instances share the same
    conversation session via a file-persisted session ID.

    Args:
        engine_context: Name of the engine this widget is embedded in.
        server_url: WebSocket server base URL.
        session_id: Explicit session ID (None = use shared or create new).
        parent: Parent widget.
    """

    # Class-level session for in-process sharing
    _shared_session_id: str | None = None

    def __init__(
        self,
        engine_context: str = "unknown",
        server_url: str = _DEFAULT_SERVER,
        session_id: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("AI Chat", parent)
        self._engine_context = engine_context
        self._server_url = server_url.rstrip("/")
        self._is_streaming = False
        self._current_bubble: ChatMessageBubble | None = None
        self._socket: QWebSocket | None = None
        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setSingleShot(True)
        self._reconnect_timer.timeout.connect(self._connect)

        # Resolve session ID: explicit > class-level > file > "new"
        if session_id:
            ChatDockWidget._shared_session_id = session_id
        elif not ChatDockWidget._shared_session_id:
            ChatDockWidget._shared_session_id = _read_shared_session_id()

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
        self._input_edit.setPlaceholderText("Ask about your simulation...")
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
            "  background-color: #FF8800; color: black;"
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
            "  background-color: #FF8800; color: black;"
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
        url = QUrl(f"{self._server_url}/api/ws/chat/{sid}")
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
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        msg_type = data.get("type")

        if msg_type == "session_info":
            sid = data.get("session_id", "")
            ChatDockWidget._shared_session_id = sid
            _write_shared_session_id(sid)
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
                _write_shared_session_id(sid)

        elif msg_type == "session_created":
            sid = data.get("session_id", "")
            ChatDockWidget._shared_session_id = sid
            _write_shared_session_id(sid)

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
                "engine_context": self._engine_context,
            }
        )

    def _send_ws(self, payload: dict) -> None:
        """Send JSON payload over WebSocket."""
        if self._socket and self._socket.isValid():
            self._socket.sendTextMessage(json.dumps(payload))

    def _add_bubble(self, role: str, content: str) -> ChatMessageBubble:
        """Add a message bubble to the scroll area."""
        bubble = ChatMessageBubble(role, content)
        # Insert before the stretch item at the end
        count = self._message_layout.count()
        self._message_layout.insertWidget(count - 1, bubble)
        self._scroll_to_bottom()
        return bubble

    def _populate_history(self, messages: list[dict]) -> None:
        """Clear and rebuild message bubbles from history."""
        # Remove existing bubbles (keep the stretch)
        while self._message_layout.count() > 1:
            item = self._message_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

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
                self._scroll_area.verticalScrollBar().setValue(
                    self._scroll_area.verticalScrollBar().maximum()
                )
                if self._scroll_area.verticalScrollBar()
                else None
            ),
        )

    # ── Cleanup ──────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Clean up WebSocket on close."""
        self._reconnect_timer.stop()
        if self._socket:
            self._socket.close()
        super().closeEvent(event)
