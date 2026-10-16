# mypy: ignore-errors
"""Quick-action chat bar for toolbar integration.

Provides a compact, single-line input widget that can be embedded in any
QMainWindow's toolbar for instant AI access. Expands to the full
ChatDockWidget on demand.

Usage::

    from chat.quick_bar import ChatQuickBar

    bar = ChatQuickBar(app_context="gasification", parent=toolbar)
    toolbar.addWidget(bar)

    # Or use the mixin for automatic integration:
    class MyApp(ChatLauncherMixin, QMainWindow):
        def __init__(self):
            super().__init__()
            self.init_chat(app_context="gasification", app_name="ips")
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ._quick_bar_theme import (
    ThemeProviderProtocol,
    _build_system_theme_provider,
    _FallbackThemeProvider,
    _resolve_colors,
)

logger = logging.getLogger(__name__)

# Re-export so callers can do ``from chat.quick_bar import ThemeProviderProtocol``.
__all__ = [
    "ChatLauncherMixin",
    "ChatQuickBar",
    "ThemeProviderProtocol",
    "_FallbackThemeProvider",
    "_resolve_colors",
]

# -- Lazy Qt imports ---------------------------------------------------------

try:
    from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
    from PyQt6.QtWebSockets import QWebSocket
    from PyQt6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QToolBar,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


def _require_qt() -> None:
    """Raise if PyQt6 is not available."""
    if not _QT_AVAILABLE:
        raise ImportError(
            "PyQt6 is required for ChatQuickBar. "
            "Install with: pip install PyQt6 PyQt6-WebSockets"
        )


class ChatQuickBar(QFrame if _QT_AVAILABLE else object):  # type: ignore[misc]
    """Compact single-line AI chat input for toolbar embedding.

    Provides an inline text input with send button that sends queries
    to the chat WebSocket and displays responses in a floating popup.

    Signals:
        expand_requested: Emitted when user clicks the expand button.
        response_received: Emitted with response text for inline display.

    Args:
        app_context: Application context (e.g., "gasification").
        server_url: Chat server WebSocket URL.
        theme_provider: Optional theme provider implementing
            :class:`ThemeProviderProtocol`. When *None* the widget tries the
            installed theme manager and falls back to built-in dark defaults.
        parent: Parent widget.
    """

    if _QT_AVAILABLE:
        expand_requested = pyqtSignal()
        response_received = pyqtSignal(str)

    def __init__(
        self,
        app_context: str = "assistant",
        server_url: str = "ws://127.0.0.1:8000",
        theme_provider: ThemeProviderProtocol | None = None,
        parent: QWidget | None = None,
    ) -> None:
        _require_qt()
        super().__init__(parent)
        self.setObjectName("QuickBarChatWidget")

        self._app_context = app_context
        self._server_url = server_url.rstrip("/")
        self._socket: QWebSocket | None = None
        self._is_waiting = False
        self._response_buffer = ""

        if theme_provider is None:
            self._theme: ThemeProviderProtocol = _build_system_theme_provider()
        else:
            self._theme = theme_provider

        self._setup_ui()
        QTimer.singleShot(1000, self._connect_ws)

    def _get_colors(self) -> dict[str, str]:
        """Resolve theme colors via the stored provider."""
        return _resolve_colors(self._theme)

    def _setup_ui(self) -> None:
        """Build the compact quick-bar UI."""
        c = self._get_colors()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # Robot icon label
        icon_label = QLabel("\U0001f916")
        icon_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(icon_label)

        # Input field
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask AI...")
        self._input.setMinimumWidth(200)
        self._input.setStyleSheet(
            f"QLineEdit {{"
            f"  background-color: {c['input_bg']}; color: {c['text']};"
            f"  border: 1px solid {c['border']}; border-radius: 4px;"
            f"  font-size: 12px; padding: 4px 8px;"
            f"}}"
            f"QLineEdit:focus {{"
            f"  border-color: {c['accent']};"
            f"}}"
        )
        self._input.returnPressed.connect(self._on_send)
        layout.addWidget(self._input, stretch=1)

        # Send button
        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(50)
        self._send_btn.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {c['accent']}; color: black;"
            f"  border-radius: 4px; font-weight: bold;"
            f"  font-size: 11px; padding: 4px;"
            f"}}"
            f"QPushButton:hover {{ background-color: {c['accent_hover']}; }}"
            f"QPushButton:disabled {{"
            f"  background-color: {c['disabled_bg']};"
            f"  color: {c['disabled_fg']};"
            f"}}"
        )
        self._send_btn.clicked.connect(self._on_send)
        layout.addWidget(self._send_btn)

        # Expand button
        self._expand_btn = QPushButton("▼")
        self._expand_btn.setFixedWidth(28)
        self._expand_btn.setToolTip("Open full chat panel")
        self._expand_btn.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {c['button_bg']}; color: {c['text']};"
            f"  border-radius: 4px; font-size: 10px; padding: 4px;"
            f"}}"
            f"QPushButton:hover {{ background-color: {c['button_hover']}; }}"
        )
        self._expand_btn.clicked.connect(self.expand_requested.emit)
        layout.addWidget(self._expand_btn)

        # Status indicator
        self._status = QLabel("●")
        self._status.setFixedWidth(14)
        self._status.setToolTip("Disconnected")
        self._status.setStyleSheet(f"color: {c['muted']}; font-size: 10px;")
        layout.addWidget(self._status)

        # Frame styling
        self.setStyleSheet(
            f"ChatQuickBar {{"
            f"  background-color: {c['bg']};"
            f"  border: 1px solid {c['border']};"
            f"  border-radius: 6px;"
            f"}}"
        )

    def apply_theme(self) -> None:
        """Re-apply the current theme to all child widgets.

        Call this whenever the application theme changes so the quick-bar
        stays in visual sync with the dock widget.
        """
        c = self._get_colors()

        self._input.setStyleSheet(
            f"QLineEdit {{"
            f"  background-color: {c['input_bg']}; color: {c['text']};"
            f"  border: 1px solid {c['border']}; border-radius: 4px;"
            f"  font-size: 12px; padding: 4px 8px;"
            f"}}"
            f"QLineEdit:focus {{"
            f"  border-color: {c['accent']};"
            f"}}"
        )
        self._send_btn.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {c['accent']}; color: black;"
            f"  border-radius: 4px; font-weight: bold;"
            f"  font-size: 11px; padding: 4px;"
            f"}}"
            f"QPushButton:hover {{ background-color: {c['accent_hover']}; }}"
            f"QPushButton:disabled {{"
            f"  background-color: {c['disabled_bg']};"
            f"  color: {c['disabled_fg']};"
            f"}}"
        )
        self._expand_btn.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {c['button_bg']}; color: {c['text']};"
            f"  border-radius: 4px; font-size: 10px; padding: 4px;"
            f"}}"
            f"QPushButton:hover {{ background-color: {c['button_hover']}; }}"
        )
        self._status.setStyleSheet(f"color: {c['muted']}; font-size: 10px;")
        self.setStyleSheet(
            f"ChatQuickBar {{"
            f"  background-color: {c['bg']};"
            f"  border: 1px solid {c['border']};"
            f"  border-radius: 6px;"
            f"}}"
        )

    # -- WebSocket -----------------------------------------------------------

    def _connect_ws(self) -> None:
        """Connect to the chat WebSocket."""
        if self._socket is not None:
            self._socket.close()
            self._socket.deleteLater()

        self._socket = QWebSocket()
        self._socket.connected.connect(self._on_ws_connected)
        self._socket.disconnected.connect(self._on_ws_disconnected)
        self._socket.textMessageReceived.connect(self._on_ws_message)

        url = QUrl(f"{self._server_url}/api/ws/chat/new")
        self._socket.open(url)

    def _on_ws_connected(self) -> None:
        """Handle WebSocket connection."""
        c = self._get_colors()
        self._status.setStyleSheet(f"color: {c['focus']}; font-size: 10px;")
        self._status.setToolTip("Connected")

    def _on_ws_disconnected(self) -> None:
        """Handle WebSocket disconnection."""
        c = self._get_colors()
        self._status.setStyleSheet(f"color: {c['muted']}; font-size: 10px;")
        self._status.setToolTip("Disconnected")
        self._is_waiting = False
        self._send_btn.setEnabled(True)
        QTimer.singleShot(5000, self._connect_ws)

    def _on_ws_message(self, raw: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        msg_type = data.get("type")

        if msg_type == "chunk":
            content = data.get("content", "")
            self._response_buffer += content

        elif msg_type == "complete":
            self._is_waiting = False
            self._send_btn.setEnabled(True)
            self._input.setPlaceholderText("Ask AI...")
            if self._response_buffer:
                self.response_received.emit(self._response_buffer)
                # Show truncated response in placeholder
                preview = self._response_buffer[:80].replace("\n", " ")
                if len(self._response_buffer) > 80:
                    preview += "..."
                self._input.setPlaceholderText(f"AI: {preview}")
            self._response_buffer = ""

        elif msg_type == "error":
            self._is_waiting = False
            self._send_btn.setEnabled(True)
            detail = data.get("detail", "Error")
            self._input.setPlaceholderText(f"Error: {detail}")

    # -- Actions -------------------------------------------------------------

    def _on_send(self) -> None:
        """Send user query."""
        text = self._input.text().strip()
        if not text or self._is_waiting:
            return

        self._input.clear()
        self._input.setPlaceholderText("Thinking...")
        self._is_waiting = True
        self._send_btn.setEnabled(False)
        self._response_buffer = ""

        if self._socket and self._socket.isValid():
            self._socket.sendTextMessage(
                json.dumps(
                    {
                        "action": "send",
                        "message": text,
                        "app_context": self._app_context,
                    }
                )
            )

    def focus_input(self) -> None:
        """Focus the input field (for keyboard shortcut)."""
        self._input.setFocus()
        self._input.selectAll()


# -- Mixin for easy MainWindow integration -----------------------------------


class ChatLauncherMixin:
    """Mixin that adds AI chat quick-bar and dock widget to any QMainWindow.

    Usage::

        class MyApp(ChatLauncherMixin, QMainWindow):
            def __init__(self):
                super().__init__()
                self.init_chat(app_context="gasification", app_name="ips")
    """

    _chat_dock: Any = None
    _chat_quick_bar: ChatQuickBar | None = None

    def init_chat(
        self,
        app_context: str = "assistant",
        app_name: str = "shared",
        server_url: str = "ws://127.0.0.1:8000",
        auto_show_dock: bool = False,
        theme_provider: ThemeProviderProtocol | None = None,
    ) -> None:
        """Initialize the chat integration.

        Adds a quick-bar to the toolbar and optionally shows the
        full chat dock widget.

        Args:
            app_context: Application context for AI prompts.
            app_name: Application name for session file path.
            server_url: Chat server WebSocket URL.
            auto_show_dock: If True, show dock widget on startup.
            theme_provider: Optional theme provider for color resolution.
        """
        _require_qt()

        if not isinstance(self, QMainWindow):
            raise TypeError("ChatLauncherMixin requires QMainWindow")

        # Create quick bar
        self._chat_quick_bar = ChatQuickBar(
            app_context=app_context,
            server_url=server_url,
            theme_provider=theme_provider,
        )
        self._chat_quick_bar.expand_requested.connect(
            lambda: self._toggle_chat_dock(app_context, app_name, server_url)
        )

        # Add to toolbar
        chat_toolbar = QToolBar("AI Chat")
        chat_toolbar.setMovable(False)
        chat_toolbar.addWidget(self._chat_quick_bar)

        import typing

        main_window = typing.cast(QMainWindow, self)
        main_window.addToolBar(Qt.ToolBarArea.TopToolBarArea, chat_toolbar)

        # Keyboard shortcut (Ctrl+Shift+A)
        from PyQt6.QtGui import QKeySequence, QShortcut

        shortcut = QShortcut(QKeySequence("Ctrl+Shift+A"), main_window)
        shortcut.activated.connect(self._chat_quick_bar.focus_input)

        # Auto-show dock if configured
        if auto_show_dock:
            self._toggle_chat_dock(app_context, app_name, server_url)

    def _toggle_chat_dock(
        self,
        app_context: str,
        app_name: str,
        server_url: str,
    ) -> None:
        """Show or hide the full chat dock widget."""
        if self._chat_dock is not None:
            self._chat_dock.setVisible(not self._chat_dock.isVisible())
            return

        # Lazy import to avoid circular deps
        import typing

        from shared.python.chat.chat_dock_widget import (
            ChatConnectionConfig,
            ChatDockWidget,
        )

        main_window = typing.cast(QMainWindow, self)

        self._chat_dock = ChatDockWidget(
            connection=ChatConnectionConfig(
                app_context=app_context,
                app_name=app_name,
                server_url=server_url,
            ),
            parent=main_window,
        )
        main_window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self._chat_dock
        )
        self._chat_dock.show()
