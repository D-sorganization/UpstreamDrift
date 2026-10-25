# ruff: noqa: E501
# mypy: ignore-errors
"""Lightweight AI Chat dock widget for embedding in any PyQt6 application.

Connects to a FastAPI server's WebSocket chat endpoint and provides a
minimal streaming chat interface. Shares conversation context across
all windows via a common session ID persisted to disk.

This widget keeps application-specific integrations behind injectable or
lazy-loaded collaborators so the chat package can be imported independently.

This module historically held the entire dock widget implementation in
one file. To stay under the repo's per-file budget (Tools #5641: UpstreamDrift
carries a verbatim copy under a 1200-line limit) the non-class helpers
and large method groups live in the private ``chat._qt`` package -- the
``StreamingMixin``, ``TerminalModeMixin`` and ``MessagingMixin`` classes
are mixed into :class:`ChatDockWidget`. This module remains the canonical public entry
point and re-exports every name external code or tests reference
(``ChatDockWidget``, ``ChatMessageBubble``, ``HistorySidebar``,
``QWebSocket``, ``QFileDialog`` etc.) so the public API is unchanged.

Usage::

    from chat import ChatDockWidget, ChatConnectionConfig, ChatPresentationConfig

    dock = ChatDockWidget(
        connection=ChatConnectionConfig(
            app_context="gasification",
            app_name="integrated_process_simulator",
        ),
        presentation=ChatPresentationConfig(accent_color="#3498db"),
        parent=main_window,
    )
    main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QSize, QTimer, QUrl, pyqtSignal
from PyQt6.QtWebSockets import QWebSocket
from PyQt6.QtWidgets import (
    QApplication,  # noqa: F401
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QWidget,
)

from . import chat_dock_widget as _connection_contract
from ._qt import ai_dropdowns as _ai
from ._qt.bubbles import ChatMessageBubble
from ._qt.history_sidebar import HistorySidebar

# Backwards-compatible re-exports: existing tests and downstream code may
# patch these names on the ``_chat_dock_widget_qt`` namespace, so they must
# remain importable from this module even though the class body uses the
# helpers from ``_qt`` directly.
from ._qt.input import install_enter_submit as _install_enter_submit  # noqa: F401
from ._qt.messaging import MessagingMixin
from ._qt.queue_panel import QueuedMessage, QueuePanel  # noqa: F401
from ._qt.streaming import StreamingMixin
from ._qt.styling import get_theme_colors as _get_theme_colors  # noqa: F401
from ._qt.terminal_mode import TerminalModeMixin
from ._qt.ui_builder import ChatDockView, build_chat_dock_ui, mirror_chat_dock_view
from ._theme_protocol import ThemeProviderProtocol, _DefaultDarkTheme
from ._thinking_indicator import ThinkingIndicator  # noqa: F401
from ._workspace_protocol import WorkspaceContextProtocol
from .ai_settings_controller import AiSettingsController
from .chat_dock_widget import (
    _read_shared_session_id,
    _resolve_default_server,
    _session_file_path,
    _write_shared_session_id,
)
from .terminal_contracts import TerminalProviderRegistry
from .terminal_providers import build_default_terminal_provider_registry
from .voice_input_manager import VoiceInputManager
from .workspace_command_handler import WorkspaceCommandHandler

logger = logging.getLogger(__name__)

# Re-exports kept for backwards compatibility — test suites and
# downstream code patch these on the ``_chat_dock_widget_qt`` namespace.
__all__ = [
    "ChatConnectionConfig",
    "ChatDockWidget",
    "ChatIntegrationHooks",
    "ChatMessageBubble",
    "ChatPresentationConfig",
    "HistorySidebar",
    "QFileDialog",
    "QWebSocket",
    "QWidget",
]


@dataclass
class ChatConnectionConfig:
    """Connection + session-identity settings for :class:`ChatDockWidget`.

    Groups the WebSocket connection parameters (``server_url``,
    ``session_id``, ``ws_path_template``) with the identity/session
    parameters (``app_context``, ``app_name``, ``project_root``) — the
    latter travel with the former because they are baked into every
    outgoing WS payload and the session-file path. Tools issue #4896.
    """

    server_url: str | None = None
    session_id: str | None = None
    ws_path_template: str = "/api/ws/chat/{session_id}"
    app_context: str = "unknown"
    app_name: str = "shared_chat"
    project_root: str | Path | None = None


@dataclass
class ChatPresentationConfig:
    """Presentation/theming settings for :class:`ChatDockWidget`.

    Tools issue #4896.
    """

    placeholder_text: str = "Ask a question..."
    accent_color: str = "#FF8800"
    theme_provider: ThemeProviderProtocol | None = None
    auto_index_on_open: bool = False


@dataclass
class ChatIntegrationHooks:
    """Dependency-injection hooks for host-application integrations.

    Every field is an optional collaborator the widget falls back to a
    default implementation for when unset (terminal registry, session
    manager, memory manager factory) or simply leaves inert (workspace
    provider, plot request sink). Tools issue #4896.
    """

    terminal_registry: TerminalProviderRegistry | None = None
    workspace_provider: WorkspaceContextProtocol | None = None
    plot_request_sink: Callable[[Any], None] | None = None
    session_manager: Any | None = None
    memory_manager_factory: Callable[[], Any] | None = None


def _build_default_session_manager(app_name: str) -> Any:
    """Create the default chat session manager on demand.

    The AI package still owns the concrete persistence implementation while
    the chat dock is being disentangled for Tools #3331. Keeping this import
    local makes the chat widget module importable without importing ``ai`` and
    lets hosts inject a manager when they already own session persistence.
    """
    if not isinstance(app_name, str) or not app_name.strip():
        raise ValueError("app_name must be a non-empty string")
    import importlib

    storage_dir = Path.home() / f".{app_name}" / "chat_sessions"
    try:
        module = importlib.import_module("src.shared.python.ai.gui.session_manager")
    except ImportError:
        module = importlib.import_module("ai.gui.session_manager")
    return module.ChatSessionManager(storage_dir=storage_dir)


class ChatDockWidget(StreamingMixin, TerminalModeMixin, MessagingMixin, QDockWidget):
    """Lightweight chat dock widget that connects to a FastAPI chat server.

    Uses QWebSocket for real-time streaming. All instances share the same
    conversation session via a file-persisted session ID.
    """

    # Class-level session for in-process sharing. All reads/writes are
    # serialized through ``_SHARED_SESSION_LOCK`` in ``chat_dock_widget``
    # to prevent the multi-window race described in Tools issue #2753.
    _shared_session_id: str | None = None
    _session_lock: threading.Lock = threading.Lock()

    # AI dropdown constants — kept on the class because existing tests
    # introspect them (Tools issue #2871).
    _AI_VALID_THINKING_NAMES = _ai.VALID_THINKING_NAMES
    _AI_VALID_FIELDS = _ai.VALID_FIELDS
    _AI_DEFAULT_PROVIDERS = _ai.DEFAULT_PROVIDERS

    @classmethod
    def _get_shared_session_id(cls) -> str | None:
        with cls._session_lock:
            return cls._shared_session_id

    @classmethod
    def _set_shared_session_id(cls, val: str | None) -> None:
        with cls._session_lock:
            cls._shared_session_id = val

    # Signals for external UI integration
    models_refreshed = pyqtSignal(list)
    index_status_changed = pyqtSignal(dict)

    def __init__(
        self,
        connection: ChatConnectionConfig | None = None,
        presentation: ChatPresentationConfig | None = None,
        integrations: ChatIntegrationHooks | None = None,
        parent: QWidget | None = None,
    ) -> None:
        connection = connection or ChatConnectionConfig()
        presentation = presentation or ChatPresentationConfig()
        integrations = integrations or ChatIntegrationHooks()
        if connection.app_context is None:
            raise ValueError("app_context must be provided")
        super().__init__("AI Chat", parent)
        self._app_context = connection.app_context
        self._app_name = connection.app_name
        resolved_server_url = connection.server_url or _resolve_default_server()
        self._server_url = resolved_server_url.rstrip("/")
        self._ws_path_template = connection.ws_path_template
        self._accent_color = presentation.accent_color
        self._placeholder_text = presentation.placeholder_text
        self._auto_index_on_open = bool(presentation.auto_index_on_open)
        self._project_root = (
            Path(connection.project_root).resolve()
            if connection.project_root
            else Path.cwd()
        )
        self._terminal_registry = (
            integrations.terminal_registry or build_default_terminal_provider_registry()
        )
        self._theme_provider: ThemeProviderProtocol = (
            presentation.theme_provider or _DefaultDarkTheme()
        )
        self._workspace_provider: WorkspaceContextProtocol | None = (
            integrations.workspace_provider
        )
        self._plot_request_sink: Callable[[Any], None] | None = (
            integrations.plot_request_sink
        )
        self._memory_manager_factory = (
            integrations.memory_manager_factory or self._default_memory_manager_factory
        )
        self._is_streaming = False
        # Busy-state message queue: messages typed/sent while streaming
        # land here and are flushed FIFO on each ``complete`` arrival.
        # Stored as ``QueuedMessage`` dataclasses so each row carries a
        # stable id (needed for the steer-by-id protocol exposed by the
        # inline ``QueuePanel`` preview widget).
        self._queued_messages: list[QueuedMessage] = []
        # Chunk-batching: incoming streaming chunks accumulate here and
        # flush to the active bubble on a short QTimer instead of forcing
        # a Qt repaint per network frame. This is the single biggest
        # perceived-latency win measured on warm-model traces because the
        # GUI thread no longer ping-pongs between every 2–10 byte chunk.
        self._chunk_buffer: list[str] = []
        self._chunk_flush_timer = QTimer(self)
        self._chunk_flush_timer.setSingleShot(True)
        self._chunk_flush_timer.setInterval(50)  # 50 ms = 20 Hz max repaint
        self._chunk_flush_timer.timeout.connect(self._flush_chunk_buffer)
        # Send-button visual state machine. Tracks idle/awaiting/stop so
        # the button colour reflects what Enter will do next.
        self._send_button_state: str = "idle"
        self._last_chunk_at: float | None = None
        self._stop_state_timer = QTimer(self)
        self._stop_state_timer.setSingleShot(True)
        self._stop_state_timer.setInterval(10_000)  # 10 s without a chunk
        self._stop_state_timer.timeout.connect(self._on_stop_state_timeout)
        self._current_bubble: ChatMessageBubble | None = None
        self._terminal_session_id: str | None = None
        # Tools issue #2871: mid-thread provider/model/thinking state.
        self._message_history: list[dict[str, Any]] = []
        self._current_provider: str = "ollama"
        self._current_model: str = "llama3"
        self._current_thinking_level: str = "none"
        # ADR-0022 / issue #6119: non-Qt controllers owned by composition.
        # State stays on the widget (the ``current_*`` view properties bridge
        # to ``_current_*``); the controllers hold only the routing rules.
        self._ai_settings = AiSettingsController(self)
        self._workspace_commands = WorkspaceCommandHandler(
            emit=lambda text: self._add_bubble("assistant", text),
            provider=self._workspace_provider,
            plot_sink=self._plot_request_sink,
        )
        self._voice_manager = VoiceInputManager()
        self._terminal_start_pending = False
        self._terminal_runtime_available = False
        self._socket: QWebSocket | None = None
        self._session_file = _session_file_path(self._app_name)
        # Tools issue #2872: conversation-management state.
        self._loaded_context_sessions: list[str] = []
        self._session_manager = (
            integrations.session_manager
            or _build_default_session_manager(self._app_name)
        )
        self._breadcrumb_widget: Any | None = None
        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setSingleShot(True)
        self._reconnect_timer.timeout.connect(self._connect)

        # Resolve session ID: explicit > class-level > file > "new"
        if connection.session_id:
            ChatDockWidget._set_shared_session_id(connection.session_id)
        elif not ChatDockWidget._get_shared_session_id():
            ChatDockWidget._set_shared_session_id(
                _read_shared_session_id(self._session_file)
            )

        self._intentional_disconnect = False
        self._is_closing = False
        self._collapsed: bool = False
        self._setup_ui()
        self._set_terminal_runtime_available(False)
        self._connect_on_show = True

    def _default_memory_manager_factory(self) -> Any:
        """Lazily create the AI memory manager for the memory panel."""
        import importlib

        try:
            module = importlib.import_module("src.shared.python.ai.memory_manager")
        except ImportError:
            module = importlib.import_module("ai.memory_manager")
        return module.MemoryManager()

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        """Switch between full and collapsed state, hiding the main UI components when collapsed."""
        self._collapsed = collapsed

        widgets_to_hide = [
            self._status_label,
            self._tools_btn,
            self._token_indicator,
            self._ai_provider_combo,
            self._ai_model_combo,
            self._ai_thinking_combo,
            self._mode_combo,
            self._content_stack,
            self._input_edit,
            self._upload_btn,
            self._screenshot_btn,
            self._mic_btn,
            self._agent_mode_combo,
            self._send_btn,
            self._steer_btn,
            self._stop_agent_btn,
        ]
        terminal_widgets = [
            self._shell_combo,
            self._provider_combo,
            self._terminal_start_btn,
            self._terminal_stop_btn,
        ]

        if collapsed:
            for w in widgets_to_hide + terminal_widgets:
                if w is not None:
                    w.setVisible(False)
        else:
            for w in widgets_to_hide:
                if w is not None:
                    w.setVisible(True)
            is_terminal = self._current_mode() == "terminal"
            for w in terminal_widgets:
                if w is not None:
                    w.setVisible(is_terminal)

        self.updateGeometry()

    def minimumSizeHint(self) -> QSize:
        if self._collapsed:
            return QSize(56, 0)
        return QSize(320, 0)

    def _setup_ui(self) -> None:
        self._view: ChatDockView = build_chat_dock_ui(self)
        mirror_chat_dock_view(self, self._view)
        self._populate_shell_combo()
        self._populate_provider_combo()
        self._on_mode_changed()
        self._recompute_send_button_state()

    # ── WebSocket connection ─────────────────────────────────────────

    def _connect(self) -> None:
        """Establish WebSocket connection to the chat server."""
        self._intentional_disconnect = False
        self._is_closing = False
        if self._socket is not None:
            self._socket.close()
            self._socket.deleteLater()
        sid = ChatDockWidget._get_shared_session_id() or "new"
        path = self._ws_path_template.replace("{session_id}", sid)
        origin, url_text = _connection_contract._native_websocket_connection(
            self._server_url, path
        )
        self._socket = QWebSocket(origin)
        self._socket.connected.connect(self._on_connected)
        self._socket.disconnected.connect(self._on_disconnected)
        self._socket.textMessageReceived.connect(self._on_message)

        self._status_label.setText("Connecting...")
        self._socket.open(QUrl(url_text))

    def connection_diagnostics(self) -> dict[str, Any]:
        """Return host-readable WebSocket readiness diagnostics."""
        socket = self._socket
        state = "not_started"
        error = ""
        if socket is not None:
            try:
                raw_state = socket.state()
                state = getattr(raw_state, "name", str(raw_state))
            except RuntimeError as exc:
                state = "deleted"
                error = str(exc)
            else:
                try:
                    error = socket.errorString()
                except RuntimeError as exc:
                    error = str(exc)
        return {
            "ready": state == "ConnectedState",
            "server_url": self._server_url,
            "ws_path_template": self._ws_path_template,
            "session_id": ChatDockWidget._get_shared_session_id(),
            "socket_state": state,
            "error": error,
            "connect_on_show": bool(getattr(self, "_connect_on_show", False)),
        }

    def _on_connected(self) -> None:
        self._status_label.setText("Connected")
        self._status_label.setStyleSheet("color: #3fb950; font-size: 10px;")
        self.refresh_models()
        if self._auto_index_on_open:
            self.index_codebase()

    def refresh_models(self) -> None:
        self._send_ws({"action": "refresh_models"})

    def index_codebase(self) -> None:
        self._send_ws({"action": "index_codebase"})

    def open_memory_panel(self) -> None:
        """Open the Sidekick memory management panel (Tools issue #2688)."""
        from .memory_panel import MemoryPanel

        existing = self.__dict__.get("_memory_panel_window")
        if existing is not None:
            try:
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return
            except RuntimeError:
                self._memory_panel_window = None

        manager = self.__dict__.get("_memory_manager")
        if manager is None:
            try:
                manager = self._memory_manager_factory()
            except ImportError:
                logger.warning(
                    "Memory panel unavailable: ai.memory_manager not importable"
                )
                return
            self._memory_manager = manager

        panel = MemoryPanel(manager=manager)
        panel.setWindowTitle("Sidekick Memory")
        panel.resize(520, 480)
        panel.show()
        self._memory_panel_window = panel

    def _on_disconnected(self) -> None:
        if bool(getattr(self, "_intentional_disconnect", False)) or bool(
            getattr(self, "_is_closing", False)
        ):
            if hasattr(self, "_exit_thinking_state"):
                self._exit_thinking_state()
            else:
                self._is_streaming = False
                if hasattr(self, "_send_btn") and self._send_btn is not None:
                    self._send_btn.setEnabled(True)
            return
        self._status_label.setText(
            "Sidekick API unavailable — retrying in 3s. Set UD_CHAT_WS_URL if the local API is external."
        )
        self._status_label.setStyleSheet("color: #f85149; font-size: 10px;")
        if hasattr(self, "_exit_thinking_state"):
            self._exit_thinking_state()
        else:
            self._is_streaming = False
            if hasattr(self, "_send_btn") and self._send_btn is not None:
                self._send_btn.setEnabled(True)
        self._reconnect_timer.start(3000)

    def _on_message(self, raw: str) -> None:
        """Handle incoming WebSocket message."""
        if raw is None:
            raise ValueError("raw must be provided")
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        msg_type = data.get("type")

        if msg_type == "session_info":
            sid = data.get("session_id", "")
            ChatDockWidget._set_shared_session_id(sid)
            _write_shared_session_id(sid, self._session_file)
            capabilities = data.get("capabilities", {})
            self._set_terminal_runtime_available(
                bool(
                    isinstance(capabilities, dict)
                    and capabilities.get("terminal_runtime")
                )
            )
            self._send_ws({"action": "history"})

        elif msg_type == "chunk":
            content = data.get("content", "")
            if self._current_bubble and content:
                # Buffer rather than render synchronously. The flush timer
                # coalesces multiple bytes-per-frame chunks into a single
                # Qt label update — measured ~10x faster end-to-end for
                # warm cloud models that stream short tokens.
                self._chunk_buffer.append(content)
                if not self._chunk_flush_timer.isActive():
                    self._chunk_flush_timer.start()
                # Reset the no-chunk-for-10s timer used by the Send-button
                # "Stop" state. Each chunk arrival pushes the deadline out.
                import time as _time

                self._last_chunk_at = _time.monotonic()
                if self._is_streaming:
                    self._stop_state_timer.start()

        elif msg_type == "complete":
            # Drain any pending chunk fragments before tearing down the
            # current bubble so trailing content is never lost.
            self._flush_chunk_buffer()
            self._stop_state_timer.stop()
            self._exit_thinking_state()
            self._current_bubble = None
            sid = data.get("session_id")
            if sid:
                ChatDockWidget._set_shared_session_id(sid)
                _write_shared_session_id(sid, self._session_file)
            self._flush_queued_messages()

        elif msg_type == "session_created":
            sid = data.get("session_id", "")
            ChatDockWidget._set_shared_session_id(sid)
            _write_shared_session_id(sid, self._session_file)
            while self._message_layout.count() > 1:
                item = self._message_layout.takeAt(0)
                if item:
                    w = item.widget()
                    if w:
                        w.deleteLater()
            self._message_history = []
            self._add_bubble("assistant", "Hello! How can I help you today?")
            if hasattr(self, "_history_sidebar") and self._history_sidebar is not None:
                self._history_sidebar.refresh_lists()

        elif msg_type == "history":
            self._populate_history(data.get("messages", []))

        elif msg_type == "model_list":
            models = data.get("models", [])
            if isinstance(models, list):
                self.models_refreshed.emit(models)

        elif msg_type == "index_status":
            self.index_status_changed.emit(dict(data))
            state = data.get("state")
            if state == "running":
                files = data.get("files_parsed", 0)
                self._status_label.setText(f"Indexing codebase ({files} files)...")
            elif state == "complete":
                self._status_label.setText("Connected")
                self._status_label.setStyleSheet("color: #3fb950; font-size: 10px;")
            elif state == "error":
                detail = data.get("error", "Unknown indexing error")
                self._status_label.setText(f"Index error: {detail}")
                self._status_label.setStyleSheet("color: #f85149; font-size: 10px;")

        elif msg_type == "error":
            detail = data.get("detail", "Unknown error")
            self._status_label.setText(f"Error: {detail}")
            self._exit_thinking_state()
            # Surface remaining queue state to the user; do NOT auto-flush
            # after a server error so queued steering messages are not
            # silently delivered against an unhealthy session.
            self._update_queue_affordance()

        elif msg_type == "terminal_session":
            session = data.get("session", {})
            self._terminal_session_id = session.get("session_id")
            self._terminal_start_pending = False
            state = session.get("state", "unknown")
            self._status_label.setText(f"Terminal {state}")
            if self._terminal_session_id:
                self._append_terminal_line(f"[terminal] session {state}")
            if state in {"stopped", "exited", "error"}:
                self._terminal_session_id = None
            self._sync_terminal_controls()

        elif msg_type == "terminal_events":
            for event in data.get("events", []):
                self._append_terminal_line(event.get("data", ""))

        elif msg_type == "terminal_ack":
            self._status_label.setText("Terminal input sent")

    # ── AI Provider/Model/Thinking dropdowns (Tools issue #2871) ────

    @staticmethod
    def _build_header_combobox(
        *, label: str, items: list[tuple[str, str]]
    ) -> QComboBox:
        return _ai.build_header_combobox(label=label, items=items)

    @staticmethod
    def _build_available_cli_provider_items() -> list[tuple[str, str]]:
        return _ai.build_available_cli_provider_items()

    def _build_ai_dropdowns(self, mode_row: QHBoxLayout) -> None:
        view = self.__dict__.get("_view")
        if not isinstance(view, ChatDockView):
            view = ChatDockView()
            self._view = view
        _ai.build_ai_dropdowns(
            view,
            mode_row,
            on_combo_changed=self._on_ai_combo_changed,
            refresh_model_combo=self._refresh_ai_model_combo,
            refresh_thinking_combo=self._refresh_ai_thinking_combo,
            sync_dropdowns=self._sync_ai_dropdowns,
        )
        mirror_chat_dock_view(self, view)

    def _combo_for_field(self, field: str) -> QComboBox:
        view = self.__dict__.get("_view")
        if isinstance(view, ChatDockView):
            view_field = f"ai_{field}_combo"
            combo = getattr(view, view_field, None)
            if isinstance(combo, QComboBox):
                return combo
        if field == "provider":
            return self._ai_provider_combo
        if field == "model":
            return self._ai_model_combo
        if field == "thinking":
            return self._ai_thinking_combo
        raise ValueError(
            f"_combo_for_field: unknown field {field!r}; expected one of "
            f"{sorted(self._AI_VALID_FIELDS)!r}"
        )

    def _on_ai_combo_changed(self, field: str) -> None:
        combo = self._combo_for_field(field)
        value = combo.currentData()
        if not isinstance(value, str) or not value.strip():
            return
        self._apply_settings_change(field, value)

    def _ai_settings_controller(self) -> AiSettingsController:
        """Return the owned controller, creating one if absent.

        Lazy fallback covers code paths (e.g. tests building a bare stand-in
        via ``__new__``) that bypass ``__init__`` and never set ``_ai_settings``.
        """
        controller = self.__dict__.get("_ai_settings")
        if not isinstance(controller, AiSettingsController):
            controller = AiSettingsController(self)
            self._ai_settings = controller
        return controller

    def _apply_settings_change(self, field: str, value: str) -> None:
        self._ai_settings_controller().apply_settings_change(field, value)

    def _refresh_ai_model_combo(self) -> None:
        _ai.refresh_ai_model_combo(self._view, self._get_active_ai_adapter())

    def _refresh_ai_thinking_combo(self) -> None:
        _ai.refresh_ai_thinking_combo(self._view, self._get_active_ai_adapter())

    def _sync_ai_dropdowns(self) -> None:
        _ai.sync_ai_dropdowns(
            self._view,
            current_provider=self._current_provider,
            current_model=self._current_model,
            current_thinking_level=self._current_thinking_level,
        )

    def _get_active_ai_adapter(self) -> Any | None:
        return _ai.get_active_ai_adapter(self._current_provider)

    def _persist_ai_settings(self) -> None:
        """Persist the current AI selections to a QSettings store.

        Stub by default; hosts override for real persistence.
        """
        return

    # ── AiSettingsController view bridge (issue #6119) ───────────────
    # These map the controller's typed ``AiSettingsView`` protocol onto the
    # widget's canonical ``_current_*`` state + Qt refresh helpers, so the
    # selections remain on the widget (preserving the attributes existing
    # tests read/write) while the routing rules live in the controller.

    @property
    def current_provider(self) -> str:
        return self._current_provider

    @current_provider.setter
    def current_provider(self, value: str) -> None:
        self._current_provider = value

    @property
    def current_model(self) -> str:
        return self._current_model

    @current_model.setter
    def current_model(self, value: str) -> None:
        self._current_model = value

    @property
    def current_thinking_level(self) -> str:
        return self._current_thinking_level

    @current_thinking_level.setter
    def current_thinking_level(self, value: str) -> None:
        self._current_thinking_level = value

    def refresh_model_combo(self) -> None:
        self._refresh_ai_model_combo()

    def refresh_thinking_combo(self) -> None:
        self._refresh_ai_thinking_combo()

    def sync_ai_view(self) -> None:
        view = self.__dict__.get("_view")
        if (
            isinstance(view, ChatDockView)
            and view.ai_provider_combo is not None
            and view.ai_model_combo is not None
            and view.ai_thinking_combo is not None
        ):
            self._sync_ai_dropdowns()

    def persist_ai_settings(self) -> None:
        self._persist_ai_settings()

    def switch_provider(
        self,
        name: str,
        model: str,
        thinking_level: str,
    ) -> None:
        """Switch AI provider / model / thinking-level mid-thread.

        Delegates the validation + state update to the headless
        :class:`AiSettingsController`. The ``_message_history`` immutability
        invariant (issue #2871) is asserted here because the controller never
        touches the history.
        """
        history_before = self._message_history
        snapshot_before = list(history_before)
        self._ai_settings_controller().switch_provider(name, model, thinking_level)
        assert self._message_history is history_before, (
            "switch_provider invariant: _message_history must remain the same list"
        )
        assert self._message_history == snapshot_before, (
            "switch_provider invariant: _message_history contents must not change"
        )

    # ── Qt lifecycle ────────────────────────────────────────────────

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        if getattr(self, "_connect_on_show", False):
            self._connect_on_show = False
            self._connect()

    def closeEvent(self, event: Any) -> None:
        self._intentional_disconnect = True
        self._is_closing = True
        self._reconnect_timer.stop()
        # Stop every QTimer owned by the dock so Qt does not leak a
        # running QTimer past the widget's destruction.
        try:
            self._thinking_indicator.stop()
        except (AttributeError, RuntimeError):
            pass
        try:
            self._chunk_flush_timer.stop()
        except (AttributeError, RuntimeError):
            pass
        try:
            self._stop_state_timer.stop()
        except (AttributeError, RuntimeError):
            pass
        self._stop_scroll_timer()
        if self._socket:
            self._socket.close()
        super().closeEvent(event)

    def _on_new_chat_clicked(self) -> None:
        """Handle 'New Chat' click by requesting a new session via WebSocket."""
        self._send_ws({"action": "new_session"})

    def _on_toggle_history(self) -> None:
        """Toggle history sidebar pane visibility."""
        if hasattr(self, "_history_sidebar") and self._history_sidebar is not None:
            visible = not self._history_sidebar.isVisible()
            self._history_sidebar.setVisible(visible)
            if visible:
                self._history_sidebar.refresh_lists()

    def load_session(self, session_id: str) -> None:
        """Load an existing chat session by ID and reconnect the WebSocket."""
        if not session_id:
            return
        ChatDockWidget._set_shared_session_id(session_id)
        _write_shared_session_id(session_id, self._session_file)

        while self._message_layout.count() > 1:
            item = self._message_layout.takeAt(0)
            if item:
                w = item.widget()
                if w:
                    w.deleteLater()

        self._message_history = []
        self._connect()
        if hasattr(self, "_history_sidebar") and self._history_sidebar is not None:
            self._history_sidebar.refresh_lists()
