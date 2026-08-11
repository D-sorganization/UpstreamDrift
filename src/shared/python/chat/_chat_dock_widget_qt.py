# ruff: noqa: E501
# mypy: ignore-errors
"""Lightweight AI Chat dock widget for embedding in any PyQt6 application.

Connects to a FastAPI server's WebSocket chat endpoint and provides a
minimal streaming chat interface. Shares conversation context across
all windows via a common session ID persisted to disk.

This widget keeps application-specific integrations behind injectable or
lazy-loaded collaborators so the chat package can be imported independently.

This module historically held the entire dock widget implementation in
one file. To stay under the repo's 1,200-line per-file budget the
non-class helpers and large method bodies now live in the private
``chat._qt`` package; this module remains the canonical public entry
point and re-exports every name external code or tests reference
(``ChatDockWidget``, ``ChatMessageBubble``, ``HistorySidebar``,
``QWebSocket``, ``QFileDialog`` etc.) so the public API is unchanged.

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

import base64
import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QSize, QTimer, QUrl, pyqtSignal
from PyQt6.QtWebSockets import QWebSocket
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QWidget,
)

from . import chat_dock_widget as _connection_contract
from ._qt import ai_dropdowns as _ai
from ._qt import exports as _exports
from ._qt import runtime as _runtime
from ._qt import sessions as _sessions
from ._qt.bubbles import ChatMessageBubble
from ._qt.history_sidebar import HistorySidebar

# Backwards-compatible re-exports: existing tests and downstream code may
# patch these names on the ``_chat_dock_widget_qt`` namespace, so they must
# remain importable from this module even though the class body uses the
# helpers from ``_qt`` directly.
from ._qt.input import install_enter_submit as _install_enter_submit  # noqa: F401
from ._qt.queue_panel import QueuedMessage, QueuePanel  # noqa: F401
from ._qt.styling import get_theme_colors as _get_theme_colors  # noqa: F401
from ._qt.ui_builder import ChatDockView, build_chat_dock_ui, mirror_chat_dock_view
from ._theme_protocol import ThemeProviderProtocol, _DefaultDarkTheme
from ._thinking_indicator import ThinkingIndicator
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
import contextlib

logger = logging.getLogger(__name__)

# Re-exports kept for backwards compatibility — test suites and
# downstream code patch these on the ``_chat_dock_widget_qt`` namespace.
__all__ = [
    "ChatDockWidget",
    "ChatMessageBubble",
    "HistorySidebar",
    "QFileDialog",
    "QWebSocket",
    "QWidget",
]


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


class ChatDockWidget(QDockWidget):
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
        app_context: str = "unknown",
        app_name: str = "shared_chat",
        server_url: str | None = None,
        session_id: str | None = None,
        ws_path_template: str = "/api/ws/chat/{session_id}",
        placeholder_text: str = "Ask a question...",
        accent_color: str = "#FF8800",
        auto_index_on_open: bool = False,
        project_root: str | Path | None = None,
        terminal_registry: TerminalProviderRegistry | None = None,
        theme_provider: ThemeProviderProtocol | None = None,
        workspace_provider: WorkspaceContextProtocol | None = None,
        plot_request_sink: Callable[[Any], None] | None = None,
        parent: QWidget | None = None,
        *,
        session_manager: Any | None = None,
        memory_manager_factory: Callable[[], Any] | None = None,
    ) -> None:
        if app_context is None:
            raise ValueError("app_context must be provided")
        super().__init__("AI Chat", parent)
        self._app_context = app_context
        self._app_name = app_name
        resolved_server_url = server_url or _resolve_default_server()
        self._server_url = resolved_server_url.rstrip("/")
        self._ws_path_template = ws_path_template
        self._accent_color = accent_color
        self._placeholder_text = placeholder_text
        self._auto_index_on_open = bool(auto_index_on_open)
        self._project_root = (
            Path(project_root).resolve() if project_root else Path.cwd()
        )
        self._terminal_registry = (
            terminal_registry or build_default_terminal_provider_registry()
        )
        self._theme_provider: ThemeProviderProtocol = (
            theme_provider or _DefaultDarkTheme()
        )
        self._workspace_provider: WorkspaceContextProtocol | None = workspace_provider
        self._plot_request_sink: Callable[[Any], None] | None = plot_request_sink
        self._memory_manager_factory = (
            memory_manager_factory or self._default_memory_manager_factory
        )
        _runtime.initialize_streaming_state(self, timer_factory=QTimer)
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
        self._session_file = _session_file_path(app_name)
        # Tools issue #2872: conversation-management state.
        self._loaded_context_sessions: list[str] = []
        self._session_manager = session_manager or _build_default_session_manager(
            self._app_name
        )
        self._breadcrumb_widget: Any | None = None
        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setSingleShot(True)
        self._reconnect_timer.timeout.connect(self._connect)

        # Resolve session ID: explicit > class-level > file > "new"
        if session_id:
            ChatDockWidget._set_shared_session_id(session_id)
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
        _runtime.set_chat_dock_collapsed(self, collapsed)

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
        _runtime.connect(
            self,
            native_connection=_connection_contract._native_websocket_connection,
            websocket_factory=QWebSocket,
            url_factory=QUrl,
        )

    def connection_diagnostics(self) -> dict[str, Any]:
        """Return host-readable WebSocket readiness diagnostics."""
        return _runtime.connection_diagnostics(self)

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
        _runtime.on_disconnected(self)

    def _on_message(self, raw: str) -> None:
        """Handle incoming WebSocket message."""
        _runtime.handle_message(
            self,
            raw,
            set_shared_session_id=ChatDockWidget._set_shared_session_id,
            write_shared_session_id=_write_shared_session_id,
        )

    # ── Thinking indicator (Tools chat thinking-indicator) ───────────

    @property
    def thinking_indicator(self) -> ThinkingIndicator:
        """Return the dock's shared "AI is thinking" indicator widget.

        The dock builds exactly one indicator and parks it between the
        message stack and the input row. Callers must not reach through the
        widget hierarchy to find it — this property is the canonical handle.
        """
        return self._thinking_indicator

    def _enter_thinking_state(self) -> None:
        """Transition the dock into the "agent is thinking" state.

        Pre: Qt application is running and the indicator widget exists.
        Post: ``_is_streaming`` is ``True`` and the indicator is animating.

        Idempotent — safe to call across a queue flush where the indicator
        is already on. This is the single hand-off point used by both the
        regular send path and the slash-command path so the wiring stays DRY.
        """
        self._is_streaming = True
        # Send button stays enabled in ``awaiting`` / ``stop`` states so
        # the user can keep typing steering messages or click to abort.
        # The indicator may not exist during very early initialization or
        # in tests that intentionally bypass ``_setup_ui``.
        with contextlib.suppress(AttributeError):
            self._thinking_indicator.start()
        # Start the no-chunk-for-10s watchdog so the Send button flips to
        # red "Stop" if the agent stalls.
        self._stop_state_timer.start()
        self._recompute_send_button_state()

    def _exit_thinking_state(self) -> None:
        """Transition the dock back to ``idle`` and stop the indicator.

        Idempotent — multiple terminal chunks (``complete``/``error``) and
        the disconnect handler all converge here without ill effect.
        """
        self._is_streaming = False
        with contextlib.suppress(AttributeError):
            self._send_btn.setEnabled(True)
        with contextlib.suppress(AttributeError):
            self._thinking_indicator.stop()
        self._stop_state_timer.stop()
        self._recompute_send_button_state()

    # ── Chunk batching + Send-button visual state ───────────────────

    # Style sheets per Send-button visual state. Kept as class-level
    # constants so theming swaps stay DRY and tests can assert exact
    # colour transitions without duplicating literals.
    _SEND_BTN_STYLES: dict[str, str] = {
        "idle": (
            "QPushButton {"
            "  background-color: #3fb950; color: black;"
            "  border-radius: 4px; font-weight: bold; padding: 4px;"
            "}"
            "QPushButton:hover { background-color: #4ade80; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        ),
        "awaiting": (
            "QPushButton {"
            "  background-color: #c79100; color: black;"
            "  border-radius: 4px; font-weight: bold; padding: 4px;"
            "}"
            "QPushButton:hover { background-color: #e7b800; }"
        ),
        "stop": (
            "QPushButton {"
            "  background-color: #f85149; color: white;"
            "  border-radius: 4px; font-weight: bold; padding: 4px;"
            "}"
            "QPushButton:hover { background-color: #ff6b63; }"
        ),
    }

    def _flush_chunk_buffer(self) -> None:
        """Drain the chunk buffer into the active bubble in one paint.

        Idempotent — safe to call when the buffer is empty (a no-op) or
        when no current bubble exists (drops the buffer, since there is
        nowhere to render it).
        """
        if not self._chunk_buffer:
            return
        joined = "".join(self._chunk_buffer)
        self._chunk_buffer.clear()
        if self._current_bubble is None:
            return
        self._current_bubble.append_content(joined)
        self._scroll_to_bottom()

    def _on_stop_state_timeout(self) -> None:
        """10 s elapsed without a chunk — promote the Send button to "Stop"."""
        if self._is_streaming:
            self.set_send_button_state("stop")

    def set_send_button_state(self, state: str) -> None:
        """Set the Send button's visual state.

        DbC:
            Pre: ``state`` ∈ {"idle", "awaiting", "stop"}.
            Post: ``self._send_button_state == state`` and the Send
                button's text + stylesheet reflect that state.
        """
        if state not in self._SEND_BTN_STYLES:
            raise ValueError(
                f"set_send_button_state: unknown state {state!r}; expected "
                f"one of {sorted(self._SEND_BTN_STYLES)!r}"
            )
        self._send_button_state = state
        btn = getattr(self, "_send_btn", None)
        if btn is None:
            return
        depth = len(self._queued_messages)
        label = {
            "idle": "Send" if depth == 0 else f"Queue ({depth})",
            "awaiting": ("Steer" if depth == 0 else f"Steer ({depth})"),
            "stop": "Stop",
        }[state]
        tip = {
            "idle": "Send message",
            "awaiting": "Press Enter to queue + steer the in-progress response",
            "stop": "Press to stop the in-progress response",
        }[state]
        btn.setText(label)
        btn.setToolTip(tip)
        btn.setStyleSheet(self._SEND_BTN_STYLES[state])

    def _recompute_send_button_state(self) -> None:
        """Re-derive the Send-button state from ``input_state`` + timing.

        Called from any path that may have changed the state inputs:
        queue mutation, streaming start/stop, chunk arrival, stop-timer.
        """
        # If a 10-second no-chunk window has elapsed the stop-timer has
        # already promoted the state to ``"stop"``. Preserve that here
        # rather than overwriting it back down to ``awaiting``.
        if self._is_streaming and self._send_button_state == "stop":
            self.set_send_button_state("stop")
            return
        state = "idle" if not self._is_streaming else "awaiting"
        self.set_send_button_state(state)

    # ── Busy-state message queue (Tools input keybindings) ───────────

    @property
    def input_state(self) -> str:
        """Return the current input state.

        One of:
            * ``"idle"``: nothing in flight, nothing queued.
            * ``"sending"``: a message is generating but nothing is queued.
            * ``"awaiting"``: a message is generating *and* at least one
              steering message is queued waiting to flush.
        """
        if not self._is_streaming:
            return "idle"
        if self._queued_messages:
            return "awaiting"
        return "sending"

    def queued_messages(self) -> list[str]:
        """Return the queued steering-message texts in FIFO order.

        Returns a shallow copy so external callers (tests, observers)
        cannot mutate the dock's internal list. Kept ``list[str]`` for
        backwards compatibility with downstream code; the richer
        :meth:`queued_message_records` exposes the full dataclasses.
        """
        return [m.text for m in self._queued_messages]

    def queued_message_records(self) -> list[QueuedMessage]:
        """Return a shallow copy of the queued :class:`QueuedMessage` records."""
        return list(self._queued_messages)

    def _update_queue_affordance(self) -> None:
        """Refresh Send button + input placeholder + preview panel."""
        depth = len(self._queued_messages)
        # Mirror the queue into the inline preview panel if it has been
        # built. The panel is constructed by ``ui_builder`` and stored at
        # ``_queue_panel``; tests that bypass the builder leave it unset.
        panel = getattr(self, "_queue_panel", None)
        if panel is not None:
            panel.set_messages(list(self._queued_messages))
        if depth > 0:
            self._input_edit.setPlaceholderText(
                f"Queued: {depth} — type to queue another (steers next turn)"
            )
        else:
            self._input_edit.setPlaceholderText(self._placeholder_text)
        self._recompute_send_button_state()

    def _submit_or_queue(self, text: str) -> None:
        """Submit ``text`` immediately or queue it if the agent is busy.

        DRY: this is the single internal pathway shared by the Send-button
        click and the input widget's Enter keypress.

        DbC:
            Pre: ``text`` is a non-empty / non-whitespace string after strip.
            Pre: ``self.input_state`` ∈ {"idle", "sending", "awaiting"}.
            Post: either an ``action="send"`` WS payload is dispatched and
                  ``_is_streaming`` becomes True, OR the text is appended
                  to ``self._queued_messages``.
        """
        if not isinstance(text, str):
            raise ValueError("_submit_or_queue: text must be a string")
        stripped = text.strip()
        if not stripped:
            raise ValueError("_submit_or_queue: text must be non-empty")
        assert self.input_state in {"idle", "sending", "awaiting"}, (
            "_submit_or_queue: invariant — input_state must be one of "
            "{idle, sending, awaiting}"
        )

        if self._is_streaming:
            # Queue as a steering message; do NOT add a user bubble yet —
            # the bubble will appear when the queue flushes so the visible
            # conversation matches what the server actually sees.
            self._queued_messages.append(QueuedMessage(text=stripped))
            self._update_queue_affordance()
            return

        self._add_bubble("user", stripped)
        self._enter_thinking_state()
        self._current_bubble = self._add_bubble("assistant", "")

        payload: dict[str, Any] = {
            "action": "send",
            "message": stripped,
            "app_context": self._app_context,
        }
        workspace_context = self._build_workspace_context_block()
        if workspace_context:
            payload["workspace_context"] = workspace_context
        self._send_ws(payload)
        assert self._is_streaming is True

    def _flush_queued_messages(self) -> None:
        """Flush the next queued message (if any) as a fresh user turn."""
        if not self._queued_messages:
            self._update_queue_affordance()
            return
        next_msg = self._queued_messages.pop(0)
        self._update_queue_affordance()
        self._submit_or_queue(next_msg.text)

    def steer_to_front(self, message_id: str) -> None:
        """Move the queued message with ``message_id`` to the front of the queue.

        Wired to :class:`QueuePanel.steer_requested`. Idempotent for the
        message already at the front. Raises ``ValueError`` for unknown
        ids so callers can surface the inconsistency in tests.

        DbC:
            Pre: ``message_id`` is a non-empty string.
            Pre: a queued message with that id exists.
            Post: ``self._queued_messages[0].id == message_id``.
        """
        if not isinstance(message_id, str) or not message_id:
            raise ValueError("steer_to_front: message_id must be a non-empty string")
        for i, msg in enumerate(self._queued_messages):
            if msg.id == message_id:
                if i != 0:
                    self._queued_messages.insert(0, self._queued_messages.pop(i))
                    self._update_queue_affordance()
                return
        raise ValueError(f"steer_to_front: no queued message with id {message_id!r}")

    # ── UI button handlers ───────────────────────────────────────────

    def _on_steer(self) -> None:
        """Explicitly queue the current input as a steering message."""
        text = self._input_edit.toPlainText().strip()
        if not text:
            return
        self._input_edit.clear()
        self._queued_messages.append(QueuedMessage(text=text))
        self._update_queue_affordance()

    def _on_stop_agent(self) -> None:
        logger.info("Agent response stopped by user")
        if hasattr(self, "_chat_client") and hasattr(
            self._chat_client, "cancel_current_stream"
        ):
            self._chat_client.cancel_current_stream()

    def _on_send(self) -> None:
        """Send-button click and Enter-keypress entry point."""
        text = self._input_edit.toPlainText().strip()
        if not text:
            return

        if not self._is_streaming and self._current_mode() == "terminal":
            self._on_terminal_input(text)
            return

        if not self._is_streaming and text.startswith("/"):
            self._input_edit.clear()
            self._handle_slash_command(text)
            return

        self._input_edit.clear()
        self._submit_or_queue(text)

    def _handle_slash_command(self, text: str) -> None:
        if text is None:
            raise ValueError("text must be provided")
        parts = text.split(maxsplit=1)
        cmd = parts[0][1:].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in {"ws.read", "ws.write", "plot"}:
            self._input_edit.clear()
            self._add_bubble("user", text)
            self._dispatch_workspace_command(cmd, arg)
            return

        if cmd == "use-session":
            self._input_edit.clear()
            self._add_bubble("user", text)
            self._handle_use_session(arg.strip())
            return

        self._input_edit.clear()
        self._add_bubble("user", text)
        self._enter_thinking_state()
        self._current_bubble = self._add_bubble(
            "assistant", f"Starting workflow: {cmd}..."
        )
        self._send_ws(
            {
                "action": "skill_invoke",
                "skill_id": cmd,
                "app_context": self._app_context,
            }
        )

    # ── Workspace bridge (Tools issue #2849; #6119 controller) ───────

    def _build_workspace_context_block(self) -> str:
        return self._workspace_commands.context_block()

    def _dispatch_workspace_command(self, cmd: str, arg: str) -> None:
        self._workspace_commands.dispatch(cmd, arg)

    def _handle_ws_read(self, arg: str) -> None:
        self._workspace_commands.handle_ws_read(arg)

    def _handle_ws_write(self, arg: str) -> None:
        self._workspace_commands.handle_ws_write(arg)

    def _handle_plot(self, arg: str) -> None:
        self._workspace_commands.handle_plot(arg)

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

    # ── Terminal mode ───────────────────────────────────────────────

    def _populate_shell_combo(self) -> None:
        _runtime.populate_shell_combo(self)

    def _populate_provider_combo(self) -> None:
        _runtime.populate_provider_combo(self)

    def _on_terminal_shell_changed(self, _index: int) -> None:
        self._populate_provider_combo()

    def _on_terminal_start(self) -> None:
        _runtime.on_terminal_start(self)

    def _on_terminal_stop(self) -> None:
        _runtime.on_terminal_stop(self)

    def _on_terminal_input(self, text: str) -> None:
        _runtime.on_terminal_input(self, text)

    # ── Input affordances ───────────────────────────────────────────

    def _on_upload(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Attach File", "", "All Files (*)"
        )
        if file_path:
            path = Path(file_path)
            try:
                data = path.read_bytes()
                b64 = base64.b64encode(data).decode("ascii")
                self._send_ws(
                    {"action": "file_upload", "filename": path.name, "content": b64}
                )
                self._add_bubble("user", f"[Uploaded file: {path.name}]")
            except (OSError, ValueError) as exc:
                self._status_label.setText(f"Upload failed: {exc}")

    def _on_screenshot(self) -> None:
        from typing import cast as _cast

        app = QApplication.instance()
        if not app:
            return
        parent = self.parentWidget()
        screen = _cast("QApplication", app).primaryScreen()
        if not screen:
            return
        pixmap = parent.grab() if parent else screen.grabWindow(0)  # type: ignore[arg-type]
        from PyQt6.QtCore import QBuffer, QByteArray, QIODevice

        ba = QByteArray()
        buffer = QBuffer(ba)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "PNG")
        b64 = base64.b64encode(ba.data()).decode("ascii")
        self._send_ws(
            {"action": "file_upload", "filename": "screenshot.png", "content": b64}
        )
        self._add_bubble("user", "[Captured screenshot]")

    def _on_mic_toggle(self) -> None:
        if self._voice_manager.is_recording:
            self._voice_manager.stop()
            self._mic_btn.setText("\U0001f3a4")
            self._mic_btn.setToolTip("Voice input (Ctrl+Shift+V)")
        else:
            self._voice_manager.start()
            self._mic_btn.setText("[REC]")
            self._mic_btn.setToolTip("Recording... click to stop (Ctrl+Shift+V)")
            self._status_label.setText("Listening...")

    def _on_voice_transcription(self, text: str) -> None:
        self._mic_btn.setText("\U0001f3a4")
        self._mic_btn.setToolTip("Voice input (Ctrl+Shift+V)")
        cursor = self._input_edit.textCursor()
        cursor.insertText(text)
        self._input_edit.setTextCursor(cursor)
        self._status_label.setText("Transcription complete")

    def _on_voice_error(self, message: str) -> None:
        self._mic_btn.setText("\U0001f3a4")
        self._mic_btn.setToolTip("Voice input (Ctrl+Shift+V)")
        self._status_label.setText(f"Voice: {message}")

    def _on_mode_changed(self) -> None:
        _runtime.on_mode_changed(self)

    def _current_mode(self) -> str:
        return _runtime.current_mode(self)

    def _set_terminal_runtime_available(self, available: bool) -> None:
        _runtime.set_terminal_runtime_available(self, available)

    def _sync_terminal_controls(self) -> None:
        _runtime.sync_terminal_controls(self)

    def _append_terminal_line(self, text: str) -> None:
        _runtime.append_terminal_line(self, text)

    # ── Messaging helpers ───────────────────────────────────────────

    def _send_ws(self, payload: dict) -> None:
        if self._socket and self._socket.isValid():
            self._socket.sendTextMessage(json.dumps(payload))

    def _format_agent_label(self) -> str:
        """Return the human-readable label for the current AI agent.

        Format: ``Agent (<model>)`` when the active model name is known,
        falling back to ``Agent (<provider>)`` when only the provider id
        is set, and finally to plain ``Agent`` when neither is populated.
        Centralised here so the bubble factory, header chrome, and any
        future call sites (e.g. status pill, tooltips) share one source
        of truth — DRY.

        DbC postcondition: returned string is non-empty.
        """
        model = (self._current_model or "").strip()
        provider = (self._current_provider or "").strip()
        if model:
            return f"Agent ({model})"
        if provider:
            return f"Agent ({provider})"
        return "Agent"

    def _add_bubble(self, role: str, content: str) -> ChatMessageBubble:
        """Add a message bubble to the scroll area.

        Assistant bubbles get an ``"Agent (<model>)"`` label sourced from
        the active provider/model state so users can tell which model
        produced each turn (e.g. ``Agent (llama3.1:8b)``,
        ``Agent (gpt-4o)``). User bubbles always show ``"You"``.
        """
        if role is None:
            raise ValueError("role must be provided")
        bubble = ChatMessageBubble(
            role,
            content,
            accent_color=self._accent_color,
            theme_provider=self._theme_provider,
            agent_label=self._format_agent_label() if role != "user" else None,
        )
        count = self._message_layout.count()
        self._message_layout.insertWidget(count - 1, bubble)
        self._scroll_to_bottom()
        return bubble

    def _populate_history(self, messages: list[dict]) -> None:
        """Clear and rebuild message bubbles from history."""
        if messages is None:
            raise ValueError("messages must be provided")
        while self._message_layout.count() > 1:
            item = self._message_layout.takeAt(0)
            if item:
                w = item.widget()
                if w:
                    w.deleteLater()
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                self._add_bubble(role, content)

    def _scroll_to_bottom(self) -> None:
        QTimer.singleShot(
            10,
            lambda: (
                scrollbar.setValue(scrollbar.maximum())
                if (scrollbar := self._scroll_area.verticalScrollBar()) is not None
                else None
            ),
        )

    # ── Export / condense (Tools issues #2735, #2736) ───────────────

    def _get_thread_markdown(self) -> str:
        return _exports.get_thread_markdown(self)

    def _copy_entire_thread(self) -> None:
        _exports.copy_entire_thread(self)

    def _export_to_markdown(self) -> None:
        """Backwards-compat shim retained for older history-browser code paths."""
        self._export_thread("markdown", "Markdown Files (*.md)", ".md")

    def _export_thread(self, fmt: str, file_filter: str, suffix: str) -> None:
        _exports.export_thread(self, fmt, file_filter, suffix)

    def _condense_thread(self) -> None:
        """Legacy server-side condense; retained for back-compat."""
        self._status_label.setText("Condensing thread...")
        self._send_ws({"action": "condense", "app_context": self._app_context})

    def _build_session_snapshot(self) -> Any:
        return _exports.build_session_snapshot(self)

    def _run_condense_local(self, strategy: str) -> None:
        _exports.run_condense_local(self, strategy)

    def _refresh_token_indicator(self) -> None:
        _exports.refresh_token_indicator(self)

    def _request_review(self) -> None:
        self._status_label.setText("Review requested...")
        self._send_ws({"action": "request_review", "provider": "openai"})

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
        with contextlib.suppress(AttributeError, RuntimeError):
            self._thinking_indicator.stop()
        with contextlib.suppress(AttributeError, RuntimeError):
            self._chunk_flush_timer.stop()
        with contextlib.suppress(AttributeError, RuntimeError):
            self._stop_state_timer.stop()
        if self._socket:
            self._socket.close()
        super().closeEvent(event)

    # ── Conversation management (Tools issue #2872) ─────────────────

    def _resolve_use_session_target(self, target: str) -> str | None:
        return _sessions.resolve_use_session_target(self, target)

    def _handle_use_session(self, target: str) -> None:
        sid = self._resolve_use_session_target(target)
        if sid is None:
            self._add_bubble("assistant", f"No matching session for '{target}'.")
            return
        self._add_context_session(sid)

    def _add_context_session(self, session_id: str) -> None:
        _sessions.add_context_session(self, session_id)

    def _remove_context_session(self, session_id: str) -> None:
        _sessions.remove_context_session(self, session_id)

    def breadcrumb_labels(self) -> list[str]:
        return _sessions.breadcrumb_labels(self)

    def _refresh_breadcrumb(self) -> None:
        _sessions.refresh_breadcrumb(self)

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
