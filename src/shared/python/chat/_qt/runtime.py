"""WebSocket and terminal runtime helpers for ``ChatDockWidget``.

The public widget retains its historical method surface while delegating
transport event routing and terminal-mode state transitions here.  Every
helper receives the dock explicitly so ownership stays visible and tests can
exercise the behavior without reaching through another object graph.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def initialize_streaming_state(
    dock: Any,
    *,
    timer_factory: Callable[[Any], Any],
) -> None:
    """Initialize streaming, queue, and watchdog state for a new dock."""
    dock._is_streaming = False
    dock._queued_messages = []
    dock._chunk_buffer = []
    dock._chunk_flush_timer = timer_factory(dock)
    dock._chunk_flush_timer.setSingleShot(True)
    dock._chunk_flush_timer.setInterval(50)
    dock._chunk_flush_timer.timeout.connect(dock._flush_chunk_buffer)
    dock._send_button_state = "idle"
    dock._last_chunk_at = None
    dock._stop_state_timer = timer_factory(dock)
    dock._stop_state_timer.setSingleShot(True)
    dock._stop_state_timer.setInterval(10_000)
    dock._stop_state_timer.timeout.connect(dock._on_stop_state_timeout)
    dock._current_bubble = None
    dock._terminal_session_id = None


def connect(
    dock: Any,
    *,
    native_connection: Callable[[str, str], tuple[str, str]],
    websocket_factory: Callable[[str], Any],
    url_factory: Callable[[str], Any],
) -> None:
    """Establish the dock's WebSocket connection."""
    dock._intentional_disconnect = False
    dock._is_closing = False
    if dock._socket is not None:
        dock._socket.close()
        dock._socket.deleteLater()
    sid = dock._get_shared_session_id() or "new"
    path = dock._ws_path_template.replace("{session_id}", sid)
    origin, url_text = native_connection(dock._server_url, path)
    dock._socket = websocket_factory(origin)
    dock._socket.connected.connect(dock._on_connected)
    dock._socket.disconnected.connect(dock._on_disconnected)
    dock._socket.textMessageReceived.connect(dock._on_message)

    dock._status_label.setText("Connecting...")
    dock._socket.open(url_factory(url_text))


def connection_diagnostics(dock: Any) -> dict[str, Any]:
    """Return host-readable WebSocket readiness diagnostics."""
    socket = dock._socket
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
        "server_url": dock._server_url,
        "ws_path_template": dock._ws_path_template,
        "session_id": dock._get_shared_session_id(),
        "socket_state": state,
        "error": error,
        "connect_on_show": bool(getattr(dock, "_connect_on_show", False)),
    }


def on_disconnected(dock: Any) -> None:
    """Apply the dock's intentional or unexpected disconnect policy."""
    if bool(getattr(dock, "_intentional_disconnect", False)) or bool(
        getattr(dock, "_is_closing", False)
    ):
        _exit_thinking_state(dock)
        return
    dock._status_label.setText(
        "Sidekick API unavailable — retrying in 3s. "
        "Set UD_CHAT_WS_URL if the local API is external."
    )
    dock._status_label.setStyleSheet("color: #f85149; font-size: 10px;")
    _exit_thinking_state(dock)
    dock._reconnect_timer.start(3000)


def _exit_thinking_state(dock: Any) -> None:
    """Exit streaming with compatibility for partially constructed test docks."""
    if hasattr(dock, "_exit_thinking_state"):
        dock._exit_thinking_state()
        return
    dock._is_streaming = False
    if hasattr(dock, "_send_btn") and dock._send_btn is not None:
        dock._send_btn.setEnabled(True)


def _decode_message(raw: str) -> tuple[bool, Any]:
    """Decode one message while preserving the legacy malformed-input policy."""
    if raw is None:
        raise ValueError("raw must be provided")
    try:
        return True, json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False, None


def handle_message(
    dock: Any,
    raw: str,
    *,
    set_shared_session_id: Callable[[str | None], None],
    write_shared_session_id: Callable[[str, Path], None],
) -> None:
    """Route one incoming WebSocket message to the dock state machine."""
    decoded, data = _decode_message(raw)
    if not decoded:
        return

    msg_type = data.get("type")
    if msg_type == "session_info":
        sid = data.get("session_id", "")
        set_shared_session_id(sid)
        write_shared_session_id(sid, dock._session_file)
        capabilities = data.get("capabilities", {})
        dock._set_terminal_runtime_available(
            bool(
                isinstance(capabilities, dict) and capabilities.get("terminal_runtime")
            )
        )
        dock._send_ws({"action": "history"})
    elif msg_type == "chunk":
        content = data.get("content", "")
        if dock._current_bubble and content:
            dock._chunk_buffer.append(content)
            if not dock._chunk_flush_timer.isActive():
                dock._chunk_flush_timer.start()
            import time as _time

            dock._last_chunk_at = _time.monotonic()
            if dock._is_streaming:
                dock._stop_state_timer.start()
    elif msg_type == "complete":
        dock._flush_chunk_buffer()
        dock._stop_state_timer.stop()
        dock._exit_thinking_state()
        dock._current_bubble = None
        sid = data.get("session_id")
        if sid:
            set_shared_session_id(sid)
            write_shared_session_id(sid, dock._session_file)
        dock._flush_queued_messages()
    elif msg_type == "session_created":
        sid = data.get("session_id", "")
        set_shared_session_id(sid)
        write_shared_session_id(sid, dock._session_file)
        while dock._message_layout.count() > 1:
            item = dock._message_layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        dock._message_history = []
        dock._add_bubble("assistant", "Hello! How can I help you today?")
        if hasattr(dock, "_history_sidebar") and dock._history_sidebar is not None:
            dock._history_sidebar.refresh_lists()
    elif msg_type == "history":
        dock._populate_history(data.get("messages", []))
    elif msg_type == "model_list":
        models = data.get("models", [])
        if isinstance(models, list):
            dock.models_refreshed.emit(models)
    elif msg_type == "index_status":
        dock.index_status_changed.emit(dict(data))
        state = data.get("state")
        if state == "running":
            files = data.get("files_parsed", 0)
            dock._status_label.setText(f"Indexing codebase ({files} files)...")
        elif state == "complete":
            dock._status_label.setText("Connected")
            dock._status_label.setStyleSheet("color: #3fb950; font-size: 10px;")
        elif state == "error":
            detail = data.get("error", "Unknown indexing error")
            dock._status_label.setText(f"Index error: {detail}")
            dock._status_label.setStyleSheet("color: #f85149; font-size: 10px;")
    elif msg_type == "error":
        detail = data.get("detail", "Unknown error")
        dock._status_label.setText(f"Error: {detail}")
        dock._exit_thinking_state()
        dock._update_queue_affordance()
    elif msg_type == "terminal_session":
        session = data.get("session", {})
        dock._terminal_session_id = session.get("session_id")
        dock._terminal_start_pending = False
        state = session.get("state", "unknown")
        dock._status_label.setText(f"Terminal {state}")
        if dock._terminal_session_id:
            dock._append_terminal_line(f"[terminal] session {state}")
        if state in {"stopped", "exited", "error"}:
            dock._terminal_session_id = None
        dock._sync_terminal_controls()
    elif msg_type == "terminal_events":
        for event in data.get("events", []):
            dock._append_terminal_line(event.get("data", ""))
    elif msg_type == "terminal_ack":
        dock._status_label.setText("Terminal input sent")


def populate_shell_combo(dock: Any) -> None:
    """Populate the terminal shell selector from the injected registry."""
    dock._shell_combo.clear()
    for shell in dock._terminal_registry.shells():
        dock._shell_combo.addItem(shell.display_name, shell.id)


def populate_provider_combo(dock: Any) -> None:
    """Populate providers compatible with the selected terminal shell."""
    shell_id = str(dock._shell_combo.currentData() or "")
    providers = dock._terminal_registry.providers_for_shell(shell_id)
    current_provider = dock._provider_combo.currentData()
    dock._provider_combo.blockSignals(True)
    try:
        dock._provider_combo.clear()
        for provider in providers:
            dock._provider_combo.addItem(provider.display_name, provider.id)
        if current_provider:
            index = dock._provider_combo.findData(current_provider)
            if index >= 0:
                dock._provider_combo.setCurrentIndex(index)
    finally:
        dock._provider_combo.blockSignals(False)
    dock._sync_terminal_controls()


def on_terminal_start(dock: Any) -> None:
    """Validate terminal readiness and request a new terminal session."""
    if not dock._terminal_runtime_available:
        dock._append_terminal_line("[terminal] host has not enabled terminal runtime")
        return
    if dock._terminal_session_id or dock._terminal_start_pending:
        dock._append_terminal_line("[terminal] session already active")
        return
    if not dock._shell_combo.currentData() or not dock._provider_combo.currentData():
        dock._append_terminal_line("[terminal] select a shell and provider first")
        return
    dock._terminal_start_pending = True
    dock._sync_terminal_controls()
    dock._terminal_output.clear()
    dock._append_terminal_line("[terminal] starting...")
    dock._send_ws(
        {
            "action": "terminal_start",
            "project_root": str(dock._project_root),
            "shell_id": dock._shell_combo.currentData(),
            "provider_id": dock._provider_combo.currentData(),
            "app_context": dock._app_context,
        }
    )


def on_terminal_stop(dock: Any) -> None:
    """Request termination of the active terminal session."""
    if not dock._terminal_session_id:
        dock._append_terminal_line("[terminal] start a session first")
        return
    dock._send_ws(
        {
            "action": "terminal_stop",
            "terminal_session_id": dock._terminal_session_id,
        }
    )


def on_terminal_input(dock: Any, text: str) -> None:
    """Send one line of input to the active terminal session."""
    if not dock._terminal_session_id:
        dock._append_terminal_line("[terminal] start a session first")
        return
    dock._input_edit.clear()
    dock._append_terminal_line(f"> {text}")
    dock._send_ws(
        {
            "action": "terminal_input",
            "terminal_session_id": dock._terminal_session_id,
            "text": f"{text}\n",
        }
    )


def current_mode(dock: Any) -> str:
    """Return the selected dock mode, defaulting to chat."""
    mode = dock._mode_combo.currentData()
    return str(mode or "chat")


def on_mode_changed(dock: Any) -> None:
    """Synchronize visible controls with the selected runtime mode."""
    is_terminal = current_mode(dock) == "terminal" and dock._terminal_runtime_available
    dock._content_stack.setCurrentIndex(1 if is_terminal else 0)
    dock._shell_combo.setVisible(is_terminal)
    dock._provider_combo.setVisible(is_terminal)
    dock._terminal_start_btn.setVisible(is_terminal)
    dock._terminal_stop_btn.setVisible(is_terminal)
    dock._sync_terminal_controls()
    placeholder = "Type terminal input..." if is_terminal else dock._placeholder_text
    dock._input_edit.setPlaceholderText(placeholder)


def set_terminal_runtime_available(dock: Any, available: bool) -> None:
    """Expose or remove terminal mode according to server capabilities."""
    dock._terminal_runtime_available = bool(available)
    if not hasattr(dock, "_mode_combo"):
        if not dock._terminal_runtime_available:
            dock._terminal_session_id = None
            dock._terminal_start_pending = False
        return
    terminal_index = dock._mode_combo.findData("terminal")
    if dock._terminal_runtime_available:
        if terminal_index < 0:
            dock._mode_combo.addItem("Terminal", "terminal")
    else:
        if terminal_index >= 0:
            if dock._current_mode() == "terminal":
                chat_index = dock._mode_combo.findData("chat")
                dock._mode_combo.setCurrentIndex(max(0, chat_index))
            dock._mode_combo.removeItem(terminal_index)
        dock._terminal_session_id = None
        dock._terminal_start_pending = False
    dock._sync_terminal_controls()
    dock._on_mode_changed()


def sync_terminal_controls(dock: Any) -> None:
    """Enable terminal controls from the current session state."""
    if not hasattr(dock, "_terminal_start_btn"):
        return
    active = bool(dock._terminal_session_id)
    pending = bool(dock._terminal_start_pending)
    startable = (
        dock._terminal_runtime_available
        and not active
        and not pending
        and bool(dock._shell_combo.currentData())
        and bool(dock._provider_combo.currentData())
    )
    dock._terminal_start_btn.setEnabled(startable)
    dock._terminal_stop_btn.setEnabled(active)
    dock._shell_combo.setEnabled(not active and not pending)
    dock._provider_combo.setEnabled(not active and not pending)


def append_terminal_line(dock: Any, text: str) -> None:
    """Append a non-empty line to the terminal output."""
    if text:
        dock._terminal_output.appendPlainText(text)


def set_chat_dock_collapsed(dock: Any, collapsed: bool) -> None:
    """Apply the compact or expanded visibility state to the chat dock."""
    dock._collapsed = collapsed
    widgets_to_hide = [
        dock._status_label,
        dock._tools_btn,
        dock._token_indicator,
        dock._ai_provider_combo,
        dock._ai_model_combo,
        dock._ai_thinking_combo,
        dock._mode_combo,
        dock._content_stack,
        dock._input_edit,
        dock._upload_btn,
        dock._screenshot_btn,
        dock._mic_btn,
        dock._agent_mode_combo,
        dock._send_btn,
        dock._steer_btn,
        dock._stop_agent_btn,
    ]
    terminal_widgets = [
        dock._shell_combo,
        dock._provider_combo,
        dock._terminal_start_btn,
        dock._terminal_stop_btn,
    ]

    if collapsed:
        for widget in widgets_to_hide + terminal_widgets:
            if widget is not None:
                widget.setVisible(False)
    else:
        for widget in widgets_to_hide:
            if widget is not None:
                widget.setVisible(True)
        is_terminal = dock._current_mode() == "terminal"
        for widget in terminal_widgets:
            if widget is not None:
                widget.setVisible(is_terminal)

    dock.updateGeometry()


__all__ = [
    "append_terminal_line",
    "connect",
    "connection_diagnostics",
    "current_mode",
    "handle_message",
    "initialize_streaming_state",
    "on_disconnected",
    "on_mode_changed",
    "on_terminal_input",
    "on_terminal_start",
    "on_terminal_stop",
    "populate_provider_combo",
    "populate_shell_combo",
    "set_chat_dock_collapsed",
    "set_terminal_runtime_available",
    "sync_terminal_controls",
]
