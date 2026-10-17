# mypy: ignore-errors
"""Terminal-mode controls for the chat dock.

Extracted verbatim from ``chat._chat_dock_widget_qt`` (Tools issue #5641) so
the dock module fits UpstreamDrift's 1200-line budget. The mixin carries no
state of its own: it operates on attributes that ``ChatDockWidget.__init__``
sets up, and is only meant to be mixed into that class.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TerminalModeMixin:
    """Terminal-mode controls for the chat dock."""

    # ── Terminal mode ───────────────────────────────────────────────

    def _populate_shell_combo(self) -> None:
        self._shell_combo.clear()
        for shell in self._terminal_registry.shells():
            self._shell_combo.addItem(shell.display_name, shell.id)

    def _populate_provider_combo(self) -> None:
        shell_id = str(self._shell_combo.currentData() or "")
        providers = self._terminal_registry.providers_for_shell(shell_id)
        current_provider = self._provider_combo.currentData()
        self._provider_combo.blockSignals(True)
        try:
            self._provider_combo.clear()
            for provider in providers:
                self._provider_combo.addItem(provider.display_name, provider.id)
            if current_provider:
                idx = self._provider_combo.findData(current_provider)
                if idx >= 0:
                    self._provider_combo.setCurrentIndex(idx)
        finally:
            self._provider_combo.blockSignals(False)
        self._sync_terminal_controls()

    def _on_terminal_shell_changed(self, _index: int) -> None:
        self._populate_provider_combo()

    def _on_terminal_start(self) -> None:
        if not self._terminal_runtime_available:
            self._append_terminal_line(
                "[terminal] host has not enabled terminal runtime"
            )
            return
        if self._terminal_session_id or self._terminal_start_pending:
            self._append_terminal_line("[terminal] session already active")
            return
        if (
            not self._shell_combo.currentData()
            or not self._provider_combo.currentData()
        ):
            self._append_terminal_line("[terminal] select a shell and provider first")
            return
        self._terminal_start_pending = True
        self._sync_terminal_controls()
        self._terminal_output.clear()
        self._append_terminal_line("[terminal] starting...")
        self._send_ws(
            {
                "action": "terminal_start",
                "project_root": str(self._project_root),
                "shell_id": self._shell_combo.currentData(),
                "provider_id": self._provider_combo.currentData(),
                "app_context": self._app_context,
            }
        )

    def _on_terminal_stop(self) -> None:
        if not self._terminal_session_id:
            self._append_terminal_line("[terminal] start a session first")
            return
        self._send_ws(
            {
                "action": "terminal_stop",
                "terminal_session_id": self._terminal_session_id,
            }
        )

    def _on_terminal_input(self, text: str) -> None:
        if not self._terminal_session_id:
            self._append_terminal_line("[terminal] start a session first")
            return
        self._input_edit.clear()
        self._append_terminal_line(f"> {text}")
        self._send_ws(
            {
                "action": "terminal_input",
                "terminal_session_id": self._terminal_session_id,
                "text": f"{text}\n",
            }
        )

    def _on_mode_changed(self) -> None:
        is_terminal = (
            self._current_mode() == "terminal" and self._terminal_runtime_available
        )
        self._content_stack.setCurrentIndex(1 if is_terminal else 0)
        self._shell_combo.setVisible(is_terminal)
        self._provider_combo.setVisible(is_terminal)
        self._terminal_start_btn.setVisible(is_terminal)
        self._terminal_stop_btn.setVisible(is_terminal)
        self._sync_terminal_controls()
        placeholder = (
            "Type terminal input..." if is_terminal else self._placeholder_text
        )
        self._input_edit.setPlaceholderText(placeholder)

    def _current_mode(self) -> str:
        mode = self._mode_combo.currentData()
        return str(mode or "chat")

    def _set_terminal_runtime_available(self, available: bool) -> None:
        self._terminal_runtime_available = bool(available)
        if not hasattr(self, "_mode_combo"):
            if not self._terminal_runtime_available:
                self._terminal_session_id = None
                self._terminal_start_pending = False
            return
        terminal_index = self._mode_combo.findData("terminal")
        if self._terminal_runtime_available:
            if terminal_index < 0:
                self._mode_combo.addItem("Terminal", "terminal")
        else:
            if terminal_index >= 0:
                if self._current_mode() == "terminal":
                    chat_index = self._mode_combo.findData("chat")
                    self._mode_combo.setCurrentIndex(max(0, chat_index))
                self._mode_combo.removeItem(terminal_index)
            self._terminal_session_id = None
            self._terminal_start_pending = False
        self._sync_terminal_controls()
        self._on_mode_changed()

    def _sync_terminal_controls(self) -> None:
        if not hasattr(self, "_terminal_start_btn"):
            return
        active = bool(self._terminal_session_id)
        pending = bool(self._terminal_start_pending)
        startable = (
            self._terminal_runtime_available
            and not active
            and not pending
            and bool(self._shell_combo.currentData())
            and bool(self._provider_combo.currentData())
        )
        self._terminal_start_btn.setEnabled(startable)
        self._terminal_stop_btn.setEnabled(active)
        self._shell_combo.setEnabled(not active and not pending)
        self._provider_combo.setEnabled(not active and not pending)

    def _append_terminal_line(self, text: str) -> None:
        if text:
            self._terminal_output.appendPlainText(text)
