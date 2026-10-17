# mypy: ignore-errors
"""Messaging, input affordances, export and session helpers for the chat dock.

Extracted verbatim from ``chat._chat_dock_widget_qt`` (Tools issue #5641) so
the dock module fits UpstreamDrift's 1200-line budget. The mixin carries no
state of its own: it operates on attributes that ``ChatDockWidget.__init__``
sets up, and is only meant to be mixed into that class.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QFileDialog

from . import exports as _exports
from . import sessions as _sessions
from .bubbles import ChatMessageBubble

logger = logging.getLogger(__name__)


class MessagingMixin:
    """Messaging, input affordances, export and session helpers for the chat dock."""

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
