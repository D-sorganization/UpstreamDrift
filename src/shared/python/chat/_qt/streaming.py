# mypy: ignore-errors
"""Streaming, send-button, queue and slash-command behaviour for the chat dock.

Extracted verbatim from ``chat._chat_dock_widget_qt`` (Tools issue #5641) so
the dock module fits UpstreamDrift's 1200-line budget. The mixin carries no
state of its own: it operates on attributes that ``ChatDockWidget.__init__``
sets up, and is only meant to be mixed into that class.
"""

from __future__ import annotations

import logging
from typing import Any

from .._thinking_indicator import ThinkingIndicator
from .queue_panel import QueuedMessage

logger = logging.getLogger(__name__)


class StreamingMixin:
    """Streaming, send-button, queue and slash-command behaviour for the chat dock."""

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
        try:
            self._thinking_indicator.start()
        except AttributeError:
            # Indicator widget not yet constructed (very early init / tests
            # that bypass _setup_ui) — silently skip.
            pass
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
        try:
            self._send_btn.setEnabled(True)
        except AttributeError:
            pass
        try:
            self._thinking_indicator.stop()
        except AttributeError:
            pass
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
