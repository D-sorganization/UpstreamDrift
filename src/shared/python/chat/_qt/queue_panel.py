# ruff: noqa: E501
"""Inline queue-preview panel for the shared chat dock (Tools chat UX).

When the user types and presses Enter while the agent is still
streaming a response, the dock's :class:`ChatDockWidget` queues the
message via ``_submit_or_queue``. This module renders that queue as a
compact panel above the input area so each pending message is visible
and individually steerable.

Design
------
- **DbC**: every public method documents its preconditions and
  postconditions, and validates inputs raising ``ValueError`` /
  ``TypeError`` on misuse.
- **LOD**: the panel emits Qt signals; it never reaches into the parent
  dock. Wiring lives in the dock's ``_qt`` package and the dock body.
- **DRY**: a single :class:`QueuePanel` instance is built by the UI
  builder and reused for the lifetime of the dock; tests construct it
  directly when they need to assert behaviour in isolation.

The "steer" verb here matches the steer protocol used by the keybinding
agent: clicking a queued message's button moves that message to the
*front* of the queue so it dispatches first on the next flush.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


@dataclass
class QueuedMessage:
    """One queued steering message.

    Attributes:
        text: The exact text the user typed (already ``strip()``-ed).
        id: A stable unique identifier so the panel can address a row
            without depending on its position in the list (positions
            shift when the user steers).
        created_at: Monotonic timestamp (``time.monotonic()``) at the
            moment the message was queued. Used for ordering and for
            future age-based affordances.
    """

    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.monotonic)


class QueuePanel(QFrame):
    """Compact, inline preview of the busy-state message queue.

    Public Qt signals:
        * ``steer_requested(str)`` — emitted when the user clicks the
          per-row steer button. The payload is :attr:`QueuedMessage.id`.

    Public API:
        * :meth:`set_messages` — replace the rendered list. Idempotent.
        * :meth:`clear` — remove all rows and hide the panel.
        * :attr:`row_count` — number of rendered rows (read-only).

    The panel hides itself when ``set_messages([])`` is called so the
    chrome only appears while there is something to preview.
    """

    steer_requested = pyqtSignal(str)

    _ROW_BUTTON_WIDTH: int = 56
    _MAX_PREVIEW_CHARS: int = 80

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ChatQueuePanel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self._layout = layout
        self._rows: list[tuple[QueuedMessage, QWidget]] = []
        # Start collapsed — only show when there is something queued.
        self.hide()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setStyleSheet(
            "QFrame#ChatQueuePanel {"
            "  background-color: rgba(255, 200, 0, 0.06);"
            "  border: 1px solid rgba(255, 200, 0, 0.4);"
            "  border-radius: 4px;"
            "}"
            "QLabel#ChatQueueRowLabel { color: #d0d0d0; font-size: 11px; }"
            "QPushButton#ChatQueueSteerBtn {"
            "  background-color: #c79100; color: black;"
            "  border-radius: 3px; padding: 2px 6px; font-size: 10px;"
            "  font-weight: bold;"
            "}"
            "QPushButton#ChatQueueSteerBtn:hover { background-color: #e7b800; }"
        )

    # ── public API ────────────────────────────────────────────────────

    @property
    def row_count(self) -> int:
        """Number of currently rendered rows."""
        return len(self._rows)

    def set_messages(self, messages: list[QueuedMessage]) -> None:
        """Render ``messages`` as the panel's current list.

        DbC:
            Pre: ``messages`` is a (possibly empty) ``list`` of
                :class:`QueuedMessage` instances. ``None`` is not
                permitted — callers pass an explicit empty list to clear.
            Post: ``row_count`` equals ``len(messages)`` and the panel
                is visible iff ``messages`` is non-empty.
        """
        if messages is None:
            raise ValueError("set_messages: messages must be a list, not None")
        if not isinstance(messages, list):
            raise TypeError("set_messages: messages must be a list")
        for m in messages:
            if not isinstance(m, QueuedMessage):
                raise TypeError(
                    "set_messages: every entry must be a QueuedMessage instance"
                )

        # Clear previous rows. ``deleteLater`` is the safe Qt teardown.
        while self._rows:
            _, row_widget = self._rows.pop()
            self._layout.removeWidget(row_widget)
            row_widget.deleteLater()

        for msg in messages:
            row = self._build_row(msg)
            self._layout.addWidget(row)
            self._rows.append((msg, row))

        self.setVisible(bool(messages))

    def clear(self) -> None:
        """Convenience: equivalent to ``set_messages([])``."""
        self.set_messages([])

    # ── internals ─────────────────────────────────────────────────────

    def _build_row(self, msg: QueuedMessage) -> QWidget:
        row = QWidget(self)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(4, 2, 4, 2)
        row_layout.setSpacing(6)

        preview = msg.text.replace("\n", " ")
        if len(preview) > self._MAX_PREVIEW_CHARS:
            preview = preview[: self._MAX_PREVIEW_CHARS - 1] + "…"

        label = QLabel(preview, row)
        label.setObjectName("ChatQueueRowLabel")
        label.setToolTip(msg.text)
        label.setWordWrap(False)
        row_layout.addWidget(label, stretch=1)

        btn = QPushButton("Steer", row)
        btn.setObjectName("ChatQueueSteerBtn")
        btn.setFixedWidth(self._ROW_BUTTON_WIDTH)
        btn.setToolTip("Move this message to the front of the queue")
        btn.clicked.connect(lambda _checked=False, mid=msg.id: self._on_steer(mid))
        row_layout.addWidget(btn)
        return row

    def _on_steer(self, message_id: str) -> None:
        # DbC: id must be a non-empty string at this point — the dataclass
        # default uses ``uuid4`` so this should always hold; the guard is
        # cheap and protects test stubs that bypass the constructor.
        if not isinstance(message_id, str) or not message_id:
            raise ValueError("_on_steer: message_id must be a non-empty string")
        self.steer_requested.emit(message_id)


__all__ = ["QueuePanel", "QueuedMessage"]
