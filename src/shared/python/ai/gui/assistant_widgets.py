"""Small reusable widgets for the shared AI assistant panel."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from PyQt6 import QtGui
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ai.types import ConversationContext
from src.shared.python.compatibility import UTC
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.style_constants import Styles

if TYPE_CHECKING:
    from shared.python.ai.adapters.base import BaseAgentAdapter

logger = get_logger(__name__)


class MessageWidget(QFrame):
    """Widget displaying a single message in the conversation."""

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: datetime | None = None,
        parent: QWidget | None = None,
    ) -> None:
        if role is None:
            raise ValueError("role must be provided")
        if role is None:
            raise ValueError("role must be provided")
        super().__init__(parent)
        self._role = role
        self._content = content
        self._timestamp = timestamp or datetime.now(UTC)
        self._setup_ui()
        self._apply_style()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        header = QHBoxLayout()

        role_label = QLabel(self._get_role_display())
        role_label.setStyleSheet(Styles.TEXT_LABEL_BOLD_WHITE)
        header.addWidget(role_label)

        header.addStretch()

        time_label = QLabel(self._timestamp.strftime("%H:%M"))
        time_label.setStyleSheet(Styles.TEXT_MUTED)
        header.addWidget(time_label)

        self._copy_btn = QToolButton()
        self._copy_btn.setText("Copy")
        self._copy_btn.setToolTip("Copy message to clipboard")
        self._copy_btn.setStyleSheet(
            "QToolButton { color: #aaaaaa; border: none; padding: 2px 4px; }"
            "QToolButton:hover { color: #ffffff; }"
        )
        self._copy_btn.clicked.connect(self._on_copy_clicked)
        header.addWidget(self._copy_btn)

        layout.addLayout(header)

        self._content_label = QTextEdit()
        self._content_label.setReadOnly(True)
        self._content_label.setFrameShape(QFrame.Shape.NoFrame)
        self._content_label.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._content_label.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._content_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self._content_label.setMarkdown(self._content)
        self._content_label.setStyleSheet(Styles.TEXT_CONTENT_TRANSPARENT)

        doc = self._content_label.document()
        if doc is not None:
            doc.contentsChanged.connect(self._adjust_height)
        self._adjust_height()

        layout.addWidget(self._content_label)

    def _get_role_display(self) -> str:
        """Get display name for role."""
        role_map = {
            "user": "You",
            "assistant": "AI Assistant",
            "system": "System",
            "tool": "Tool Result",
        }
        return role_map.get(self._role, self._role.title())

    def _apply_style(self) -> None:
        """Apply styling based on role and current theme."""
        self.refresh_theme()

    def refresh_theme(self) -> None:
        """Refresh colors from ThemeManager."""
        try:
            from shared.python.theme.theme_manager import get_theme_manager

            color_source: object = get_theme_manager().get_current_colors()

            def _get(key: str, fallback: Any) -> Any:
                if isinstance(color_source, dict):
                    return color_source.get(key, fallback)
                return getattr(color_source, key, fallback)

            bg_alt = _get("bg_elevated", _get("group_bg", "#2d2d2d"))
            bg_secondary = _get("bg_highlight", _get("input_bg", "#252526"))
            text_primary = _get("text_primary", _get("text", "#e0e0e0"))
        except ImportError:
            bg_alt = "#2d2d2d"
            bg_secondary = "#252526"
            text_primary = "#e0e0e0"

        bg = bg_alt if self._role == "user" else bg_secondary
        self.setStyleSheet(
            f"MessageWidget {{ background-color: {bg}; border-radius: 6px; }}"
        )
        self._content_label.setStyleSheet(
            f"color: {text_primary}; background: transparent; border: none;"
        )

    def _adjust_height(self) -> None:
        """Adjust height to fit content."""
        doc = self._content_label.document()
        if doc is not None:
            doc_height = doc.size().height()
            self._content_label.setFixedHeight(int(doc_height) + 10)

    def append_content(self, text: str) -> None:
        """Append content to the message for streaming."""
        if text is None:
            raise ValueError("text must be provided")
        if text is None:
            raise ValueError("text must be provided")
        self._content += text
        self._content_label.setMarkdown(self._content)

    def set_content(self, text: str) -> None:
        """Set message content."""
        if text is None:
            raise ValueError("text must be provided")
        if text is None:
            raise ValueError("text must be provided")
        self._content = text
        self._content_label.setMarkdown(self._content)

    def get_content(self) -> str:
        """Get current content."""
        return self._content

    def _on_copy_clicked(self) -> None:
        """Copy this message's raw text to the system clipboard."""
        from shared.python.ai.gui.chat_export import copy_message_to_clipboard

        copy_message_to_clipboard(self._content)
        original = self._copy_btn.text()
        self._copy_btn.setText("Copied!")
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(1500, lambda: self._copy_btn.setText(original))


class StreamWorker(QThread):
    """Worker thread for streaming AI responses."""

    chunk_received = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        adapter: BaseAgentAdapter,
        message: str,
        context: ConversationContext,
        tools: list[Any],
    ) -> None:
        if adapter is None:
            raise ValueError("adapter must be provided")
        if adapter is None:
            raise ValueError("adapter must be provided")
        super().__init__()
        self._adapter = adapter
        self._message = message
        self._context = context
        self._tools = tools

    def run(self) -> None:
        """Execute streaming in background thread."""
        try:
            for chunk in self._adapter.stream_response(
                self._message,
                self._context,
                self._tools,
            ):
                if chunk.content:
                    self.chunk_received.emit(chunk.content)
        except (RuntimeError, ValueError, OSError) as e:
            logger.exception("Streaming error")
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class _ThunkCall:
    """Cross-thread carrier for one main-thread dispatch.

    Holds the thunk to run plus slots for its result/error and an
    ``Event`` the calling (worker) thread waits on.
    """

    __slots__ = ("thunk", "result", "error", "done")

    def __init__(self, thunk: Callable[[], Any]) -> None:
        self.thunk = thunk
        self.result: Any = None
        self.error: BaseException | None = None
        self.done = threading.Event()


class MainThreadToolDispatcher(QObject):
    """Runs GUI-thread-affine tool handlers on the GUI thread.

    The chat executes tools from a background :class:`StreamWorker`
    thread. Handlers that mutate Qt widgets must run on the thread that
    owns those widgets; this dispatcher marshals such a call across the
    thread boundary and returns its result synchronously (the agent loop
    needs the ``ToolResult`` before continuing).

    Install on the registry::

        registry.set_main_thread_dispatcher(MainThreadToolDispatcher(panel))

    The instance must live on the GUI thread (pass a GUI-thread parent,
    or construct it there). Calling it from the GUI thread runs the thunk
    inline, so it never deadlocks against itself.
    """

    _dispatch = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # Default AutoConnection is exactly what we want: a cross-thread
        # emit (from the worker) is delivered as a queued call on this
        # object's owning (GUI) thread, while a same-thread emit runs
        # directly. ``__call__`` short-circuits same-thread callers anyway.
        self._dispatch.connect(self._run)

    def __call__(self, thunk: Callable[[], Any]) -> Any:
        """Run ``thunk`` on the GUI thread and return its result.

        Args:
            thunk: Zero-argument callable to execute on the GUI thread.

        Returns:
            Whatever ``thunk`` returns.

        Raises:
            Whatever ``thunk`` raises (re-raised on the calling thread).
        """
        if QThread.currentThread() is self.thread():
            return thunk()
        call = _ThunkCall(thunk)
        self._dispatch.emit(call)
        call.done.wait()
        if call.error is not None:
            raise call.error
        return call.result

    def _run(self, call: _ThunkCall) -> None:
        try:
            call.result = call.thunk()
        except Exception as exc:  # noqa: BLE001 - propagate across threads
            call.error = exc
        finally:
            call.done.set()


class ChatInput(QPlainTextEdit):
    """Custom input widget handling Send vs Newline."""

    submit_requested = pyqtSignal()

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        """Handle key press events."""
        if event is None:
            return
        if (
            event.key() == Qt.Key.Key_Return
            and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            event.accept()
            self.submit_requested.emit()
        else:
            super().keyPressEvent(event)
