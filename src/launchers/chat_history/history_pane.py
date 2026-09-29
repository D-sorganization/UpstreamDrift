"""Chat history sidebar pane for the launcher's chat panel.

Renders two stacked sections — recent (date-grouped) and archived
(collapsible) — fed by a :class:`HistoryServiceAdapter`. Includes a
debounced search field that calls the adapter's full-text search and a
per-conversation action hook (Restore / Archive / Delete / Export /
Load as context) that emits Qt signals so the parent panel can react.

UI never reaches into Sidekick internals — every action flows through
the adapter (Law of Demeter).
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.launchers.chat_history._formatting import (
    format_conversation_preview,
    group_by_date,
)
from src.launchers.chat_history.chat_history_service import (
    HistoryServiceAdapter,
)


class HistorySidebarPane(QWidget):
    """Sidebar widget showing recent + archived conversations.

    Signals
    -------
    context_loaded(dict)
        Emitted after a successful ``load_as_context`` so the parent
        chat panel can swap its live conversation.
    conversation_action(str, str)
        Emitted for any per-conversation action with
        ``(conversation_id, action)``. Useful for higher-level routing.
    """

    context_loaded = pyqtSignal(dict)
    conversation_action = pyqtSignal(str, str)

    def __init__(
        self,
        adapter: HistoryServiceAdapter,
        parent: QWidget | None = None,
        debounce_ms: int = 250,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ChatHistorySidebarPane")
        self._adapter = adapter
        self._debounce_ms = max(0, int(debounce_ms))

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._run_search)

        self._recent_browser = QTextBrowser()
        self._recent_browser.setOpenExternalLinks(False)
        self._archived_browser = QTextBrowser()
        self._archived_browser.setOpenExternalLinks(False)

        self._search_field = QLineEdit()
        self._search_field.setPlaceholderText("Search conversations…")
        self._search_field.textChanged.connect(self._on_search_text_changed)

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.addWidget(QLabel("History"))
        header.addStretch()
        new_btn = QPushButton("⊕ New")
        new_btn.setToolTip("Start a new conversation")
        new_btn.setObjectName("ChatHistoryNewButton")
        header.addWidget(new_btn)
        layout.addLayout(header)

        layout.addWidget(self._search_field)

        layout.addWidget(QLabel("Recent"))
        layout.addWidget(self._recent_browser, 2)

        self._archived_label = QLabel("Archived")
        layout.addWidget(self._archived_label)
        layout.addWidget(self._archived_browser, 1)

    # ------------------------------------------------------------------
    # Public API (used by parent panel + tests)
    # ------------------------------------------------------------------

    @property
    def search_field(self) -> QLineEdit:
        """Return the embedded search ``QLineEdit`` (test hook)."""
        return self._search_field

    def refresh(self) -> None:
        """Reload conversations from the adapter and re-render."""
        try:
            active = self._adapter.list_active()
        except RuntimeError:
            active = []
        try:
            archived = self._adapter.list_archived()
        except RuntimeError:
            archived = []
        self._render_recent(active)
        self._render_archived(archived)

    def recent_text(self) -> str:
        """Return the plain-text representation of the recent section."""
        return self._recent_browser.toPlainText()

    def archived_text(self) -> str:
        """Return the plain-text representation of the archived section."""
        return self._archived_browser.toPlainText()

    def flush_pending_search(self) -> None:
        """Force any pending debounced search to run immediately.

        Exposed so tests don't have to spin the Qt event loop.
        """
        if self._search_timer.isActive():
            self._search_timer.stop()
        self._run_search()

    def trigger_action(
        self,
        conversation_id: str,
        action: str,
        *,
        target: str | None = None,
    ) -> None:
        """Dispatch a per-conversation hover-bar action.

        Supported actions: ``restore``, ``archive``, ``delete``,
        ``export``, ``load_as_context``.

        Precondition: ``action`` must be in the supported set.
        """
        if action == "restore":
            self._adapter.restore(conversation_id)
        elif action == "archive":
            self._adapter.archive(conversation_id)
        elif action == "delete":
            self._adapter.delete(conversation_id)
        elif action == "export":
            export_target = target or ""
            self._adapter.export(conversation_id, export_target)
        elif action == "load_as_context":
            payload = self._adapter.load_as_context(conversation_id)
            self.context_loaded.emit(payload)
        else:
            raise ValueError(f"unknown action: {action!r}")

        self.conversation_action.emit(conversation_id, action)

        # After any state-changing action, reload to reflect the new list.
        if action in {"restore", "archive", "delete"}:
            self.refresh()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_search_text_changed(self, _text: str) -> None:
        if self._debounce_ms <= 0:
            # Even at zero we let the caller drive via flush_pending_search()
            # in tests, but we also start the (immediate) timer so the
            # production path still works without manual flushing.
            self._search_timer.start(0)
            return
        self._search_timer.start(self._debounce_ms)

    def _run_search(self) -> None:
        query = self._search_field.text().strip()
        if not query:
            # Empty query reverts to the standard recent listing.
            self.refresh()
            return
        try:
            results = self._adapter.search(query)
        except (RuntimeError, ValueError):
            results = []
        self._render_recent(results)

    def _render_recent(self, conversations: list[dict[str, Any]]) -> None:
        self._recent_browser.setPlainText(_render_conversation_groups(conversations))

    def _render_archived(self, conversations: list[dict[str, Any]]) -> None:
        self._archived_browser.setPlainText(_render_conversation_groups(conversations))


def _render_conversation_groups(
    conversations: list[dict[str, Any]],
) -> str:
    """Render a list of conversations as a date-grouped plain-text block.

    Kept module-private and shared by both sections to avoid duplication.
    """
    if not conversations:
        return "(no conversations)"
    groups = group_by_date(conversations)
    lines: list[str] = []
    for date_key, convs in groups.items():
        lines.append(f"── {date_key} ──")
        for conv in convs:
            lines.append(format_conversation_preview(conv))
        lines.append("")
    return "\n".join(lines).rstrip()


def build_conversation_context_menu(
    parent: QWidget,
    conversation_id: str,
    is_archived: bool,
    pane: HistorySidebarPane,
) -> QMenu:
    """Build a Qt context menu wired to :meth:`HistorySidebarPane.trigger_action`.

    Exposed at module level for reuse by parent panels that want to
    surface the same action set from a different entry point (e.g. a
    keyboard shortcut palette).
    """
    menu = QMenu(parent)
    if is_archived:
        menu.addAction(
            "Restore",
            lambda: pane.trigger_action(conversation_id, "restore"),
        )
    else:
        menu.addAction(
            "Archive",
            lambda: pane.trigger_action(conversation_id, "archive"),
        )
    menu.addAction(
        "Load as context",
        lambda: pane.trigger_action(conversation_id, "load_as_context"),
    )
    menu.addSeparator()
    menu.addAction(
        "Export…",
        lambda: pane.trigger_action(conversation_id, "export"),
    )
    menu.addAction(
        "Delete",
        lambda: pane.trigger_action(conversation_id, "delete"),
    )
    return menu
