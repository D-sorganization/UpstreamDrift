"""Memory management panel: ``user_memory.json`` viewer + editor.

Renders four structured sections (identity / preferences / projects /
knowledge) loaded via :class:`HistoryServiceAdapter`. Users can edit each
section as JSON and save, archive-digest the archived conversations into
the memory document, or reset the document entirely.

All persistence flows through the adapter — this widget never touches
disk directly (Law of Demeter).
"""

from __future__ import annotations

import json
from typing import Any

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.launchers.chat_history.chat_history_service import (
    MEMORY_SECTIONS,
    HistoryServiceAdapter,
)


class MemoryPanel(QWidget):
    """Editor for the persistent ``user_memory.json`` document."""

    def __init__(
        self,
        adapter: HistoryServiceAdapter,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ChatMemoryPanel")
        self._adapter = adapter
        self._editors: dict[str, QPlainTextEdit] = {}
        self._last_digest_status: str | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Persistent Memory"))

        # Four structured section editors.
        for section in MEMORY_SECTIONS:
            layout.addWidget(QLabel(section.capitalize()))
            editor = QPlainTextEdit()
            editor.setObjectName(f"MemorySection_{section}")
            editor.setPlaceholderText(f"{section} (JSON object)")
            self._editors[section] = editor
            layout.addWidget(editor, 1)

        # Buttons row.
        buttons = QHBoxLayout()

        save_btn = QPushButton("Save")
        save_btn.setToolTip("Persist memory to disk")
        save_btn.clicked.connect(self._on_save_clicked)
        buttons.addWidget(save_btn)

        digest_btn = QPushButton("Archive digest")
        digest_btn.setToolTip("Condense archived conversations into memory entries")
        digest_btn.clicked.connect(self._on_digest_clicked)
        self._digest_btn = digest_btn
        self._refresh_digest_button_state()
        buttons.addWidget(digest_btn)

        buttons.addStretch()

        reset_btn = QPushButton("Reset memory")
        reset_btn.setToolTip("Delete the entire memory document")
        reset_btn.clicked.connect(self._on_reset_clicked)
        buttons.addWidget(reset_btn)

        layout.addLayout(buttons)

    # ------------------------------------------------------------------
    # Public API (used by parent panel + tests)
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload the memory document from disk and repopulate editors."""
        memory = self._adapter.load_memory()
        for section, editor in self._editors.items():
            value = memory.get(section, {})
            editor.setPlainText(_format_section(value))

    def has_section(self, section: str) -> bool:
        return section in self._editors

    def section_text(self, section: str) -> str:
        return self._editors[section].toPlainText()

    def set_section_text(self, section: str, text: str) -> None:
        self._editors[section].setPlainText(text)

    def save(self) -> None:
        """Parse all section editors and persist via the adapter.

        Raises:
            ValueError: if any section's text is not a JSON object.
        """
        memory = self._collect_memory_from_editors()
        self._adapter.save_memory(memory)

    def reset(self) -> None:
        """Delete the memory document via the adapter and reload."""
        self._adapter.reset_memory()
        self.refresh()

    def archive_digest(self) -> None:
        """Condense archived conversations into the memory document."""
        if not self._adapter_can_condense():
            self._last_digest_status = "unavailable"
            return
        try:
            archived = self._adapter.list_archived()
        except RuntimeError:
            archived = []
        if not archived:
            self._last_digest_status = "noop"
            return
        ids = [conv["id"] for conv in archived if "id" in conv]
        result = self._adapter.condense_to_memory(ids)
        self._last_digest_status = result.get("status", "unknown")
        # Reload so any condenser-side merge is visible.
        self.refresh()

    def last_digest_status(self) -> str | None:
        """Return the status string from the last archive-digest call."""
        return self._last_digest_status

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_memory_from_editors(self) -> dict[str, Any]:
        memory: dict[str, Any] = {}
        for section, editor in self._editors.items():
            raw = editor.toPlainText().strip() or "{}"
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"section {section!r} has invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"section {section!r} must be a JSON object")
            memory[section] = parsed
        return memory

    def _on_save_clicked(self) -> None:
        try:
            self.save()
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot save memory", str(exc))

    def _on_reset_clicked(self) -> None:
        reply = QMessageBox.question(
            self,
            "Reset memory?",
            "This permanently deletes the user_memory.json document. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.reset()

    def _on_digest_clicked(self) -> None:
        self.archive_digest()
        status = self._last_digest_status or "unknown"
        if status == "unavailable":
            QMessageBox.information(
                self,
                "Memory digest",
                "Memory condensation is not available in this Sidekick build.",
            )
            return
        if status == "stub":
            QMessageBox.information(
                self,
                "Memory digest",
                "Memory condensation is not wired for this launcher session. "
                "Inject a chat service with condense_to_memory and try again.",
            )

    def _adapter_can_condense(self) -> bool:
        capability = getattr(self._adapter, "has_condensation_api", None)
        return bool(capability()) if callable(capability) else True

    def _refresh_digest_button_state(self) -> None:
        can_condense = self._adapter_can_condense()
        self._digest_btn.setEnabled(can_condense)
        if not can_condense:
            self._digest_btn.setToolTip(
                "Memory condensation is not available in this Sidekick build"
            )


def _format_section(value: Any) -> str:
    """Render a section value as pretty-printed JSON."""
    return json.dumps(value, indent=2, sort_keys=True)
