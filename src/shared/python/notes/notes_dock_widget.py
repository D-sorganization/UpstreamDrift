"""Reusable PyQt6 notes widget with recycle-bin semantics."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .storage import NotesStorage


class NotesDockWidget(QDockWidget):
    """Dockable notes workspace that can be embedded or popped out."""

    def __init__(
        self,
        project_dir: Path | str,
        title: str = "Notes",
        parent: QWidget | None = None,
    ) -> None:
        if project_dir is None:
            raise ValueError("project_dir must be provided")
        super().__init__(title, parent)
        self.storage = NotesStorage(project_dir=project_dir)
        self._status_label: QLabel | None = None
        self._text_edit: QPlainTextEdit | None = None
        self._build_ui()
        self.reload_from_disk()

    def _build_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)

        self._status_label = QLabel("Ready")
        layout.addWidget(self._status_label)

        self._text_edit = QPlainTextEdit()
        self._text_edit.setPlaceholderText("Paste or write notes here...")
        layout.addWidget(self._text_edit, stretch=1)

        row = QHBoxLayout()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_notes)
        row.addWidget(save_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_editor)
        row.addWidget(clear_btn)

        delete_btn = QPushButton("Move to Bin")
        delete_btn.clicked.connect(self.delete_to_recycle_bin)
        row.addWidget(delete_btn)

        restore_btn = QPushButton("Restore")
        restore_btn.clicked.connect(self.restore_latest_deleted)
        row.addWidget(restore_btn)

        pop_btn = QPushButton("Pop Out")
        pop_btn.clicked.connect(self.pop_out)
        row.addWidget(pop_btn)

        layout.addLayout(row)
        self.setWidget(container)

    def save_notes(self) -> None:
        """Persist the current editor contents to the notes file on disk."""
        text = self._require_editor().toPlainText()
        self.storage.save_text(text)
        self._set_status("Saved")

    def reload_from_disk(self) -> None:
        """Reload the editor contents from the notes file on disk."""
        self._require_editor().setPlainText(self.storage.load_text())

    def clear_editor(self) -> None:
        """Clear the editor and remove the notes file from storage."""
        self._require_editor().setPlainText("")
        self.storage.clear()
        self._set_status("Cleared")

    def delete_to_recycle_bin(self) -> bool:
        """Save, then move the notes file to the recycle bin. Return True on success."""
        self.save_notes()
        try:
            self.storage.move_to_recycle(reason="user_delete")
        except FileNotFoundError:
            self._set_status("No notes file to delete")
            return False

        self._require_editor().setPlainText("")
        self._set_status("Moved to recycle bin")
        return True

    def restore_latest_deleted(self) -> bool:
        """Restore the most recently recycled notes file. Return True on success."""
        item_id = self.storage.latest_recycled_id()
        if item_id is None:
            self._set_status("Recycle bin empty")
            return False

        restored = self.storage.restore(item_id)
        if restored is None:
            self._set_status("Restore failed")
            return False

        self.reload_from_disk()
        self._set_status("Restored")
        return True

    def pop_out(self) -> None:
        """Detach the dock widget into a floating window."""
        self.setFloating(True)
        self.show()

    def embed_in(self, main_window: QWidget, area: Qt.DockWidgetArea) -> None:
        """Re-dock the widget into *main_window* at the given *area*."""
        if not hasattr(main_window, "addDockWidget"):
            raise ValueError("main_window must support addDockWidget")
        main_window.addDockWidget(area, self)
        self.setFloating(False)
        self.show()

    def _set_status(self, text: str) -> None:
        self._require_status_label().setText(text)

    def _require_editor(self) -> QPlainTextEdit:
        if self._text_edit is None:
            raise RuntimeError("notes editor not initialized")
        return self._text_edit

    def _require_status_label(self) -> QLabel:
        if self._status_label is None:
            raise RuntimeError("status label not initialized")
        return self._status_label
