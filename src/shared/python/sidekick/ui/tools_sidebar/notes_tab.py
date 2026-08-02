"""Project notes widget for Sidekick runtime tabs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .qt_compat import QtCore, QtWidgets

SIDEKICK_NOTES_OBJECT_NAME = "SidekickNotesTab"


class SidekickNotesWidget(QtWidgets.QWidget):
    """Project note-card editor with explicit save and debounced persistence."""

    def __init__(
        self,
        *,
        project_root: Path,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if project_root is None:
            raise ValueError("project_root must be provided")
        super().__init__(parent)
        self.setObjectName(SIDEKICK_NOTES_OBJECT_NAME)
        self._store = _note_card_store(project_root)
        self._active_card_id: str | None = None
        self._autosave = QtCore.QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(500)
        self._autosave.timeout.connect(self.save_notes)
        self._build_ui()
        self._load_first_card()
        self._apply_board_style()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._status = QtWidgets.QLabel("Ready", self)
        self._status.setObjectName("SidekickNotesStatus")
        self._status.setToolTip("Reports the latest notes persistence status.")
        layout.addWidget(self._status)

        self._card_frame = QtWidgets.QFrame(self)
        self._card_frame.setObjectName("SidekickNotesCard")
        card_layout = QtWidgets.QVBoxLayout(self._card_frame)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(8)

        self._color = QtWidgets.QLineEdit(self._card_frame)
        self._color.setObjectName("SidekickNotesCardColor")
        self._color.setPlaceholderText("#fff7cc")
        self._color.setToolTip("Sets the active note card color as a #RRGGBB value.")
        card_layout.addWidget(self._color)

        self._editor = QtWidgets.QPlainTextEdit(self)
        self._editor.setObjectName("SidekickNotesEditor")
        self._editor.setPlaceholderText("Project notes")
        self._editor.setToolTip("Edit the active project-scoped markdown note card.")
        self._editor.textChanged.connect(self._schedule_autosave)
        card_layout.addWidget(self._editor, stretch=1)
        layout.addWidget(self._card_frame, stretch=1)

        self._board_color = QtWidgets.QLineEdit(self)
        self._board_color.setObjectName("SidekickNotesBoardColor")
        self._board_color.setPlaceholderText("#f7f7f7")
        self._board_color.setToolTip("Sets the notes screen background color.")
        layout.addWidget(self._board_color)

        row = QtWidgets.QHBoxLayout()
        self._save = QtWidgets.QPushButton("Save", self)
        self._save.setObjectName("SidekickNotesSave")
        self._save.setToolTip("Persist the current notes text immediately.")
        self._save.clicked.connect(self.save_notes)
        row.addWidget(self._save)

        clear = QtWidgets.QPushButton("Clear", self)
        clear.setObjectName("SidekickNotesClear")
        clear.setToolTip(
            "Clear the current note text while keeping the notes file available."
        )
        clear.clicked.connect(self.clear_notes)
        row.addWidget(clear)

        restore = QtWidgets.QPushButton("Restore", self)
        restore.setObjectName("SidekickNotesRestore")
        restore.setToolTip(
            "Restore the latest recycled notes snapshot when one exists."
        )
        restore.clicked.connect(self.restore_latest)
        row.addWidget(restore)

        apply_colors = QtWidgets.QPushButton("Apply Colors", self)
        apply_colors.setObjectName("SidekickNotesApplyColors")
        apply_colors.setToolTip("Validate and persist note and screen colors.")
        apply_colors.clicked.connect(self.apply_colors)
        row.addWidget(apply_colors)
        layout.addLayout(row)

    def save_notes(self) -> None:
        """Persist the current notes text to the active markdown card."""
        color = self._color.text().strip() or "#fff7cc"
        if self._active_card_id is None:
            card = self._store.create_note(
                "Project Notes",
                self._editor.toPlainText(),
                color=color,
            )
            self._active_card_id = card.note_id
        else:
            self._store.update_note(
                self._active_card_id,
                title="Project Notes",
                markdown_body=self._editor.toPlainText(),
                color=color,
            )
        self.apply_colors(save_note=False)
        self._status.setText("Saved")

    def clear_notes(self) -> None:
        """Clear notes while preserving the active markdown card."""
        self._editor.setPlainText("")
        self.save_notes()
        self._status.setText("Cleared")

    def restore_latest(self) -> None:
        """Restore the latest recycled note file when available."""
        item_id = self._store.latest_recycled_id()
        restored = None if item_id is None else self._store.restore_note(item_id)
        if restored is None:
            self._status.setText("Nothing to restore")
            return
        self._active_card_id = restored.note_id
        self._editor.setPlainText(restored.markdown_body)
        self._color.setText(restored.color)
        self._apply_card_style(restored.color)
        self._status.setText("Restored")

    def apply_colors(self, *, save_note: bool = True) -> None:
        """Validate and persist note-card and board colors."""
        from shared.python.notes.models import NotesBoardSettings, normalize_color

        note_color = normalize_color(self._color.text().strip() or "#fff7cc")
        board = NotesBoardSettings(
            background_color=self._board_color.text().strip() or "#f7f7f7"
        )
        self._color.setText(note_color)
        self._board_color.setText(board.background_color)
        self._store.save_settings(board)
        self._apply_card_style(note_color)
        self._apply_board_style()
        if save_note:
            self.save_notes()

    def _load_first_card(self) -> None:
        card = self._store.migrate_legacy_text_note()
        if card is None:
            notes = self._store.list_notes()
            card = notes[0] if notes else None
        if card is not None:
            self._active_card_id = card.note_id
            self._editor.setPlainText(card.markdown_body)
            self._color.setText(card.color)
            self._apply_card_style(card.color)
        else:
            self._color.setText("#fff7cc")
        self._board_color.setText(self._store.load_settings().background_color)

    def _apply_board_style(self) -> None:
        color = self._store.load_settings().background_color
        self.setStyleSheet(f"#{SIDEKICK_NOTES_OBJECT_NAME} {{ background: {color}; }}")

    def _apply_card_style(self, color: str) -> None:
        self._card_frame.setStyleSheet(
            "#SidekickNotesCard { "
            f"background: {color}; border: 1px solid #d0d0d0; border-radius: 6px;"
            " }"
        )

    def _schedule_autosave(self) -> None:
        self._autosave.start()


def _note_card_store(project_root: Path) -> Any:
    from shared.python.notes.card_store import NoteCardStore

    return NoteCardStore(project_dir=project_root)
