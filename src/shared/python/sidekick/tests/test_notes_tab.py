"""Tests for notes_tab.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PyQt6", reason="PyQt6 not installed")

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QDialog, QWidget
from sidekick.notes_tab import (
    NotesTab,
    _color_button_style,
    _make_note_card_widget,
    _NoteEditDialog,
    _render_markdown,
)


def test_render_markdown() -> None:
    """Test mistune markdown rendering fallback and plugin config."""
    # Test with mistune forced to True and False
    import sidekick.notes_tab

    if not hasattr(sidekick.notes_tab, "mistune"):
        # mistune wasn't imported successfully. Mock it.
        mock_mistune = MagicMock()
        mock_md = MagicMock(return_value="mocked_html")
        mock_mistune.create_markdown.return_value = mock_md
        with (
            patch("sidekick.notes_tab._MISTUNE_AVAILABLE", True),
            patch("sidekick.notes_tab.mistune", mock_mistune, create=True),
        ):
            res = _render_markdown("~~strike~~")
            assert res == "mocked_html"
            mock_mistune.create_markdown.assert_called_once_with(
                plugins=["strikethrough", "table"]
            )
            mock_md.assert_called_once_with("~~strike~~")
    else:
        # mistune is actually available, so we test it for real
        with patch("sidekick.notes_tab._MISTUNE_AVAILABLE", True):
            res = _render_markdown("~~strike~~")
            # should render html
            assert "<del>" in res or "<p>" in res or "strike" in res

    with patch("sidekick.notes_tab._MISTUNE_AVAILABLE", False):
        res = _render_markdown("~~strike~~")
        assert res == "<pre>~~strike~~</pre>"


def test_color_button_style() -> None:
    """Test button stylesheet generation."""
    style = _color_button_style("#ff0000")
    assert "background: #ff0000" in style


def test_make_note_card_widget(qapp: Any) -> None:
    """Test note card widget creation and signals connections."""
    card = MagicMock()
    card.note_id = "test_note_1"
    card.title = "My Title"
    card.color = "#ff00ff"
    card.markdown_body = "Body text"

    on_edit = MagicMock()
    on_delete = MagicMock()
    on_color_change = MagicMock()

    widget = _make_note_card_widget(card, on_edit, on_delete, on_color_change)
    assert isinstance(widget, QWidget)

    # Trigger clicks by calling the buttons' clicked signals
    edit_btn = widget.findChild(QWidget, "NoteCardEdit_test_note_1")
    assert edit_btn is not None
    edit_btn.click()  # type: ignore[attr-defined]
    on_edit.assert_called_once_with("test_note_1")

    delete_btn = widget.findChild(QWidget, "NoteCardDelete_test_note_1")
    assert delete_btn is not None
    delete_btn.click()  # type: ignore[attr-defined]
    on_delete.assert_called_once_with("test_note_1")

    color_btn = widget.findChild(QWidget, "NoteCardColor_test_note_1")
    assert color_btn is not None
    color_btn.click()  # type: ignore[attr-defined]
    on_color_change.assert_called_once_with("test_note_1")


def test_note_edit_dialog_new(qapp: Any) -> None:
    """Test dialog creation in 'new note' mode."""
    dialog = _NoteEditDialog()
    assert dialog.windowTitle() == "New Note"
    assert dialog.title_text == ""
    assert dialog.body_text == ""
    assert dialog.color == "#fff7cc"  # DEFAULT_NOTE_COLOR


def test_note_edit_dialog_edit(qapp: Any) -> None:
    """Test dialog creation in 'edit note' mode."""
    card = MagicMock()
    card.title = "Existing Title"
    card.markdown_body = "Existing Body"
    card.color = "#ff0000"

    dialog = _NoteEditDialog(card=card)
    assert dialog.windowTitle() == "Edit Note"
    assert dialog.title_text == "Existing Title"
    assert dialog.body_text == "Existing Body"
    assert dialog.color == "#ff0000"


def test_note_edit_dialog_pick_color(qapp: Any) -> None:
    """Test color picker inside the edit dialog."""
    dialog = _NoteEditDialog()
    picked_color = QColor("#00ff00")

    with patch("PyQt6.QtWidgets.QColorDialog.getColor", return_value=picked_color):
        dialog._pick_color()
        assert dialog.color == "#00ff00"


def test_notes_tab_init_validation() -> None:
    """Test notes tab constructor validates project_root."""
    with pytest.raises(TypeError, match="project_root must be provided"):
        NotesTab(None)  # type: ignore[arg-type]


def test_notes_tab_lifecycle(tmp_path: Path, qapp: Any) -> None:
    """Test notes tab CRUD operations, refresh UI and signals."""
    tab = NotesTab(tmp_path)

    # Empty state empty label is visible
    empty_label = tab.findChild(QWidget, "NotesTabEmptyLabel")
    assert empty_label is not None

    # Test creating a new note
    # Mock _NoteEditDialog exec to return Accepted
    with (
        patch(
            "sidekick.notes_tab._NoteEditDialog.exec",
            return_value=QDialog.DialogCode.Accepted,
        ),
        patch("sidekick.notes_tab._NoteEditDialog.title_text", "Note A"),
        patch("sidekick.notes_tab._NoteEditDialog.body_text", "Body A"),
        patch("sidekick.notes_tab._NoteEditDialog.color", "#ff0000"),
    ):
        notes_changed_signal = MagicMock()
        tab.notes_changed.connect(notes_changed_signal)

        tab._on_new_note()

        assert len(tab.note_ids()) == 1
        notes_changed_signal.assert_called_once()

    # Empty label should be removed from the layout
    layout_widgets = [
        tab._card_layout.itemAt(i).widget() for i in range(tab._card_layout.count())
    ]
    assert empty_label not in layout_widgets

    # Retrieve note id
    note_id = tab.note_ids()[0]

    # Test editing the note
    with (
        patch(
            "sidekick.notes_tab._NoteEditDialog.exec",
            return_value=QDialog.DialogCode.Accepted,
        ),
        patch("sidekick.notes_tab._NoteEditDialog.title_text", "Note A Updated"),
        patch("sidekick.notes_tab._NoteEditDialog.body_text", "Body A Updated"),
        patch("sidekick.notes_tab._NoteEditDialog.color", "#00ff00"),
    ):
        tab._on_edit_note(note_id)
        card = tab._store.load_note(note_id)
        assert card is not None
        assert card.title == "Note A Updated"
        assert card.markdown_body == "Body A Updated"
        assert card.color == "#00ff00"

    # Test changing color via color button callback
    picked_color = QColor("#0000ff")
    with patch("PyQt6.QtWidgets.QColorDialog.getColor", return_value=picked_color):
        tab._on_change_color(note_id)
        card = tab._store.load_note(note_id)
        assert card is not None
        assert card.color == "#0000ff"

    # Test deleting the note
    tab._on_delete_note(note_id)
    assert len(tab.note_ids()) == 0


def test_notes_tab_edge_cases(tmp_path: Path, qapp: Any) -> None:
    """Test non-existent notes operations."""
    tab = NotesTab(tmp_path)

    # Try editing non-existent note
    tab._on_edit_note("fake_note")  # Should log warning but not raise

    # Try deleting non-existent note
    tab._on_delete_note("fake_note")  # Should log warning but not raise

    # Try changing color of non-existent note
    tab._on_change_color("fake_note")  # Should return early
