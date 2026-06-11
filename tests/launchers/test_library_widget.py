"""Tests for launcher-hosted Library layout and escaping."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt

from src.launchers.library_widget import LibraryManager, LibraryWidget

pytestmark = pytest.mark.unit


def _insert_document(db_path: Path, **overrides: str) -> None:
    values = {
        "file_name": "doc.pdf",
        "file_path": str(db_path.parent / "doc.pdf"),
        "title": "Golf <script>alert(1)</script>",
        "author": "Ada <Lovelace>",
        "year": "2026",
        "topic": "Topology",
    }
    values.update(overrides)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO documents (file_name, file_path, title, author, year, topic)
            VALUES (:file_name, :file_path, :title, :author, :year, :topic)
            """,
            values,
        )


def test_library_uses_injected_manager_and_launcher_layout(
    qapp, tmp_path: Path
) -> None:
    manager = LibraryManager(tmp_path / "library_index.db")
    _insert_document(manager.db_path)

    widget = LibraryWidget(manager=manager)

    assert widget.objectName() == "LibraryWidget"
    assert widget.findChild(type(widget.table), "LibraryTable") is widget.table
    assert widget.findChild(type(widget.preview_browser), "LibraryPreview") is not None
    assert "QWidget#LibraryWidget" in widget.styleSheet()
    assert widget.table.rowCount() == 1


def test_library_preview_escapes_document_metadata(qapp, tmp_path: Path) -> None:
    manager = LibraryManager(tmp_path / "library_index.db")
    _insert_document(manager.db_path)
    widget = LibraryWidget(manager=manager)

    widget.table.selectRow(0)
    item = widget.table.item(0, 0)
    widget.table.setCurrentItem(item)
    widget._on_document_selected()

    html = widget.preview_browser.toHtml()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert not widget.chat_input.isEnabled()
    assert "backend not configured" in widget.chat_input.placeholderText().lower()


def test_document_chat_does_not_fabricate_backend_response(
    qapp, tmp_path: Path
) -> None:
    manager = LibraryManager(tmp_path / "library_index.db")
    doc_path = tmp_path / "doc.pdf"
    doc_path.write_bytes(b"%PDF-1.4\n% unreadable test fixture\n")
    _insert_document(
        manager.db_path,
        file_path=str(doc_path),
        title="No Fake Chat",
    )
    widget = LibraryWidget(manager=manager)
    widget.table.selectRow(0)
    widget.table.setCurrentItem(widget.table.item(0, 0))
    widget._on_document_selected()

    widget.chat_input.setText("What does this say?")
    widget._on_chat_return_pressed()

    html = widget.preview_browser.toHtml()
    assert "processed the document context" not in html
    assert "Dispatching to Notebook LM backend" not in html
    assert "Notebook LM:" not in html
    assert "Document chat backend is not configured" in html
    assert not widget.chat_input.isEnabled()


def test_library_filter_searches_document_fields(qapp, tmp_path: Path) -> None:
    manager = LibraryManager(tmp_path / "library_index.db")
    _insert_document(
        manager.db_path,
        file_name="course.pdf",
        file_path=str(tmp_path / "course.pdf"),
        title="Course Topology",
        author="Researcher",
    )
    _insert_document(
        manager.db_path,
        file_name="swing.pdf",
        file_path=str(tmp_path / "swing.pdf"),
        title="Swing Study",
        author="Coach",
        topic="Biomechanics",
    )
    widget = LibraryWidget(manager=manager)

    widget._filter_documents("topology")

    assert widget.table.rowCount() == 1
    assert (
        widget.table.item(0, 0).data(Qt.ItemDataRole.DisplayRole) == "Course Topology"
    )
