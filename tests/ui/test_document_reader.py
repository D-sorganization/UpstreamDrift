"""Tests for DocumentReaderWidget — issue #5725: fragment link handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QUrl

from src.shared.python.ui.qt.widgets.document_reader import DocumentReaderWidget

# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def reader(qapp) -> DocumentReaderWidget:
    widget = DocumentReaderWidget()
    return widget


# ── Issue #5725: fragment links must not be treated as file paths ─────────────


def test_same_doc_fragment_scrolls_to_anchor(reader) -> None:
    """A bare '#section' URL must call scrollToAnchor, not open an external browser."""
    reader.text_browser.scrollToAnchor = MagicMock()
    with patch("PyQt6.QtGui.QDesktopServices") as mock_svc:
        url = QUrl("#introduction")
        reader._handle_link_click(url)

    reader.text_browser.scrollToAnchor.assert_called_once_with("introduction")
    mock_svc.openUrl.assert_not_called()


def test_file_with_fragment_strips_fragment_before_existence_check(
    reader, tmp_path
) -> None:
    """A link like 'quickstart.md#step-1' must open the file (without the fragment)."""
    md_file = tmp_path / "quickstart.md"
    md_file.write_text("# Quickstart\n## Step 1\nContent here.\n", encoding="utf-8")

    reader.current_file = tmp_path / "index.md"

    with patch(
        "src.shared.python.ui.qt.widgets.document_reader.show_document"
    ) as mock_show:
        url = QUrl(f"file:///{md_file.as_posix()}#step-1")
        reader._handle_link_click(url)

    mock_show.assert_called_once()
    called_path = Path(mock_show.call_args[0][0])
    assert called_path == md_file


def test_relative_file_with_fragment_resolves_correctly(reader, tmp_path) -> None:
    """A relative link 'docs/arch.md#details' must resolve relative to current file."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    arch_file = docs_dir / "arch.md"
    arch_file.write_text("# Architecture\n## Details\n", encoding="utf-8")

    reader.current_file = tmp_path / "README.md"

    with patch(
        "src.shared.python.ui.qt.widgets.document_reader.show_document"
    ) as mock_show:
        url = QUrl("docs/arch.md#details")
        reader._handle_link_click(url)

    mock_show.assert_called_once()
    called_path = Path(mock_show.call_args[0][0])
    assert called_path == arch_file


def test_external_http_link_opens_browser(reader) -> None:
    """An http:// link must open in the external browser, not try file existence."""
    with patch("PyQt6.QtGui.QDesktopServices") as mock_svc:
        url = QUrl("https://example.com/page#section")
        reader._handle_link_click(url)

    mock_svc.openUrl.assert_called_once_with(url)


def test_nonexistent_file_falls_back_to_browser(reader, tmp_path) -> None:
    """A file:// link pointing to a nonexistent path falls back to QDesktopServices."""
    reader.current_file = tmp_path / "README.md"
    missing = tmp_path / "missing.md"

    with patch("PyQt6.QtGui.QDesktopServices") as mock_svc:
        url = QUrl(f"file:///{missing.as_posix()}")
        reader._handle_link_click(url)

    mock_svc.openUrl.assert_called_once()
