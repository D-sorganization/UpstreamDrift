"""Tests for HoverCopyTextBrowser widget (issue #6683)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("PyQt6")
pytestmark = pytest.mark.gui

from src.shared.python.ui.hover_copy_browser import HoverCopyTextBrowser  # noqa: E402


@pytest.fixture(scope="module")
def qt_app() -> Any:
    """Minimal QApplication for Qt widget tests."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


class TestHoverCopyTextBrowser:
    def test_instantiation(self, qt_app: Any) -> None:
        """Verify the widget can be instantiated."""
        widget = HoverCopyTextBrowser()
        assert widget is not None
        assert widget.copy_btn is not None
        assert widget.copy_btn.toolTip() == "Copy to clipboard"

    def test_append_plain_text(self, qt_app: Any) -> None:
        """Verify append acts as a drop-in replacement for QTextBrowser."""
        widget = HoverCopyTextBrowser()
        widget.append("Hello World")
        assert "Hello World" in widget.toPlainText()

    def test_copy_all_text(self, qt_app: Any) -> None:
        """Verify copy_all_text copies content to clipboard."""
        from PyQt6.QtWidgets import QApplication

        widget = HoverCopyTextBrowser()
        # Test basic append and copy
        widget.append("Test Clipboard Content")
        widget.copy_all_text()

        clipboard = QApplication.clipboard()
        if clipboard:
            assert "Test Clipboard Content" in clipboard.text()
