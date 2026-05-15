"""Tests for the AutoCompleteLineEdit widget."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication

from src.shared.python.ui.auto_complete import AutoCompleteLineEdit


def get_app():
    return QApplication.instance() or QApplication([])

def test_auto_complete_line_edit_initialization() -> None:
    """Test that AutoCompleteLineEdit initializes correctly with words."""
    app = get_app()
    words = ["gravity", "velocity", "acceleration"]
    widget = AutoCompleteLineEdit(words=words)
    assert widget.completer_words == words
    assert widget.completer() is not None


def test_auto_complete_set_completion_words() -> None:
    """Test dynamically updating the completion words."""
    app = get_app()
    widget = AutoCompleteLineEdit()
    assert widget.completer_words == []
    
    widget.set_completion_words(["mass", "force"])
    assert widget.completer_words == ["mass", "force"]


def test_auto_complete_add_completion_words() -> None:
    """Test adding words to the completion dictionary."""
    app = get_app()
    widget = AutoCompleteLineEdit(words=["gravity"])
    widget.add_completion_words(["mass"])
    
    assert "gravity" in widget.completer_words
    assert "mass" in widget.completer_words
    assert len(widget.completer_words) == 2


def test_auto_complete_tab_key() -> None:
    """Test that the Tab key accepts the current completion."""
    app = get_app()
    widget = AutoCompleteLineEdit(words=["acceleration"])
    widget.setText("acc")
    
    # Simulate completer state
    widget.auto_completer.setCompletionPrefix("acc")
    assert widget.auto_completer.currentCompletion() == "acceleration"
    
    # Send Tab key event
    event = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_Tab,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.keyPressEvent(event)
    
    # Check that text was updated
    assert widget.text() == "acceleration"
