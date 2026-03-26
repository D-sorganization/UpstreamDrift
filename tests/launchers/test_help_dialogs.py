"""Tests for help_dialogs.py."""

from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

# Ensure PyQt classes are available
pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt  # noqa: E402

from src.launchers.help_dialogs import (  # noqa: E402
    ContextHelpDock,
    HelpDialog,
    LayoutManagerDialog,
)


@pytest.fixture
def test_model():
    model = MagicMock()
    model.name = "Test Model"
    model.description = "A test model"
    model.id = "test_id"
    return model


def test_help_dialog_file_exists(qapp):
    """Test HelpDialog when help.md exists."""
    with (
        patch("src.launchers.help_dialogs.Path.exists", return_value=True),
        patch("src.launchers.help_dialogs.Path.read_text", return_value="# Help"),
    ):
        dialog = HelpDialog()
        assert dialog.windowTitle() == "Golf Suite - Help"
        assert "Help" in dialog.text_area.toPlainText()


def test_help_dialog_file_missing(qapp):
    """Test HelpDialog when help.md is missing."""
    with patch("src.launchers.help_dialogs.Path.exists", return_value=False):
        dialog = HelpDialog()
        assert "not found" in dialog.text_area.toPlainText()


def test_layout_manager_dialog(qapp, test_model):
    """Test LayoutManagerDialog initialization and selection."""
    models = {"test_id": test_model}
    active = ["test_id"]

    dialog = LayoutManagerDialog(models, active, None)
    assert dialog.windowTitle() == "Customize Launcher Tiles"
    assert dialog.list_widget.count() == 1

    item = dialog.list_widget.item(0)
    assert item.checkState() == Qt.CheckState.Checked

    assert dialog.selected_ids() == ["test_id"]


def test_layout_manager_dialog_unchecked(qapp, test_model):
    """Test LayoutManagerDialog with unchecked items."""
    models = {"test_id": test_model}
    active = []

    dialog = LayoutManagerDialog(models, active, None)
    item = dialog.list_widget.item(0)
    assert item.checkState() == Qt.CheckState.Unchecked
    assert dialog.selected_ids() == []


def test_layout_manager_dialog_no_model_id(qapp, test_model):
    """Test LayoutManagerDialog with a checked item having no role data."""
    models = {"test_id": test_model}
    active = ["test_id"]

    dialog = LayoutManagerDialog(models, active, None)
    item = dialog.list_widget.item(0)
    item.setData(Qt.ItemDataRole.UserRole, None)
    assert dialog.selected_ids() == []


def test_context_help_dock_init(qapp):
    """Test ContextHelpDock initialization."""
    dock = ContextHelpDock()
    assert dock.windowTitle() == "Quick Help"
    assert "Context Aware Help" in dock.text_area.toPlainText()


def test_update_context_no_id(qapp):
    """Test update_context with None."""
    dock = ContextHelpDock()
    dock.update_context(None)
    assert "Context Aware Help" in dock.text_area.toPlainText()


def test_update_context_valid_file(qapp):
    """Test update_context when doc file exists."""
    dock = ContextHelpDock()

    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_path.read_text.return_value = "Valid doc content"

    with patch.object(dock, "_get_doc_file", return_value=mock_path):
        dock.update_context("some_id")
        assert "Valid doc content" in dock.text_area.toPlainText()


def test_update_context_read_error(qapp):
    """Test update_context when doc file exists but cannot be read."""
    dock = ContextHelpDock()

    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_path.read_text.side_effect = OSError("Read failed")

    with patch.object(dock, "_get_doc_file", return_value=mock_path):
        dock.update_context("some_id")
        assert "Failed to load documentation: Read failed" in dock.text_area.toPlainText()


def test_update_context_no_file(qapp):
    """Test update_context when doc file does not exist."""
    dock = ContextHelpDock()

    mock_path = MagicMock()
    mock_path.exists.return_value = False

    with patch.object(dock, "_get_doc_file", return_value=mock_path):
        dock.update_context("missing_id")
        assert "missing_id" in dock.text_area.toPlainText()
        assert "No specific documentation available" in dock.text_area.toPlainText()


def test_get_doc_file(qapp):
    """Test _get_doc_file mapping."""
    dock = ContextHelpDock()

    assert "mujoco.md" in str(dock._get_doc_file("mujoco_model"))
    assert "drake.md" in str(dock._get_doc_file("drake_model"))
    assert "pinocchio.md" in str(dock._get_doc_file("pinocchio_model"))
    assert "matlab.md" in str(dock._get_doc_file("matlab_model"))
    assert "urdf_generator" in str(dock._get_doc_file("urdf_model"))
    assert dock._get_doc_file("unknown_model") is None
