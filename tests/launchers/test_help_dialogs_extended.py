"""Extended coverage for ``src.launchers.help_dialogs``.

The module ships HelpDialog, LayoutManagerDialog, and ContextHelpDock
widgets.  These tests exercise widget construction, the markdown / plain
fallback when the help asset is missing, the LayoutManagerDialog model
selection round-trip, and the ContextHelpDock document-resolution
branches.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from PyQt6.QtCore import Qt

from src.launchers import help_dialogs


class _FakeModel:
    def __init__(self, _id: str, name: str, description: str) -> None:
        self.id = _id
        self.name = name
        self.description = description


def test_help_dialog_uses_help_md_when_present(qapp, tmp_path, monkeypatch) -> None:
    fake_help = tmp_path / "help.md"
    fake_help.write_text("# Help\nBody", encoding="utf-8")
    monkeypatch.setattr(help_dialogs, "ASSETS_DIR", tmp_path)

    dlg = help_dialogs.HelpDialog()
    assert dlg.windowTitle() == "Golf Suite - Help"
    # Markdown was loaded; checking that the QTextEdit has non-empty text.
    assert dlg.text_area.toPlainText()
    dlg.deleteLater()


def test_help_dialog_falls_back_when_help_md_missing(
    qapp, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(help_dialogs, "ASSETS_DIR", tmp_path)
    dlg = help_dialogs.HelpDialog()
    assert "Help file not found" in dlg.text_area.toPlainText()
    dlg.deleteLater()


def test_layout_manager_dialog_round_trip(qapp) -> None:
    available = {
        "a": _FakeModel("a", "Alpha", "Alpha desc"),
        "b": _FakeModel("b", "Beta", "Beta desc"),
        "c": _FakeModel("c", "Gamma", "Gamma desc"),
    }
    active = ["a", "c"]
    dlg = help_dialogs.LayoutManagerDialog(available, active, parent=None)

    assert dlg.windowTitle() == "Customize Launcher Tiles"

    selections = dlg.selected_ids()
    # The list widget initially mirrors ``active``.
    assert set(selections) == {"a", "c"}

    # Toggle the second item (Beta) on and Alpha off.
    for i in range(dlg.list_widget.count()):
        item = dlg.list_widget.item(i)
        text = item.text()
        if text.startswith("Alpha"):
            item.setCheckState(Qt.CheckState.Unchecked)
        elif text.startswith("Beta"):
            item.setCheckState(Qt.CheckState.Checked)

    selections = dlg.selected_ids()
    assert "a" not in selections
    assert "b" in selections
    assert "c" in selections
    dlg.deleteLater()


def test_context_help_dock_default_content(qapp) -> None:
    dock = help_dialogs.ContextHelpDock()
    # Default content is set when ``model_id`` is None.
    assert "Context Aware Help" in dock.text_area.toPlainText()
    dock.deleteLater()


def test_context_help_dock_update_with_unknown_model_id(qapp) -> None:
    dock = help_dialogs.ContextHelpDock()
    dock.update_context("totally_unknown_engine")
    assert "totally_unknown_engine" in dock.text_area.toPlainText()
    dock.deleteLater()


@pytest.mark.parametrize(
    "model_id,expected_substring",
    [
        ("mujoco_humanoid", "mujoco"),
        ("drake_arm", "drake"),
        ("pinocchio_demo", "pinocchio"),
        ("matlab_simscape", "matlab"),
        ("urdf_generator", "urdf"),
    ],
)
def test_context_help_dock_get_doc_file_branches(
    qapp, model_id, expected_substring
) -> None:
    dock = help_dialogs.ContextHelpDock()
    path = dock._get_doc_file(model_id)
    assert path is not None
    assert expected_substring in str(path).lower()
    dock.deleteLater()


def test_context_help_dock_get_doc_file_returns_none_for_unknown(qapp) -> None:
    dock = help_dialogs.ContextHelpDock()
    assert dock._get_doc_file("nonsense_engine") is None
    dock.deleteLater()


def test_context_help_dock_update_context_reads_existing_doc(qapp, tmp_path) -> None:
    dock = help_dialogs.ContextHelpDock()
    fake_doc = tmp_path / "mujoco.md"
    fake_doc.write_text("# MuJoCo docs", encoding="utf-8")
    with patch.object(dock, "_get_doc_file", return_value=fake_doc):
        dock.update_context("mujoco_demo")
    assert "MuJoCo docs" in dock.text_area.toPlainText()
    dock.deleteLater()


def test_context_help_dock_update_context_handles_read_failure(qapp, tmp_path) -> None:
    dock = help_dialogs.ContextHelpDock()
    fake_doc = tmp_path / "mujoco.md"
    fake_doc.write_text("ok", encoding="utf-8")
    with (
        patch.object(dock, "_get_doc_file", return_value=fake_doc),
        patch.object(
            type(fake_doc),
            "read_text",
            side_effect=OSError("disk gone"),
        ),
    ):
        dock.update_context("mujoco")
    assert "Failed to load" in dock.text_area.toPlainText()
    dock.deleteLater()
