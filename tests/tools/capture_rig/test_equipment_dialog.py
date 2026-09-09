"""Bag navigation, capture assignment and library lineage work through Qt."""

from pathlib import Path

import pytest

from src.tools.capture_rig.capture_library import CaptureLibrary, read_notes
from src.tools.capture_rig.equipment import load_capture_club
from src.tools.capture_rig.equipment_dialog import CatalogPicker, EquipmentDialog
from src.tools.capture_rig.library_dialog import LibraryDialog
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_equipment import club
from tests.tools.capture_rig.test_library_dialog import _ready
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_save_assign_archive_reload_and_library_display(tmp_path: Path) -> None:
    _app()
    library = CaptureLibrary(tmp_path / "library")
    root = _bundle(tmp_path)
    library.register(root)
    dialog = EquipmentDialog(library.root, root)
    assert not dialog.assign_selected()
    assert "select a club" in dialog.status.text()
    item = club()
    assert dialog.save_club(item)
    dialog.items.setCurrentRow(0)
    assert dialog.assign_selected()
    assert read_notes(root).capture_id in dialog.status.text()
    assert load_capture_club(root).club == item
    dialog.archive_selected()
    assert dialog.items.count() == 0
    dialog.archived.setChecked(True)
    dialog.items.setCurrentRow(0)
    assert not dialog.assign_selected()
    assert "archived" in dialog.status.text()
    dialog.archive_selected()
    dialog.reload()
    assert dialog.items.count() == 1
    dialog.close()
    browser = LibraryDialog(library, open_capture=lambda _root: None)
    _ready(browser)
    browser.table.selectRow(0)
    assert item.label in browser.equipment_label.text()
    browser.close()


def test_stale_dialog_cannot_replace_another_bag_edit(tmp_path: Path) -> None:
    _app()
    first = EquipmentDialog(tmp_path)
    second = EquipmentDialog(tmp_path)
    assert first.save_club(club())
    assert not second.save_club(club())
    assert "Reload Bag" in second.status.text()
    second.reload()
    assert second.items.count() == 1
    first.close()
    second.close()


def test_editable_copy_rebinds_club_to_new_capture(tmp_path: Path) -> None:
    _app()
    library = CaptureLibrary(tmp_path / "library")
    root = _bundle(tmp_path)
    library.register(root)
    dialog = EquipmentDialog(library.root, root)
    dialog.save_club(club())
    dialog.items.setCurrentRow(0)
    assert dialog.assign_selected()
    copied = library.editable_copy(root)
    old, new = load_capture_club(root), load_capture_club(copied)
    assert old.club == new.club
    assert old.capture_id != new.capture_id
    assert new.capture_id == read_notes(copied).capture_id
    dialog.close()


def test_catalog_search_exposes_partial_builds_and_sources() -> None:
    _app()
    picker = CatalogPicker()
    picker.search.setText("Titleist")
    assert picker.items.count() == 2
    picker.items.setCurrentRow(0)
    assert "Unknown" in picker.details.toPlainText()
    assert "https://www.titleist.com" in picker.details.toPlainText()
    assert picker.selected_club().base.identity.manufacturer == "Titleist"
    picker.search.setText("No such model")
    assert picker.items.count() == 0
    with pytest.raises(ValueError, match="Select"):
        picker.selected_club()
    picker.close()
