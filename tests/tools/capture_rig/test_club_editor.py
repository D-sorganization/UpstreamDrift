"""Real Qt editing preserves partial evidence and canonical quantities."""

import pytest

from src.shared.python.club_data.catalog_sources import load_public_catalog
from src.shared.python.club_data.player_clubs import PlayerClub
from src.tools.capture_rig.club_editor import ClubEditorDialog
from tests.tools.capture_rig.test_equipment import club, measured_length
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_custom_unknown_and_notes_round_trip() -> None:
    _app()
    editor = ClubEditorDialog()
    editor.label.setText("Lesson wedge")
    editor.number.setText("56")
    editor.club_type.setCurrentText("wedge")
    editor.notes.setPlainText("Borrowed; length not measured")
    result = editor.record()
    assert result.base.identity.number == "56"
    assert result.base.identity.club_type == "wedge"
    assert result.notes == "Borrowed; length not measured"
    assert result.effective_record().physical_value("length", "assembled") is None
    editor.close()


def test_display_unit_switches_do_not_change_saved_measurement() -> None:
    _app()
    original = club().model_copy(update={"overrides": (measured_length(37.123456),)})
    editor = ClubEditorDialog(original)
    for _ in range(10):
        editor.length.input.set_unit("cm")
        editor.length.input.set_unit("in")
    assert editor.record() == original
    editor.close()


def test_catalog_base_is_preserved_and_unknown_override_is_explicit() -> None:
    _app()
    original = PlayerClub(label="My Titleist", base=load_public_catalog()[0])
    editor = ClubEditorDialog(original)
    assert editor.record() == original
    editor.length.mode.setCurrentText("Unknown")
    result = editor.record()
    assert result.base == original.base
    assert result.overrides[0].status == "unknown"
    assert result.effective_record().physical_value("length", "assembled") is None
    editor.close()


def test_measured_value_requires_positive_quantity() -> None:
    _app()
    editor = ClubEditorDialog(club())
    editor.length.mode.setCurrentText("Measured")
    with pytest.raises(ValueError, match="positive"):
        editor.record()
    editor.length.input.set_value(38.5, "in")
    result = editor.record()
    assert result.overrides[0].status == "measured"
    assert result.overrides[0].si_value() == pytest.approx(0.9779)
    editor.close()


def test_restoring_original_shows_catalog_value_instead_of_player_override() -> None:
    _app()
    item = PlayerClub(
        label="Shortened",
        base=load_public_catalog()[0],
        overrides=(measured_length(35),),
    )
    editor = ClubEditorDialog(item)
    editor.length.mode.setCurrentText("Use Original")
    assert "37 in" in editor.length.summary.text()
    assert editor.record().overrides == ()
    editor.close()
