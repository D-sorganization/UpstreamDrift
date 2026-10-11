"""CMB-7e (#11658): desktop Character Builder appearance picker and export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _model_cls():
    from src.tools.character_builder.core import CharacterBuilderModel

    return CharacterBuilderModel


def _document_from_dict():
    from src.shared.python.model_appearance.schema import document_from_dict

    return document_from_dict


def test_default_document_validates_and_binds_spec_hash() -> None:
    model = _model_cls()()
    doc = model.appearance_document()
    _document_from_dict()(doc)  # round-trips without raising
    assert doc["spec_sha256"] == model.compile().sha256


def test_set_appearance_rejects_unknown_name_without_changing_state() -> None:
    model = _model_cls()()
    before = model.appearance_document()
    with pytest.raises(ValueError, match="unknown"):
        model.set_appearance("wingspan", "x")
    assert model.appearance_document() == before


def test_set_appearance_rejects_unknown_value_without_changing_state() -> None:
    model = _model_cls()()
    before = model.appearance_document()
    with pytest.raises(ValueError):
        model.set_appearance("skin_tone", "skin_green")
    assert model.appearance_document() == before


def test_set_appearance_updates_document() -> None:
    model = _model_cls()()
    model.set_appearance("skin_tone", "skin_dark")
    assert model.appearance_document()["skin_tone"] == "skin_dark"


def test_editing_body_parameter_changes_bound_hash() -> None:
    model = _model_cls()()
    before_hash = model.appearance_document()["spec_sha256"]
    model.set_parameter("mass_kg", 90.0)
    after_hash = model.appearance_document()["spec_sha256"]
    assert before_hash != after_hash


def test_appearance_never_changes_compile_hash() -> None:
    model = _model_cls()()
    before = model.compile().sha256
    model.set_appearance("skin_tone", "skin_dark")
    model.set_appearance("headwear", "cap")
    assert model.compile().sha256 == before


def test_export_appearance_sits_beside_spec_export(tmp_path: Path) -> None:
    model = _model_cls()("junior")
    sha = model.compile().sha256
    path = model.export_appearance(tmp_path)
    spec_path = model.export("spec", tmp_path)
    assert path.name == f"{spec_path.stem}.appearance.json"
    assert sha[:8] in path.name
    data = json.loads(path.read_text(encoding="utf-8"))
    doc = _document_from_dict()(data)
    assert doc.spec_sha256 == sha


def test_export_appearance_filename_for_custom_edit(tmp_path: Path) -> None:
    model = _model_cls()("junior")
    model.set_parameter("mass_kg", 48.0)
    sha = model.compile().sha256
    path = model.export_appearance(tmp_path)
    assert path.name == f"custom_{sha[:8]}.appearance.json"


def test_export_appearance_does_not_change_compile_hash(tmp_path: Path) -> None:
    model = _model_cls()()
    before = model.compile().sha256
    model.export_appearance(tmp_path)
    assert model.compile().sha256 == before


def test_export_dispatches_appearance_format(tmp_path: Path) -> None:
    model = _model_cls()()
    direct = model.export_appearance(tmp_path / "a")
    via_export = model.export("appearance", tmp_path / "b")
    assert direct.name == via_export.name
    assert direct.read_text(encoding="utf-8") == via_export.read_text(encoding="utf-8")


def test_appearance_library_delegator_matches_service() -> None:
    from src.api.services.character_appearance_service import (
        appearance_library as service_library,
    )

    assert _model_cls()().appearance_library() == service_library()


class TestAppearanceWidget:
    @pytest.fixture
    def widget(self):
        QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from src.tools.character_builder.gui import CharacterBuilderWidget

        w = CharacterBuilderWidget()
        yield w
        w.cleanup()
        del app

    def test_combos_populated(self, widget) -> None:
        assert widget.appearance_combos["skin_tone"].count() == 4
        assert widget.appearance_combos["clothing"].count() >= 2
        assert widget.appearance_combos["club_finish"].count() == 4
        assert widget.appearance_combos["headwear"].count() == 3
        assert widget.appearance_combos["ground_material"].count() == 2

    def test_combo_initial_selection_matches_defaults(self, widget) -> None:
        defaults = widget.model.appearance_document()
        assert (
            widget.appearance_combos["skin_tone"].currentText() == defaults["skin_tone"]
        )
        assert (
            widget.appearance_combos["clothing"].currentText() == defaults["clothing"]
        )

    def test_picking_skin_tone_updates_model(self, widget) -> None:
        combo = widget.appearance_combos["skin_tone"]
        other = next(
            combo.itemText(i)
            for i in range(combo.count())
            if combo.itemText(i) != combo.currentText()
        )
        combo.setCurrentText(other)
        assert widget.model.appearance_document()["skin_tone"] == other

    def test_export_appearance_button_writes_file(self, widget, tmp_path: Path) -> None:
        written = widget.export("appearance", str(tmp_path))
        assert written is not None and Path(written).is_file()
        assert "Wrote" in widget.status.text()
