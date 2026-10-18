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
        from src.shared.python.model_appearance.library import SKIN_TONES

        assert widget.appearance_combos["skin_tone"].count() == len(SKIN_TONES) == 8
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


# --- HRG-B2 (#13167): custom colours become document materials -------------


def test_custom_colours_write_document_materials() -> None:
    from src.shared.python.model_appearance import library
    from src.shared.python.model_appearance.appearance_selection import hex_to_rgb

    model = _model_cls()()
    model.set_appearance("hair_color", "#336699")
    model.set_appearance("top_color", "#aa0000")
    model.set_appearance("skin_color", "#c68642")
    doc = model.appearance_document()
    parsed = _document_from_dict()(doc)
    mats = library.library_materials(parsed)
    assert mats[parsed.head.headwear_material].rgba[:3] == pytest.approx(
        hex_to_rgb("#336699")
    )
    assert mats[parsed.skin_tone].rgba[:3] == pytest.approx(hex_to_rgb("#c68642"))
    torso = library.CLOTHING[parsed.clothing]["torso"]
    assert mats[torso].rgba[:3] == pytest.approx(hex_to_rgb("#aa0000"))


def test_bad_custom_colour_is_rejected_without_changing_state() -> None:
    model = _model_cls()()
    before = model.appearance_document()
    for name in ("hair_color", "skin_color", "top_color", "bottom_color"):
        with pytest.raises(ValueError):
            model.set_appearance(name, "teal")
    assert model.appearance_document() == before


def test_clearing_a_colour_restores_the_library_default() -> None:
    model = _model_cls()()
    before = model.appearance_document()
    model.set_appearance("hair_color", "#336699")
    model.set_appearance("hair_color", None)
    assert model.appearance_document() == before


def test_custom_colour_never_changes_compile_hash() -> None:
    model = _model_cls()()
    before = model.compile().sha256
    model.set_appearance("top_color", "#aa0000")
    assert model.compile().sha256 == before


class TestColourPickers:
    @pytest.fixture
    def widget_factory(self):
        QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from src.tools.character_builder.gui import CharacterBuilderWidget

        made = []

        def make(color):
            w = CharacterBuilderWidget(color_picker=lambda current, parent: color)
            made.append(w)
            return w

        yield make
        for w in made:
            w.cleanup()
        del app

    def test_every_colour_field_exists(self, widget_factory) -> None:
        w = widget_factory("#112233")
        assert set(w.colour_fields) == {
            "skin_color",
            "hair_color",
            "top_color",
            "bottom_color",
        }

    @pytest.mark.parametrize(
        "name", ["skin_color", "hair_color", "top_color", "bottom_color"]
    )
    def test_injected_picker_writes_the_document(self, widget_factory, name) -> None:
        from PyQt6.QtWidgets import QPushButton

        w = widget_factory("#112233")
        w.colour_fields[name].findChild(QPushButton).click()
        assert w.model.appearance_choices()[name] == "#112233"
        assert w.status.text() == ""
        assert _document_from_dict()(w.model.appearance_document())

    def test_reset_clears_the_choice(self, widget_factory) -> None:
        from PyQt6.QtWidgets import QPushButton

        w = widget_factory("#112233")
        w.colour_fields["hair_color"].findChild(QPushButton).click()
        w.findChild(QPushButton, "reset_hair_color").click()
        assert "hair_color" not in w.model.appearance_choices()

    def test_hair_colour_with_cap_reports_instead_of_silently_ignoring(
        self, widget_factory
    ) -> None:
        from PyQt6.QtWidgets import QPushButton

        w = widget_factory("#112233")
        w.appearance_combos["headwear"].setCurrentText("cap")
        w.colour_fields["hair_color"].findChild(QPushButton).click()
        assert "hair_color" not in w.model.appearance_choices()
        assert "headwear" in w.status.text()
