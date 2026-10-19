"""Desktop Character Builder grip presets and sliders (HRG-GR4, #13205)."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.unit

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def widget():
    try:
        from PyQt6 import QtWidgets
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from src.tools.character_builder.gui import CharacterBuilderWidget

    w = CharacterBuilderWidget()
    yield w
    w.cleanup()
    w.deleteLater()
    del app


def test_a_preset_sets_the_hands_and_a_new_revision(widget) -> None:
    parent = widget.model.compile().sha256
    index = widget.grip_preset_combo.findData("strong")
    widget.grip_preset_combo.setCurrentIndex(index)
    assert widget.model.grip_picks() == {"preset": "strong"}
    assert widget.grip_spins["lead_shaft_rotation"].value() == 15.0
    assert widget.grip_spins["trail_palm_diagonal"].value() == 40.0
    assert widget.model.compile().sha256 != parent
    assert "refit" in widget.summary.toPlainText()


def test_a_slider_is_a_pick_and_reset_clears_it(widget) -> None:
    widget.grip_spins["lead_palm_diagonal"].setValue(55.0)
    assert widget.model.grip_picks() == {"lead_palm_diagonal": 55.0}
    parent = widget.model.compile().sha256
    widget.findChild(type(widget.export_buttons["spec"]), "reset_grip").click()
    assert widget.model.grip_picks() == {}
    assert widget.model.compile().sha256 != parent  # back to the unrevised spec


def test_style_is_picture_only_and_lands_in_the_sidecar(widget) -> None:
    from src.shared.python.model_appearance.schema import physics_spec_sha256

    before = physics_spec_sha256(widget.model.compile().document)
    widget.grip_style_combo.setCurrentIndex(
        widget.grip_style_combo.findData("interlock")
    )
    assert physics_spec_sha256(widget.model.compile().document) == before
    assert widget.model.appearance_document()["grip"] == {"style": "interlock"}


def test_a_refused_pick_shows_the_message_and_changes_nothing(widget) -> None:
    # Lead alpha -60 is a valid lead V (-40 deg, the G4 range floor) but leaves
    # the untouched trail hand 60 deg stronger, outside the trail-offset range.
    widget.grip_spins["lead_shaft_rotation"].setValue(-60.0)
    assert widget.model.grip_picks() == {}
    assert "trail_v_offset_deg" in widget.status.text()
