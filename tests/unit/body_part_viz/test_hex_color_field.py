"""Shared Qt colour field with an injectable picker (HRG-B2, #13167)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication, QLineEdit, QPushButton  # noqa: E402

from src.shared.python.body_part_viz.force_color_controls import (  # noqa: E402
    ForceColorControls,
)
from src.shared.python.ui.hex_color_field import HexColorField  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def app():
    yield QApplication.instance() or QApplication([])


def test_injected_picker_sets_the_colour_and_emits(app):
    seen = []
    field = HexColorField(
        "#102030", object_name="x", color_picker=lambda c, p: "#AABBCC"
    )
    field.colour_changed.connect(seen.append)
    field.findChild(QPushButton).click()
    assert field.text() == "#aabbcc"
    assert seen == ["#aabbcc"]
    field.close()


def test_cancelled_picker_keeps_the_colour(app):
    field = HexColorField("#102030", color_picker=lambda c, p: None)
    field.findChild(QPushButton).click()
    assert field.text() == "#102030"
    field.close()


def test_picker_receives_the_current_colour(app):
    got = []
    field = HexColorField("#102030", color_picker=lambda c, p: got.append(c))
    field.findChild(QPushButton).click()
    assert got == ["#102030"]
    field.close()


def test_invalid_initial_and_typed_colours_are_rejected(app):
    with pytest.raises(ValueError, match="hex"):
        HexColorField("red")
    field = HexColorField("#102030")
    field.set_text("#zzzzzz")
    assert field.text() == "#102030"
    field.close()


def test_force_color_controls_use_the_shared_picker(app):
    panel = ForceColorControls(color_picker=lambda c, p: "#00ff00")
    edit = panel.findChild(QLineEdit, "tension_color")
    edit.parent().findChild(QPushButton).click()
    assert edit.text() == "#00ff00"
    panel.findChild(QPushButton, "apply_force_colors").click()
    received = []
    panel.scale_changed.connect(received.append)
    panel.findChild(QPushButton, "apply_force_colors").click()
    assert received[-1].tension_color == "#00ff00"
    panel.close()
