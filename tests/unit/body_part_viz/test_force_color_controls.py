"""Reusable desktop controls publish only validated settings."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication, QCheckBox, QLineEdit, QPushButton  # noqa: E402

from src.shared.python.body_part_viz.force_color_controls import ForceColorControls  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def test_toggle_and_custom_colors_emit_validated_settings(app):
    panel = ForceColorControls()
    received = []
    panel.scale_changed.connect(received.append)
    toggle = panel.findChild(QCheckBox, "force_color_enabled")
    assert not toggle.isChecked()
    toggle.setChecked(True)
    assert received[-1].enabled
    field = panel.findChild(QLineEdit, "tension_color")
    field.setText("#00ff00")
    panel.findChild(QPushButton, "apply_force_colors").click()
    assert received[-1].tension_color == "#00ff00"
    toggle.setChecked(False)
    assert not received[-1].enabled
    panel.close()


def test_invalid_edit_preserves_last_valid_scale_and_toggle_still_works(app):
    panel = ForceColorControls()
    received = []
    panel.scale_changed.connect(received.append)
    panel.findChild(QLineEdit, "tension_color").setText("invalid")
    panel.findChild(QPushButton, "apply_force_colors").click()
    assert received == []
    panel.findChild(QCheckBox, "force_color_enabled").setChecked(True)
    assert received[-1].tension_color == "#0000ff"
    panel.close()
