"""Reusable desktop controls publish only validated settings."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QMenu,
    QPushButton,
    QSlider,
    QToolButton,
    QWidget,
)

from src.shared.python.body_part_viz.force_color_controls import (  # noqa: E402
    ForceColorControls,
    ForceColorHub,
    InlineForceColorControls,
    install_force_color_action,
)
from src.shared.python.body_part_viz.force_colors import ForceColorScale  # noqa: E402

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


def test_opacity_slider_updates_scale_opacity(app):
    panel = ForceColorControls()
    received = []
    panel.scale_changed.connect(received.append)
    slider = panel.findChild(QSlider, "force_color_opacity")
    assert (slider.minimum(), slider.maximum(), slider.value()) == (0, 100, 100)
    slider.setValue(35)
    assert received[-1].opacity == pytest.approx(0.35)
    panel.close()


def test_blend_mode_combo_offers_tint_and_replace(app):
    panel = ForceColorControls()
    received = []
    panel.scale_changed.connect(received.append)
    combo = panel.findChild(QComboBox, "force_color_blend_mode")
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert labels == ["Tint over model", "Replace colour"]
    combo.setCurrentIndex(combo.findData("tint"))
    assert received[-1].blend_mode == "tint"
    combo.setCurrentIndex(combo.findData("replace"))
    assert received[-1].blend_mode == "replace"
    panel.close()


def test_injected_picker_sets_hex_swatch_and_scale(app):
    panel = ForceColorControls()
    received = []
    panel.scale_changed.connect(received.append)
    picks = []
    panel._color_picker = lambda current, parent: picks.append(current) or "#00FF00"
    button = panel.findChild(QToolButton, "pick_tension_color")
    button.click()
    assert picks == ["#0000ff"]
    assert panel.findChild(QLineEdit, "tension_color").text() == "#00ff00"
    assert received[-1].tension_color == "#00ff00"
    assert "#00ff00" in button.styleSheet()
    panel.close()


def test_cancelled_picker_changes_nothing(app):
    panel = ForceColorControls()
    received = []
    panel.scale_changed.connect(received.append)
    panel._color_picker = lambda current, parent: None
    panel.findChild(QToolButton, "pick_compression_color").click()
    assert received == []
    assert panel.findChild(QLineEdit, "compression_color").text() == "#ff0000"
    panel.close()


def test_every_color_field_has_a_picker_button(app):
    panel = ForceColorControls()
    for name in ("tension_color", "compression_color", "neutral_color"):
        assert panel.findChild(QToolButton, f"pick_{name}") is not None
    panel.close()


def test_reset_colours_restores_defaults_but_keeps_other_settings(app):
    panel = ForceColorControls(
        scale=ForceColorScale(
            enabled=True, tension_color="#00ff00", opacity=0.5, tension_limit_n=300.0
        )
    )
    received = []
    panel.scale_changed.connect(received.append)
    panel.findChild(QPushButton, "reset_force_colors").click()
    last = received[-1]
    assert (last.tension_color, last.compression_color, last.neutral_color) == (
        ForceColorScale().tension_color,
        ForceColorScale().compression_color,
        ForceColorScale().neutral_color,
    )
    assert last.enabled and last.opacity == 0.5 and last.tension_limit_n == 300.0
    assert panel.findChild(QLineEdit, "tension_color").text() == "#0000ff"
    panel.close()


def test_set_scale_updates_widgets_without_emitting(app):
    panel = ForceColorControls()
    received = []
    panel.scale_changed.connect(received.append)
    panel.set_scale(ForceColorScale(enabled=True, opacity=0.2, blend_mode="tint"))
    assert received == []
    assert panel.findChild(QCheckBox, "force_color_enabled").isChecked()
    assert panel.findChild(QSlider, "force_color_opacity").value() == 20
    combo = panel.findChild(QComboBox, "force_color_blend_mode")
    assert combo.currentData() == "tint"
    assert panel.scale.opacity == pytest.approx(0.2)
    with pytest.raises(TypeError):
        panel.set_scale("x")
    panel.close()


def test_apply_edits_keeps_opacity_and_mode(app):
    panel = ForceColorControls()
    panel.findChild(QSlider, "force_color_opacity").setValue(40)
    panel.findChild(QComboBox, "force_color_blend_mode").setCurrentIndex(0)
    received = []
    panel.scale_changed.connect(received.append)
    panel.findChild(QPushButton, "apply_force_colors").click()
    assert received[-1].opacity == pytest.approx(0.4)
    assert received[-1].blend_mode == "tint"
    panel.close()


def test_hub_deduplicates_and_validates(app):
    hub = ForceColorHub()
    seen = []
    hub.scale_changed.connect(seen.append)
    hub.set_scale(ForceColorScale())
    assert seen == []
    hub.set_scale(ForceColorScale(enabled=True))
    assert len(seen) == 1 and hub.scale.enabled
    with pytest.raises(TypeError):
        hub.set_scale(None)


def test_inline_controls_follow_hub_both_ways(app):
    hub = ForceColorHub()
    inline = InlineForceColorControls(hub, show_opacity=True)
    check = inline.findChild(QCheckBox, "force_color_inline_enabled")
    slider = inline.findChild(QSlider, "force_color_inline_opacity")
    check.setChecked(True)
    assert hub.scale.enabled
    slider.setValue(25)
    assert hub.scale.opacity == pytest.approx(0.25)
    hub.set_scale(ForceColorScale(enabled=False, opacity=0.9))
    assert not check.isChecked() and slider.value() == 90
    inline.close()


def test_inline_controls_without_opacity_has_no_slider(app):
    inline = InlineForceColorControls(ForceColorHub())
    assert inline.findChild(QSlider, "force_color_inline_opacity") is None
    inline.close()


def test_dialog_and_hub_stay_in_sync_and_drive_targets(app):
    class Target:
        def __init__(self):
            self.scales = []

        def set_axial_color_scale(self, scale):
            self.scales.append(scale)

    target = Target()
    hub = ForceColorHub()
    parent = QWidget()
    menu = QMenu(parent)
    action = install_force_color_action(menu, lambda: [target], hub=hub)
    hub.set_scale(ForceColorScale(enabled=True))
    assert target.scales[-1].enabled
    action.trigger()
    dialog = parent.findChild(QWidget, "segment_force_colors_dialog")
    assert dialog is not None
    controls = dialog.findChild(ForceColorControls)
    assert controls.findChild(QCheckBox, "force_color_enabled").isChecked()
    controls.findChild(QSlider, "force_color_opacity").setValue(10)
    assert hub.scale.opacity == pytest.approx(0.1)
    assert target.scales[-1].opacity == pytest.approx(0.1)
    hub.set_scale(ForceColorScale(enabled=False, opacity=0.6))
    assert not controls.findChild(QCheckBox, "force_color_enabled").isChecked()
    assert controls.findChild(QSlider, "force_color_opacity").value() == 60
    dialog.close()
    parent.close()


class _Store:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def value(self, key, default=None):
        return self.data.get(key, default)

    def setValue(self, key, value):  # noqa: N802
        self.data[key] = value


def test_install_inline_shading_toggle_binds_host_attr(app):
    from PyQt6.QtWidgets import QVBoxLayout

    from src.shared.python.body_part_viz.force_color_controls import (
        install_inline_shading_toggle,
    )

    host = QWidget()
    host.force_color_hub = ForceColorHub()
    layout = QVBoxLayout()
    controls = install_inline_shading_toggle(host, layout, attr="chk_shading")
    assert controls is not None and layout.count() == 1
    host.chk_shading.setChecked(True)
    assert host.force_color_hub.scale.enabled
    host.force_color_hub.update_scale(enabled=False)
    assert not host.chk_shading.isChecked()
    bare = QWidget()
    assert install_inline_shading_toggle(bare, layout, attr="chk_shading") is None
    assert layout.count() == 1


def test_bind_persistence_restores_and_saves(app):
    from src.shared.python.body_part_viz.force_color_controls import (
        bind_force_color_persistence,
    )
    from src.shared.python.body_part_viz.force_color_settings import (
        save_force_color_state,
    )

    store = _Store()
    save_force_color_state(
        store, "x", ForceColorScale(enabled=True, opacity=0.3), {"forces": True}
    )
    hub = ForceColorHub()
    forces, torques = QCheckBox(), QCheckBox()
    bind_force_color_persistence(
        hub, "x", {"forces": forces, "torques": torques}, lambda: store
    )
    assert hub.scale.enabled and hub.scale.opacity == pytest.approx(0.3)
    assert forces.isChecked() and not torques.isChecked()
    torques.setChecked(True)
    hub.update_scale(opacity=0.8)
    hub2 = ForceColorHub()
    bind_force_color_persistence(hub2, "x", {}, lambda: store)
    assert hub2.scale.opacity == pytest.approx(0.8)


def test_bind_persistence_corrupt_value_uses_defaults(app):
    from src.shared.python.body_part_viz.force_color_controls import (
        bind_force_color_persistence,
    )

    hub = ForceColorHub()
    box = QCheckBox()
    bind_force_color_persistence(
        hub, "x", {"forces": box}, lambda: _Store({"force_color/x": "garbage"})
    )
    assert hub.scale == ForceColorScale() and not box.isChecked()
