"""Actual desktop painter receives force colors on playback and restores on off."""

import numpy as np
import pytest
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QApplication

from src.shared.python.body_part_viz import ForceColorScale
from src.shared.python.pendulum_simulator.gui.pendulum_widget import PendulumWidget
from src.shared.python.pendulum_simulator.physics import PendulumParams
from src.shared.python.pendulum_simulator.simulation import SimulationResult

pytestmark = pytest.mark.unit


def test_shared_menu_opens_controls_and_configures_existing_canvas():
    from PyQt6.QtWidgets import QCheckBox, QDialog, QMenu
    from src.shared.python.body_part_viz.force_color_controls import (
        install_force_color_action,
    )

    app = QApplication.instance() or QApplication([])
    widget = PendulumWidget()
    menu = QMenu(widget)
    action = install_force_color_action(menu, lambda: [widget])
    action.trigger()
    dialog = menu.findChild(QDialog)
    assert dialog is not None
    toggle = dialog.findChild(QCheckBox, "force_color_enabled")
    toggle.setChecked(True)
    assert widget._axial_color_scale.enabled
    toggle.setChecked(False)
    assert not widget._axial_color_scale.enabled
    dialog.close()
    widget.close()
    app.processEvents()


def test_native_painter_seek_and_toggle(monkeypatch):
    app = QApplication.instance() or QApplication([])
    widget = PendulumWidget()
    result = SimulationResult(
        np.array([0.0, 1.0]),
        np.array([[0.0, 0, 0, 0], [np.pi, 0, 0, 0]]),
        PendulumParams(m1=2, m2=3, L1=1, L2=1),
        lambda t: (0.0, 0.0),
    )
    widget.set_simulation(result)
    widget._3d_mode = True
    colors = []
    monkeypatch.setattr(
        widget, "_draw_3d_segment", lambda *args: colors.append(args[-1].name())
    )
    image = QImage(400, 400, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    try:
        widget.set_axial_color_scale(
            ForceColorScale(enabled=True, tension_limit_n=1, compression_limit_n=1)
        )
        widget._draw_pendulum(painter)
        assert colors == ["#0000ff", "#0000ff"]
        colors.clear()
        widget.set_frame(1)
        widget._draw_pendulum(painter)
        assert colors == ["#ff0000", "#ff0000"]
        colors.clear()
        widget.set_axial_color_scale(ForceColorScale())
        widget._draw_pendulum(painter)
        assert colors == [widget.COLOR_ARM.name(), widget.COLOR_CLUB.name()]
        with pytest.raises(TypeError):
            widget.set_axial_color_scale({})
    finally:
        painter.end()
        widget.close()
        app.processEvents()


def test_axial_segment_color_honours_opacity_and_tint():
    pytest.importorskip("PyQt6")
    from PyQt6.QtGui import QColor

    app = QApplication.instance() or QApplication([])
    widget = PendulumWidget()
    try:
        base = QColor("#ffff00")
        base.setAlphaF(0.5)
        widget.set_axial_color_scale(
            ForceColorScale(
                enabled=True, tension_limit_n=1, compression_limit_n=1, opacity=0.5
            )
        )
        shaded = widget._axial_segment_color(10.0, base)
        assert shaded.name() == "#808080"
        assert shaded.alphaF() == pytest.approx(0.5, abs=0.01)
        widget.set_axial_color_scale(
            ForceColorScale(enabled=True, opacity=0.5, blend_mode="tint")
        )
        assert widget._axial_segment_color(0.0, base).name() == "#ffff00"
        widget.set_axial_color_scale(ForceColorScale())
        assert widget._axial_segment_color(10.0, base).name() == "#ffff00"
    finally:
        widget.close()
        app.processEvents()
