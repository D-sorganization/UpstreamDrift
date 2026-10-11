"""Offscreen PyQt6 tests for the impact parameters widget (GCV-17)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.impact_parameters import ClubheadSeries

pytestmark = pytest.mark.unit

N = 41


_APP = None


def _qt():
    global _APP
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _series(face=True):
    t = np.arange(N) * 0.002
    vel = np.tile([0.0, -40.0, -3.0], (N, 1))
    pos = np.cumsum(vel * 0.002, axis=0)
    if face:
        return ClubheadSeries(
            t,
            pos,
            vel,
            face_normal=np.tile([0.0, -0.97, 0.22], (N, 1)),
            toe_axis=np.tile([1.0, 0.0, 0.0], (N, 1)),
            grip_axis=np.tile([0.0, 0.22, 0.97], (N, 1)),
        )
    return ClubheadSeries(t, pos, vel, face_unobservable_reason="roll unobservable")


def test_widget_shows_values_and_unavailable_states():
    _qt()
    from src.tools.impact_parameters_panel.gui import (
        UNAVAILABLE_TEXT,
        ImpactParametersWidget,
    )

    w = ImpactParametersWidget()
    w.set_series(None)
    assert UNAVAILABLE_TEXT in w.status_label.text()
    assert not w.explorer_button.isEnabled()

    w.set_series(_series(), impact_index=30)
    assert w._value_labels["clubhead_speed"].text().endswith("mph")
    assert w._value_labels["smash_factor"].text() == UNAVAILABLE_TEXT
    assert w._value_labels["smash_factor"].toolTip()
    assert w.explorer_button.isEnabled()

    w.set_series(_series(face=False), impact_index=30)
    assert w._value_labels["face_angle_deg"].text() == UNAVAILABLE_TEXT


def test_units_and_target_inputs_update_card():
    _qt()
    from src.tools.impact_parameters_panel.gui import ImpactParametersWidget

    w = ImpactParametersWidget()
    w.set_series(_series(), impact_index=30)
    mph = w._value_labels["clubhead_speed"].text()
    w.units_combo.setCurrentText("m/s")
    assert w._value_labels["clubhead_speed"].text().endswith("m/s")
    assert w._value_labels["clubhead_speed"].text() != mph
    path0 = w._value_labels["club_path_deg"].text()
    w.target_spin.setValue(10.0)
    assert w._value_labels["club_path_deg"].text() != path0
    w.cleanup()
    w.cleanup()


def test_explorer_signal_carries_delivery():
    _qt()
    from src.tools.impact_parameters_panel.gui import ImpactParametersWidget

    w = ImpactParametersWidget()
    w.set_series(_series(), impact_index=30)
    got = []
    w.open_impact_explorer_requested.connect(got.append)
    w.explorer_button.click()
    assert "clubhead_speed" in got[0] and "smash_factor" not in got[0]
    assert "clubhead_speed=" in w.explorer_query()


def test_embed_adapter_contract_headless():
    from src.shared.python.launcher_embed import get_embeddable_tool
    import src.tools.impact_parameters_panel._embed_adapter  # noqa: F401

    tool = get_embeddable_tool("impact_parameters")
    assert tool is not None and tool.embed_capabilities().prefers_dock
    tool.cleanup()
    tool.cleanup()
    assert tool.is_dirty() is False


def test_render_offscreen_grab():
    _qt()
    from src.tools.impact_parameters_panel.gui import ImpactParametersWidget

    w = ImpactParametersWidget()
    w.set_series(_series(), impact_index=30)
    w.resize(420, 640)
    image = w.grab()
    assert not image.isNull() and image.width() == 420
