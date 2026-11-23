"""Native QTest launch-and-interact flows for the JaxSim dashboard (#15522).

The real ``main()`` entry point, the real ``JaxSimBackend`` capability report
and real Qt widgets run offscreen. JaxSim/JAX themselves are never imported
(Linux-only), and nothing in the dashboard is mocked: the only seam replaced is
the blocking ``QApplication.exec`` event loop, so ``main()`` can be launched
and inspected inside a test worker.
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytestmark = pytest.mark.unit

PHONE_WIDTH = 390
PHONE_HEIGHT = 844
DESKTOP_WIDTH = 1280
DESKTOP_HEIGHT = 800


@pytest.fixture
def qt():
    """Yield ``(QtCore, QtTest.QTest, QtWidgets)`` or skip when Qt is absent."""
    try:
        from PyQt6 import QtCore, QtTest, QtWidgets
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return QtCore, QtTest.QTest, QtWidgets


@pytest.fixture
def dashboard(qt):
    from src.launchers.jaxsim_dashboard import JaxSimDashboard

    widget = JaxSimDashboard(exercise_filter="gait")
    widget.resize(DESKTOP_WIDTH, DESKTOP_HEIGHT)
    widget.show()
    qt[2].QApplication.processEvents()
    assert widget.isVisible()
    yield widget
    widget.close()
    widget.deleteLater()


def _state(widget) -> dict[str, tuple[str, bool, str]]:
    return {
        name: (label.text(), label.isEnabled(), label.toolTip())
        for name, label in widget.feature_controls.items()
    }


def test_main_launches_real_window_for_exercise(qt, monkeypatch) -> None:
    """``main --exercise squat`` shows a visible, titled, populated window."""
    core, _qtest, widgets = qt
    from src.launchers import jaxsim_dashboard

    seen: list = []

    def fake_exec(_self=None) -> int:
        seen.extend(
            w
            for w in widgets.QApplication.topLevelWidgets()
            if isinstance(w, jaxsim_dashboard.JaxSimDashboard)
        )
        return 0

    monkeypatch.setattr(sys, "argv", ["jaxsim_dashboard", "--exercise", "squat"])
    monkeypatch.setattr(widgets.QApplication, "exec", fake_exec)
    with pytest.raises(SystemExit) as exit_info:
        jaxsim_dashboard.main()
    assert exit_info.value.code == 0
    assert len(seen) == 1
    window = seen[0]
    try:
        assert window.isVisible()
        assert window.windowTitle() == "JaxSim Dashboard"
        assert window.exercise_filter == "squat"
        heading = window.findChildren(widgets.QLabel)[0]
        assert heading.text() == "JaxSim — Squat (Linux CI engine)"
    finally:
        window.close()
        window.deleteLater()


def test_indicators_reflect_real_backend_capability_levels(dashboard) -> None:
    from src.engines.physics_engines.jaxsim import JaxSimBackend
    from src.launchers.jaxsim_dashboard import _FEATURE_ROWS
    from src.shared.python.engine_core.capabilities import CapabilityLevel

    caps = JaxSimBackend().get_capabilities()
    glyphs = {
        CapabilityLevel.FULL: "✓",
        CapabilityLevel.PARTIAL: "–",
        CapabilityLevel.NONE: "✗",
    }
    for label, attr in _FEATURE_ROWS:
        level = getattr(caps, attr)
        indicator = dashboard.feature_controls[label]
        assert indicator.text() == f"{glyphs[level]}  {label}"
        assert indicator.isEnabled() is (level == CapabilityLevel.FULL)
        assert "read-only" in indicator.toolTip()
        assert indicator.isVisible()


def test_clicks_and_keys_never_change_read_only_indicators(dashboard, qt) -> None:
    core, qtest, widgets = qt
    before = _state(dashboard)
    for label in dashboard.feature_controls.values():
        qtest.mouseClick(label, core.Qt.MouseButton.LeftButton)
        qtest.mouseDClick(label, core.Qt.MouseButton.LeftButton)
        qtest.keyClick(label, core.Qt.Key.Key_Space)
        qtest.keyClick(label, core.Qt.Key.Key_Return)
        assert label.focusPolicy() == core.Qt.FocusPolicy.NoFocus
        assert (
            label.textInteractionFlags()
            == core.Qt.TextInteractionFlag.NoTextInteraction
        )
    widgets.QApplication.processEvents()
    assert _state(dashboard) == before
    assert not dashboard.findChildren(widgets.QAbstractButton)


def test_parameter_sensitivity_stub_stays_gated_on_click(dashboard, qt) -> None:
    core, qtest, _widgets = qt
    from src.launchers.jaxsim_dashboard import PARAMETER_SENSITIVITY_GATED_ISSUE

    stub = dashboard.parameter_sensitivity_button
    assert stub is dashboard.feature_controls["Parameter sensitivity (ZTCF)"]
    qtest.mouseClick(stub, core.Qt.MouseButton.LeftButton)
    assert not stub.isEnabled()
    assert f"#{PARAMETER_SENSITIVITY_GATED_ISSUE}" in stub.toolTip()
    assert "not implemented" in stub.toolTip()


def test_injected_capability_levels_render_all_three_glyphs(qt) -> None:
    from src.launchers.jaxsim_dashboard import JaxSimDashboard
    from src.shared.python.engine_core.capabilities import (
        CapabilityLevel,
        EngineCapabilities,
    )

    caps = EngineCapabilities(
        engine_name="JaxSim",
        forward_sim=CapabilityLevel.FULL,
        mass_matrix=CapabilityLevel.PARTIAL,
        inverse_dynamics=CapabilityLevel.NONE,
    )
    widget = JaxSimDashboard(exercise_filter="swing", capabilities=caps)
    widget.show()
    qt[2].QApplication.processEvents()
    try:
        controls = widget.feature_controls
        assert controls["Forward simulation"].text().startswith("✓")
        assert controls["Forward simulation"].isEnabled()
        assert controls["Mass matrix M(q)"].text().startswith("–")
        assert not controls["Mass matrix M(q)"].isEnabled()
        assert controls["Inverse dynamics"].text().startswith("✗")
        assert "not implemented" in controls["Inverse dynamics"].toolTip()
    finally:
        widget.close()
        widget.deleteLater()


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_exercise_filter_is_rejected(qt, blank) -> None:
    from src.launchers.jaxsim_dashboard import JaxSimDashboard

    with pytest.raises(ValueError, match="non-empty"):
        JaxSimDashboard(exercise_filter=blank)


def test_dockable_widget_fits_phone_viewport(qt) -> None:
    """Docked into a phone-sized (390x844) host, no row is clipped."""
    core, _qtest, widgets = qt
    from src.launchers.jaxsim_dashboard import JaxSimDashboard, get_dockable_ui

    docked = get_dockable_ui()
    assert isinstance(docked, JaxSimDashboard)
    host = widgets.QMainWindow()
    host.setCentralWidget(docked)
    host.resize(PHONE_WIDTH, PHONE_HEIGHT)
    host.show()
    widgets.QApplication.processEvents()
    try:
        assert docked.minimumSizeHint().width() <= PHONE_WIDTH
        for label in docked.findChildren(widgets.QLabel):
            assert label.isVisible()
            assert label.sizeHint().width() <= docked.width()
    finally:
        host.close()
        host.deleteLater()
