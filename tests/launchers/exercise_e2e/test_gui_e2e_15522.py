"""Launch the real exercise host and interact with its committed lift evidence.

These tests exercise receipt viewing and capability reporting, not live JaxSim
simulation. No engine dashboard, receipt loader or Qt widget is replaced.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import cast

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QAbstractButton, QApplication, QComboBox, QLabel, QWidget

from src.launchers.exercise_dashboard import ExerciseDashboard, get_dockable_ui
from src.launchers.jaxsim_dashboard import JaxSimDashboard
from src.launchers.lift_baseline_panel import LiftBaselinePanel

pytestmark = pytest.mark.unit

# QTest is a non-instantiable C++ namespace; some PyQt6 stubs give these
# static functions an erroneous self parameter. Preserve their runtime calls.
_key_click = cast(Callable[[QWidget, Qt.Key], None], QTest.keyClick)
_mouse_click = cast(Callable[[QWidget, Qt.MouseButton], None], QTest.mouseClick)


@pytest.fixture(params=[(1280, 800), (390, 844)], ids=["desktop", "phone"])
def exercise_window(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[ExerciseDashboard]:
    """Use the launcher factory; always dispose of the complete native Qt tree."""
    monkeypatch.setenv("BIOMECH_EXERCISE", "squat")
    monkeypatch.setenv("BIOMECH_ENGINE", "JaxSim_Models")
    window = get_dockable_ui()
    assert isinstance(window, ExerciseDashboard)
    window.resize(*request.param)
    window.show()
    qapp.processEvents()
    try:
        assert window.isVisible()
        assert isinstance(window._current_widget, JaxSimDashboard)
        yield window
    finally:
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
        assert sip.isdeleted(window)


def _panel(window: ExerciseDashboard) -> LiftBaselinePanel:
    dock = window.lift_baseline_dock
    assert dock is not None
    panel = dock.widget()
    assert isinstance(panel, LiftBaselinePanel)
    assert panel.error_message() is None
    return panel


def _choose(combo: QComboBox, index: int) -> None:
    """Select through keyboard events, including the normal Qt change signal."""
    assert 0 <= index < combo.count()
    combo.setFocus()
    _key_click(combo, Qt.Key.Key_Home)
    for _ in range(index):
        _key_click(combo, Qt.Key.Key_Down)
    assert combo.currentIndex() == index


def test_lift_selection_refreshes_real_receipt(
    exercise_window: ExerciseDashboard,
) -> None:
    panel = _panel(exercise_window)
    selector = panel.findChild(QComboBox, "lift-selector")
    assert selector is not None
    assert panel.current_lift() == "squat"
    assert panel.engine_row_count() == 4
    assert "lift-pack-parity-baseline/v1" in panel.provenance_text()

    _choose(selector, selector.findData("deadlift"))
    assert panel.current_lift() == "deadlift"
    # Committed LIFT-1 MuJoCo deadlift: 18 bodies, 56 q, 200.000001 kg.
    assert panel.cell_text("engines", 0, 0) == "MuJoCo"
    assert panel.cell_text("engines", 0, 2) == "18/56"
    assert panel.cell_text("engines", 0, 3) == "200.00 kg"
    assert panel.phase_row_count() > 0
    _choose(selector, selector.findData("squat"))
    assert panel.current_lift() == "squat"


def test_phase_and_pose_controls_refresh_tables(
    exercise_window: ExerciseDashboard,
) -> None:
    panel = _panel(exercise_window)
    lift = panel.findChild(QComboBox, "lift-selector")
    phase = panel.findChild(QComboBox, "phase-engine-selector")
    pose = panel.findChild(QComboBox, "pose-selector")
    assert lift is not None and phase is not None and pose is not None
    _choose(lift, lift.findData("deadlift"))
    assert phase.count() == 4
    _choose(phase, phase.findData("drake"))
    assert panel.phase_row_count() > 0
    assert panel.cell_text("phases", 0, 0) == "setup"
    assert panel.cell_text("phases", 0, 1) == "0.00"
    assert panel.cell_text("phases", 0, 2) == "44.5 mm"
    assert pose.count() > 1
    before = panel.cell_text("pairs", 0, 3)
    _choose(pose, pose.count() - 1)
    assert pose.currentText() == "zero"
    assert panel.pair_row_count() == 6
    assert panel.cell_text("pairs", 0, 3) == "57.0 mm"
    assert panel.cell_text("pairs", 0, 3) != before
    for row in range(panel.pair_row_count()):
        # Missing evidence must be explicit, never a blank or implied zero.
        assert panel.cell_text("pairs", row, 0)
    _choose(phase, phase.findData("mujoco"))
    assert panel.phase_row_count() > 0


def test_dock_toggle_preserves_capability_view(
    exercise_window: ExerciseDashboard,
) -> None:
    dock = exercise_window.lift_baseline_dock
    assert dock is not None
    dashboard = exercise_window._current_widget
    assert isinstance(dashboard, JaxSimDashboard)
    sensitivity = dashboard.parameter_sensitivity_button
    assert not sensitivity.isEnabled()
    assert "6656" in sensitivity.toolTip()
    note = exercise_window.findChild(QLabel, "lift-baseline-note")
    assert note is not None
    assert note.text() == "Derived, provisional; not approved."

    close = dock.findChild(QAbstractButton, "qt_dockwidget_closebutton")
    assert close is not None
    _mouse_click(close, Qt.MouseButton.LeftButton)
    assert not dock.isVisible()
    # Restore through the dock's native public action, as a View menu would.
    dock.toggleViewAction().trigger()
    assert dock.isVisible()
    assert exercise_window._current_widget is dashboard
    assert _panel(exercise_window).current_lift() == "squat"
