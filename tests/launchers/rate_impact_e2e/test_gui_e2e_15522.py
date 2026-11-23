"""Exercise the launcher-selected impact GUI with real rigid-body calculations."""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

try:
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent, Qt
    from PyQt6.QtTest import QSignalSpy, QTest
    from PyQt6.QtWidgets import QApplication
    import matplotlib  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"PyQt6/matplotlib not loadable: {exc}", allow_module_level=True)

from src.launchers.tools_repo_path import resolve_tools_repo

pytestmark = pytest.mark.unit
_qtest = cast(Any, QTest)


@pytest.fixture(params=[(1280, 800), (390, 844)], ids=["desktop", "phone"])
def explorer(qapp: QApplication, request: Any, monkeypatch: Any) -> Iterator[Any]:
    """Resolve the pinned provider, launch its registered window, and dispose it."""
    root = Path(__file__).resolve().parents[3]
    try:
        provider = resolve_tools_repo(root, None)
    except RuntimeError:
        provider = None
    if provider is None or not provider.pinned:
        # unit-test-gate does not materialize vendor/ud-tools (see #9361).
        pytest.skip(
            "requires the pinned vendored Tools checkout (vendor/ud-tools); "
            "not materialized in this CI job"
        )
    monkeypatch.syspath_prepend(str(provider.path / "src"))
    registration = importlib.import_module("rate_of_closure.gui_registration")
    config = registration.get_gui_info()["pyqt6"]
    module = importlib.import_module(config["module"])
    assert module.__file__ is not None
    assert Path(module.__file__).is_relative_to(provider.path)
    window = getattr(module, config["class"])(confirm_on_close=False)
    window.resize(*request.param)
    window.show()
    _qtest.qWait(1)
    assert window.isVisible()
    assert window.windowTitle() == "Rate of Closure Impact Explorer"
    try:
        yield window
    finally:
        assert window.close(), "Explorer must accept close without a live worker"
        assert not window._club_view.is_playing()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(window)


def _enter(spin: Any, value: str) -> None:
    """Commit a user edit through native keyboard events."""
    spin.setFocus()
    spin.selectAll()
    _qtest.keyClicks(spin, value)
    _qtest.keyClick(spin, Qt.Key.Key_Return)


def _value(window: Any, field: str) -> float:
    """Read the displayed number, independently of its selected unit suffix."""
    return float(window._rows[field].value_label.text().split()[0].rstrip("°"))


def test_typed_rotation_recomputes_rigid_body_closure(explorer: Any) -> None:
    """Pin CCV to the independent angular-velocity projection equation."""
    controls = explorer._controls
    changed = QSignalSpy(controls.scenarioChanged)
    for field, value in (
        ("omega_plane_dps", "1200"),
        ("omega_shaft_dps", "800"),
        ("lie_angle_deg", "60"),
        ("contact_duration_us", "500"),
    ):
        _enter(controls._spins[field], value)
    assert len(changed) >= 4
    scenario = controls.scenario()
    assert scenario.omega_plane_dps == 1200
    assert scenario.omega_shaft_dps == 800
    assert scenario.lie_angle_deg == 60
    expected = 800 * math.sin(math.radians(60)) + 1200 * math.cos(math.radians(60))
    assert _value(explorer, "closure_rate_dps") == pytest.approx(expected, abs=0.01)
    assert _value(explorer, "ccv_dps") == pytest.approx(expected, abs=0.01)
    assert _value(explorer, "closure_during_contact_deg") == pytest.approx(
        expected * 0.0005, abs=0.01
    )


def test_zero_rotation_then_restore_has_no_stale_metrics(explorer: Any) -> None:
    """A zero-rotation control clears deviations and can recover by typing."""
    controls = explorer._controls
    for field in ("omega_plane_dps", "omega_shaft_dps"):
        _enter(controls._spins[field], "0")
    for field in (
        "closure_rate_dps",
        "ccv_dps",
        "path_deviation_deg",
        "aoa_deviation_deg",
        "tangential_speed_mph",
        "closure_during_contact_deg",
    ):
        assert _value(explorer, field) == 0, field
    assert (
        "not closing"
        in explorer._rows["time_to_square_from_1deg_open_ms"].value_label.text()
    )
    _enter(controls._spins["omega_plane_dps"], "1000")
    scenario = controls.scenario()
    expected = 1000 * math.cos(math.radians(scenario.lie_angle_deg))
    assert _value(explorer, "ccv_dps") == pytest.approx(expected, abs=0.01)


def test_preset_keyboard_selection_republishes_results(explorer: Any) -> None:
    """Select the zero-rotation preset and restore the original via QTest."""
    controls = explorer._controls
    selector = controls._preset_combo
    initial = controls.scenario()
    initial_index = selector.currentIndex()
    initial_closure = _value(explorer, "ccv_dps")
    from rate_of_closure.presets import PRESETS

    zero_index = next(
        index
        for index in range(selector.count())
        if PRESETS[selector.itemText(index)].omega_plane_dps == 0
        and PRESETS[selector.itemText(index)].omega_shaft_dps == 0
    )
    selector.setFocus()
    _qtest.keyClick(selector, Qt.Key.Key_Home)
    for _ in range(zero_index):
        _qtest.keyClick(selector, Qt.Key.Key_Down)
    assert selector.currentIndex() == zero_index
    assert _value(explorer, "ccv_dps") == 0
    _qtest.keyClick(selector, Qt.Key.Key_Home)
    for _ in range(initial_index):
        _qtest.keyClick(selector, Qt.Key.Key_Down)
    assert controls.scenario() == initial
    assert _value(explorer, "ccv_dps") == initial_closure


def test_result_mouse_click_selects_physics_explanation(explorer: Any) -> None:
    """Click a result in the real scroll area and display its explanation."""
    from PyQt6.QtWidgets import QScrollArea

    row = explorer._rows["closure_rate_dps"]
    sidebar = explorer._shell_splitter.widget(0)
    assert isinstance(sidebar, QScrollArea)
    sidebar.ensureWidgetVisible(row)
    _qtest.mouseClick(row, Qt.MouseButton.LeftButton)
    assert row.property("selected") is True
    assert explorer._rows["path_deviation_deg"].property("selected") is False
    assert "Closure Rate (CCV)" in explorer._explanation.toPlainText()
