"""End-to-end PyQt6 interaction tests for the Impact Parameters panel (#15522).

``tests/unit/impact_parameters/test_impact_panel_widget.py`` drives the widget
through Python calls (``setValue`` / ``.click()``). These tests launch it through
the real ``__main__.main`` entry point and then behave like a user: real
``QTest`` mouse clicks and key presses on the units combo, target spin box,
left-handed check box and "Open in Impact Explorer" button, asserting on what is
shown (card text, enabled state) and on the painted D-plane pixels.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtGui import QColor  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.impact_parameters import ClubheadSeries  # noqa: E402
from src.tools.impact_parameters_panel import __main__ as panel_main  # noqa: E402
from src.tools.impact_parameters_panel.gui import (  # noqa: E402
    UNAVAILABLE_TEXT,
    ImpactParametersWidget,
)

pytestmark = pytest.mark.unit

_N = 41
_IMPACT_INDEX = 30
_ARROW_BLUE = QColor("#2a7de1").rgb()
_APP: QApplication | None = None


def _ensure_qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # keep a module-level reference alive
    return app  # type: ignore[return-value]


def _series() -> ClubheadSeries:
    t = np.arange(_N) * 0.002
    vel = np.tile([0.0, -40.0, -3.0], (_N, 1))
    pos = np.cumsum(vel * 0.002, axis=0)
    return ClubheadSeries(
        t,
        pos,
        vel,
        face_normal=np.tile([0.0, -0.97, 0.22], (_N, 1)),
        toe_axis=np.tile([1.0, 0.0, 0.0], (_N, 1)),
        grip_axis=np.tile([0.0, 0.22, 0.97], (_N, 1)),
    )


def _launch(monkeypatch: pytest.MonkeyPatch) -> ImpactParametersWidget:
    """Run the real ``main()`` with the event loop stubbed; return its window."""
    app = _ensure_qapp()
    shown: list[ImpactParametersWidget] = []

    def _fake_exec(*_args: object) -> int:
        shown.extend(
            w
            for w in QApplication.topLevelWidgets()
            if isinstance(w, ImpactParametersWidget)
        )
        return 0

    monkeypatch.setattr(QApplication, "exec", _fake_exec)
    assert panel_main.main() == 0
    assert len(shown) == 1, "main() did not create exactly one panel"
    app.processEvents()
    return shown[0]


@pytest.fixture
def panel(monkeypatch: pytest.MonkeyPatch):
    widget = _launch(monkeypatch)
    yield widget
    widget.cleanup()
    widget.close()
    widget.deleteLater()
    QApplication.processEvents()


def _center(widget) -> QPoint:  # noqa: ANN001
    return widget.rect().center()


def _checkbox_hit_point(box) -> QPoint:  # noqa: ANN001
    # Click on the indicator at the left edge; the rect centre can miss it.
    return QPoint(8, box.height() // 2)


def _click(widget, pos: QPoint | None = None) -> None:  # noqa: ANN001
    QTest.mouseClick(
        widget,
        Qt.MouseButton.LeftButton,
        pos=pos if pos is not None else _center(widget),
    )


def _has_arrow_pixels(widget: ImpactParametersWidget) -> bool:
    image = widget.diagram.grab().toImage()
    return any(
        image.pixelColor(x, y).rgb() == _ARROW_BLUE
        for y in range(0, image.height(), 2)
        for x in range(0, image.width(), 2)
    )


def test_launch_shows_empty_panel(panel: ImpactParametersWidget) -> None:
    assert panel.isVisible()
    assert panel.windowTitle() == "Impact Parameters"
    assert UNAVAILABLE_TEXT in panel.status_label.text()
    assert not panel.explorer_button.isEnabled()
    assert panel.units_combo.currentText() == "mph"
    assert panel.target_spin.value() == 0.0
    assert not panel.left_handed.isChecked()
    assert not _has_arrow_pixels(panel)


def test_clicking_disabled_explorer_button_emits_nothing(
    panel: ImpactParametersWidget,
) -> None:
    got: list[dict] = []
    panel.open_impact_explorer_requested.connect(got.append)
    _click(panel.explorer_button)
    assert got == []


def test_loading_series_populates_card_and_diagram(
    panel: ImpactParametersWidget,
) -> None:
    panel.set_series(_series(), impact_index=_IMPACT_INDEX)
    QApplication.processEvents()
    assert panel.status_label.text().startswith("Impact at t=")
    assert panel._value_labels["clubhead_speed"].text().endswith("mph")
    assert panel.explorer_button.isEnabled()
    assert _has_arrow_pixels(panel)


def test_keyboard_on_units_combo_switches_units(
    panel: ImpactParametersWidget,
) -> None:
    panel.set_series(_series(), impact_index=_IMPACT_INDEX)
    mph_text = panel._value_labels["clubhead_speed"].text()
    panel.units_combo.setFocus()
    QTest.keyClick(panel.units_combo, Qt.Key.Key_Down)
    assert panel.units_combo.currentText() == "m/s"
    speed = panel._value_labels["clubhead_speed"].text()
    assert speed.endswith("m/s") and speed != mph_text
    QTest.keyClick(panel.units_combo, Qt.Key.Key_Up)
    assert panel._value_labels["clubhead_speed"].text() == mph_text


def test_keyboard_on_target_spin_changes_club_path(
    panel: ImpactParametersWidget,
) -> None:
    panel.set_series(_series(), impact_index=_IMPACT_INDEX)
    before = panel._value_labels["club_path_deg"].text()
    panel.target_spin.setFocus()
    for _ in range(5):
        QTest.keyClick(panel.target_spin, Qt.Key.Key_Up)
    assert panel.target_spin.value() == 5.0
    assert panel._value_labels["club_path_deg"].text() != before


def test_typing_into_target_spin_clamps_to_range(
    panel: ImpactParametersWidget,
) -> None:
    panel.set_series(_series(), impact_index=_IMPACT_INDEX)
    panel.target_spin.setFocus()
    panel.target_spin.selectAll()
    QTest.keyClicks(panel.target_spin, "45")
    QTest.keyClick(panel.target_spin, Qt.Key.Key_Return)
    assert panel.target_spin.value() == 45.0
    panel.target_spin.selectAll()
    QTest.keyClicks(panel.target_spin, "999")
    QTest.keyClick(panel.target_spin, Qt.Key.Key_Return)
    assert panel.target_spin.value() <= 180.0
    assert panel.status_label.text().startswith("Impact at t=")


def test_mouse_click_on_left_handed_checkbox_toggles_and_refreshes(
    panel: ImpactParametersWidget,
) -> None:
    panel.set_series(_series(), impact_index=_IMPACT_INDEX)
    panel.target_spin.setValue(20.0)
    right_path = panel._value_labels["club_path_deg"].text()
    _click(panel.left_handed, _checkbox_hit_point(panel.left_handed))
    assert panel.left_handed.isChecked()
    assert panel.status_label.text().startswith("Impact at t=")
    assert panel._value_labels["club_path_deg"].text() != right_path
    _click(panel.left_handed, _checkbox_hit_point(panel.left_handed))
    assert not panel.left_handed.isChecked()
    assert panel._value_labels["club_path_deg"].text() == right_path


def test_mouse_click_on_explorer_button_emits_delivery(
    panel: ImpactParametersWidget,
) -> None:
    panel.set_series(_series(), impact_index=_IMPACT_INDEX)
    got: list[dict] = []
    panel.open_impact_explorer_requested.connect(got.append)
    _click(panel.explorer_button)
    assert len(got) == 1
    assert "clubhead_speed" in got[0]
    assert "smash_factor" not in got[0]  # unavailable rows never leak a number
    assert "clubhead_speed=" in panel.explorer_query()


def test_clearing_series_returns_to_unavailable(
    panel: ImpactParametersWidget,
) -> None:
    panel.set_series(_series(), impact_index=_IMPACT_INDEX)
    assert panel.explorer_button.isEnabled()
    panel.set_series(None)
    QApplication.processEvents()
    assert UNAVAILABLE_TEXT in panel.status_label.text()
    assert not panel.explorer_button.isEnabled()
    assert panel.delivery() == {}
    assert not _has_arrow_pixels(panel)
