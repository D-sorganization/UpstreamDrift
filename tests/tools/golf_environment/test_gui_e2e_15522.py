"""Native offscreen launcher flows for Golf Environment Viewer (Refs #15522).

Only the GPU viewport is replaced. Course meshes, line items, trajectory
colouring, launcher factory, Qt signals and input dispatch remain real.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph.opengl")

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget

from src.tools.golf_environment import gui

pytestmark = pytest.mark.unit


class _HeadlessViewport(QWidget):
    """Retain real scene items without creating an unsupported offscreen GL context."""

    def __init__(self) -> None:
        super().__init__()
        self.opts: dict[str, Any] = {}
        self.items: list[Any] = []

    def addItem(self, item: Any) -> None:  # noqa: N802 - pyqtgraph protocol
        self.items.append(item)

    def removeItem(self, item: Any) -> None:  # noqa: N802 - pyqtgraph protocol
        self.items.remove(item)

    def clear(self) -> None:
        self.items.clear()


@pytest.fixture
def window(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Iterator[gui.EnvironmentWindow]:
    monkeypatch.setattr(gui.gl, "GLViewWidget", _HeadlessViewport)
    instance = gui.get_dockable_ui()
    instance.show()
    qapp.processEvents()
    try:
        yield instance
    finally:
        instance.close()
        instance.deleteLater()
        qapp.processEvents()


def _select(window: gui.EnvironmentWindow, key: Qt.Key, expected: str) -> None:
    """Dispatch native input, never call the change handler directly."""
    combo = window.env_combo
    QTest.mouseClick(combo, Qt.MouseButton.LeftButton)
    QTest.keyClick(combo, key)
    QTest.keyClick(combo, Qt.Key.Key_Return)
    QApplication.processEvents()
    assert combo.currentText() == expected


def test_launcher_draws_range_and_height_shaded_ballistic_demo(
    window: gui.EnvironmentWindow,
) -> None:
    assert window.isVisible()
    assert window.windowTitle() == "Golf Environment Viewer"
    renderer = window.renderer
    assert isinstance(renderer._environment, gui.DrivingRange)
    view = renderer._gl_view
    assert len(view.items) == 8  # ground, six yardage markers, demo flight
    for marker, yards in zip(
        view.items[1:7], (50, 100, 150, 200, 250, 300), strict=True
    ):
        np.testing.assert_allclose(marker.pos[:, 0], yards * 0.9144, rtol=1e-6)
    flight = renderer._trajectories[0]
    assert flight in view.items
    points = flight.pos
    elapsed = points[:, 0] / 50.0
    np.testing.assert_allclose(
        points[:, 2], 25.0 * elapsed - 0.5 * 9.80665 * elapsed**2, atol=1e-5
    )
    assert np.all(points[:, 2] >= 0.0)
    assert np.ptp(points[:, 2]) > 30.0
    assert flight.color.shape == (len(points), 4)
    assert not np.allclose(flight.color[0], flight.color[np.argmax(points[:, 2])])


def test_native_selector_rebuilds_physical_course_geometry(
    window: gui.EnvironmentWindow,
) -> None:
    renderer = window.renderer
    initial_flight = renderer._trajectories[0]
    for key, label, par, yards in (
        (Qt.Key.Key_Down, "Par 3 (150y)", 3, 150),
        (Qt.Key.Key_Down, "Par 4 (400y)", 4, 400),
    ):
        _select(window, key, label)
        hole = renderer._environment
        assert isinstance(hole, gui.CourseHole)
        assert (hole.par, hole.yardage) == (par, yards)
        assert hole.pin_position[0] == pytest.approx(yards * 0.9144)
        assert renderer._trajectories == []
        items = renderer._gl_view.items
        assert len(items) == 4  # tee, fairway, green, pin
        assert initial_flight not in items
        fairway = items[1].opts["meshdata"].vertexes()
        assert np.max(fairway[..., 0]) == pytest.approx(yards * 0.9144, rel=1e-6)
        np.testing.assert_allclose(items[-1].pos[:, 0], yards * 0.9144, rtol=1e-6)
    _select(window, Qt.Key.Key_Home, "Driving Range")
    assert isinstance(renderer._environment, gui.DrivingRange)
    assert len(renderer._gl_view.items) == 7


def test_switching_clears_added_flights_without_leaving_stale_items(
    window: gui.EnvironmentWindow,
) -> None:
    renderer = window.renderer
    points = np.array([[0.0, 0.0, 0.0], [50.0, 2.0, 20.0], [100.0, 4.0, 0.0]])
    renderer.add_trajectory(points)
    renderer.add_trajectory(points, color=(1.0, 0.0, 0.0, 1.0))
    flights = tuple(renderer._trajectories)
    assert len(flights) == 3
    for _ in range(2):
        _select(window, Qt.Key.Key_End, "Par 4 (400y)")
        assert renderer._trajectories == []
        assert all(flight not in renderer._gl_view.items for flight in flights)
        _select(window, Qt.Key.Key_Home, "Driving Range")
        assert len(renderer._gl_view.items) == 7
    window.close()
    QApplication.processEvents()
    assert not window.isVisible()
