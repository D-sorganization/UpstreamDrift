"""Tests for the golf environment 3D visualization GUI."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from src.tools.golf_environment import gui as ge_gui
from src.tools.golf_environment.gui import (
    CourseHole,
    DrivingRange,
    EnvironmentRenderer,
    EnvironmentWindow,
    get_dockable_ui,
)

_APP: QApplication | None = None


def _ensure_qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app
    return app


@pytest.fixture(autouse=True)
def _qapp() -> QApplication:
    return _ensure_qapp()


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


def test_driving_range_defaults() -> None:
    dr = DrivingRange()
    assert dr.name == "Standard Driving Range"
    assert dr.length == 350.0
    assert dr.width == 100.0
    assert dr.markers == [50, 100, 150, 200, 250, 300]


def test_driving_range_independent_marker_lists() -> None:
    """default_factory must yield independent lists per instance."""
    a = DrivingRange()
    b = DrivingRange()
    a.markers.append(999)
    assert 999 not in b.markers


def test_driving_range_custom_values() -> None:
    dr = DrivingRange(name="Custom", width=80.0, length=300.0, markers=[100, 200])
    assert dr.name == "Custom"
    assert dr.markers == [100, 200]


def test_course_hole_defaults() -> None:
    h = CourseHole(name="Hole 1", par=4, yardage=400)
    assert h.par == 4
    assert h.yardage == 400
    assert h.tee_position == (0.0, 0.0, 0.0)
    assert h.pin_position == (100.0, 0.0, 0.0)
    assert h.fairway_width == 40.0
    assert h.green_radius == 15.0


def test_course_hole_custom_positions() -> None:
    h = CourseHole(
        name="X",
        par=3,
        yardage=150,
        tee_position=(1.0, 2.0, 3.0),
        pin_position=(150.0, 5.0, 0.0),
        fairway_width=30.0,
        green_radius=10.0,
    )
    assert h.tee_position == (1.0, 2.0, 3.0)
    assert h.pin_position == (150.0, 5.0, 0.0)
    assert h.fairway_width == 30.0
    assert h.green_radius == 10.0


# ---------------------------------------------------------------------------
# EnvironmentRenderer
# ---------------------------------------------------------------------------


def test_renderer_initialises_with_driving_range_default() -> None:
    r = EnvironmentRenderer()
    assert isinstance(r._environment, DrivingRange)
    assert r._trajectories == []


def test_renderer_gl_view_configured() -> None:
    r = EnvironmentRenderer()
    if not ge_gui.PYQTGRAPH_AVAILABLE:
        pytest.skip("pyqtgraph not available")
    assert r._gl_view is not None
    assert r._gl_view.opts["distance"] == 150
    assert r._gl_view.opts["elevation"] == 15
    assert r._gl_view.opts["azimuth"] == 90


def test_renderer_renders_driving_range_items() -> None:
    if not ge_gui.PYQTGRAPH_AVAILABLE:
        pytest.skip("pyqtgraph not available")
    r = EnvironmentRenderer()
    # After init, render_environment was called once with DrivingRange.
    # 1 ground mesh + 6 markers = 7 items
    items = r._gl_view.items
    assert len(items) == 1 + len(DrivingRange().markers)


def test_renderer_switches_to_course_hole() -> None:
    if not ge_gui.PYQTGRAPH_AVAILABLE:
        pytest.skip("pyqtgraph not available")
    r = EnvironmentRenderer()
    hole = CourseHole(name="H", par=4, yardage=400, pin_position=(200.0, 0.0, 0.0))
    r.set_environment(hole)
    assert r._environment is hole
    # tee + fairway + green + pin = 4 items
    assert len(r._gl_view.items) == 4


def test_renderer_set_environment_back_to_range() -> None:
    if not ge_gui.PYQTGRAPH_AVAILABLE:
        pytest.skip("pyqtgraph not available")
    r = EnvironmentRenderer()
    r.set_environment(CourseHole(name="H", par=3, yardage=150))
    r.set_environment(DrivingRange(markers=[100, 200]))
    # 1 ground + 2 markers
    assert len(r._gl_view.items) == 3


def test_renderer_add_trajectory_appends() -> None:
    if not ge_gui.PYQTGRAPH_AVAILABLE:
        pytest.skip("pyqtgraph not available")
    r = EnvironmentRenderer()
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 5.0], [20.0, 0.0, 0.0]])
    r.add_trajectory(pts)
    assert len(r._trajectories) == 1
    r.add_trajectory(pts, color=(1.0, 0.0, 0.0, 1.0))
    assert len(r._trajectories) == 2


def test_renderer_add_trajectory_ignores_empty() -> None:
    r = EnvironmentRenderer()
    r.add_trajectory(np.zeros((0, 3)))
    assert r._trajectories == []


def test_renderer_clear_trajectories_removes_all() -> None:
    if not ge_gui.PYQTGRAPH_AVAILABLE:
        pytest.skip("pyqtgraph not available")
    r = EnvironmentRenderer()
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    r.add_trajectory(pts)
    r.add_trajectory(pts)
    assert len(r._trajectories) == 2
    r.clear_trajectories()
    assert r._trajectories == []


def test_create_rect_shape_and_corners() -> None:
    r = EnvironmentRenderer()
    rect = r._create_rect(0.0, 0.0, 4.0, 2.0)
    assert rect.shape == (6, 3)
    # All z = 0
    assert np.all(rect[:, 2] == 0)
    # Corners present
    xs = set(rect[:, 0].tolist())
    ys = set(rect[:, 1].tolist())
    assert xs == {0.0, 4.0}
    assert ys == {0.0, 2.0}


def test_create_circle_shape_and_z() -> None:
    r = EnvironmentRenderer()
    circ = r._create_circle(0.0, 0.0, 5.0, segments=8)
    # (segments - 1) triangles, 3 verts each
    assert circ.shape == (3 * 7, 3)
    # All z = 0.05 (raised slightly to render above ground)
    assert np.allclose(circ[:, 2], 0.05)
    # All vertices within radius (center vertices have r=0, edge points have r=5)
    radii = np.sqrt(circ[:, 0] ** 2 + circ[:, 1] ** 2)
    assert radii.max() <= 5.0 + 1e-9


def test_create_circle_centered_offset() -> None:
    r = EnvironmentRenderer()
    circ = r._create_circle(10.0, -3.0, 2.0, segments=16)
    # Distance from (10,-3) should be either 0 (center) or 2 (edge)
    d = np.sqrt((circ[:, 0] - 10.0) ** 2 + (circ[:, 1] + 3.0) ** 2)
    assert d.max() <= 2.0 + 1e-9


# ---------------------------------------------------------------------------
# EnvironmentRenderer when pyqtgraph is unavailable
# ---------------------------------------------------------------------------


def test_renderer_without_pyqtgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ge_gui, "PYQTGRAPH_AVAILABLE", False)
    r = EnvironmentRenderer()
    assert r._gl_view is None
    # These should be no-ops, not raise.
    r.render_environment()
    r.set_environment(DrivingRange())
    r.add_trajectory(np.array([[0.0, 0.0, 0.0]]))
    r.clear_trajectories()
    assert r._trajectories == []


# ---------------------------------------------------------------------------
# EnvironmentWindow
# ---------------------------------------------------------------------------


def test_environment_window_initialises_with_demo_trajectory() -> None:
    if not ge_gui.PYQTGRAPH_AVAILABLE:
        pytest.skip("pyqtgraph not available")
    w = EnvironmentWindow()
    assert w.windowTitle() == "Golf Environment Viewer"
    assert w.env_combo.count() == 3
    assert len(w.renderer._trajectories) == 1


def test_environment_window_switches_environments() -> None:
    if not ge_gui.PYQTGRAPH_AVAILABLE:
        pytest.skip("pyqtgraph not available")
    w = EnvironmentWindow()

    w._on_env_changed("Par 3 (150y)")
    assert isinstance(w.renderer._environment, CourseHole)
    assert w.renderer._environment.par == 3
    # clear_trajectories was called -> the demo trajectory is gone
    assert w.renderer._trajectories == []

    w._on_env_changed("Par 4 (400y)")
    assert isinstance(w.renderer._environment, CourseHole)
    assert w.renderer._environment.par == 4

    w._on_env_changed("Driving Range")
    assert isinstance(w.renderer._environment, DrivingRange)


def test_environment_window_unknown_label_is_noop() -> None:
    if not ge_gui.PYQTGRAPH_AVAILABLE:
        pytest.skip("pyqtgraph not available")
    w = EnvironmentWindow()
    prev_env = w.renderer._environment
    w._on_env_changed("not a real label")
    # environment unchanged, but trajectories cleared
    assert w.renderer._environment is prev_env


def test_get_dockable_ui_returns_environment_window() -> None:
    w = get_dockable_ui()
    assert isinstance(w, EnvironmentWindow)
