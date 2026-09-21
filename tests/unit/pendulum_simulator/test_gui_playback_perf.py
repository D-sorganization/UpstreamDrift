"""Playback-path regression tests for the pendulum GUI (#8929).

The matrix panel must not re-evaluate dynamics inside ``paintEvent`` and the
tip trail must not re-spline / re-allocate a pen per segment on every repaint.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QFont

from src.shared.python.pendulum_simulator.gui.base_pendulum_widget import (
    BasePendulumWidget,
)
from src.shared.python.pendulum_simulator.gui.matrix_widget import MatrixWidget
from src.shared.python.pendulum_simulator.gui.pendulum_widget import PendulumWidget

pytestmark = pytest.mark.unit


class _CountingResult:
    """Minimal double-pendulum result stub that counts dynamics evaluations."""

    n_steps = 20

    def __init__(self) -> None:
        self.calls: dict[str, int] = {
            "mass": 0,
            "tau": 0,
            "gravity": 0,
            "coriolis": 0,
            "energy": 0,
        }

    def mass_matrix_at(self, idx: int) -> dict:
        self.calls["mass"] += 1
        return {"M11": 2.0 + idx, "M12": 0.5, "M21": 0.5, "M22": 1.0}

    def torques_at(self, idx: int) -> tuple[float, float]:
        self.calls["tau"] += 1
        return (1.0, 2.0)

    def gravity_at(self, idx: int) -> np.ndarray:
        self.calls["gravity"] += 1
        return np.array([0.1, 0.2])

    def coriolis_at(self, idx: int) -> np.ndarray:
        self.calls["coriolis"] += 1
        return np.array([0.3, 0.4])

    def energy_at(self, idx: int) -> dict:
        self.calls["energy"] += 1
        return {"kinetic": 1.0, "potential": 2.0, "total": 3.0}


def _total_calls(result: _CountingResult) -> int:
    return sum(result.calls.values())


def test_matrix_paint_does_not_reevaluate_dynamics(qtbot) -> None:
    widget = MatrixWidget()
    qtbot.addWidget(widget)
    widget.resize(300, 600)
    result = _CountingResult()

    widget.set_simulation(result)
    after_set = _total_calls(result)
    assert result.calls["mass"] == 1

    for _ in range(3):
        widget.grab()  # forces a full paintEvent
    assert _total_calls(result) == after_set

    widget.set_frame(0)  # unchanged frame short-circuits
    assert _total_calls(result) == after_set

    widget.set_frame(5)
    assert result.calls["mass"] == 2
    assert result.calls["energy"] == 2
    widget.grab()
    assert result.calls["mass"] == 2


def test_matrix_skips_undrawable_coriolis_matrix(qtbot) -> None:
    """An (n, n) Coriolis matrix was never rendered; stop evaluating it."""

    class _MatrixCoriolisResult(_CountingResult):
        def coriolis_at(self, idx: int) -> np.ndarray:
            self.calls["coriolis"] += 1
            return np.eye(2)

    widget = MatrixWidget()
    qtbot.addWidget(widget)
    widget.resize(300, 600)
    result = _MatrixCoriolisResult()

    widget.set_simulation(result)
    widget.set_frame(3)
    widget.set_frame(7)
    widget.grab()
    assert result.calls["coriolis"] == 1
    assert widget._current_frame_data()["coriolis"] is None
    assert result.calls["mass"] == 3


def test_matrix_fonts_are_cached(qtbot) -> None:
    widget = MatrixWidget()
    qtbot.addWidget(widget)
    font = widget._font("Monospace", 10, bold=True)
    assert isinstance(font, QFont)
    assert widget._font("Monospace", 10, bold=True) is font
    assert widget._font("Monospace", 10) is not font


def _tips(n: int) -> np.ndarray:
    theta = np.linspace(0.0, 3.0, n)
    return np.column_stack([np.sin(theta), -np.cos(theta)])


def test_trail_window_slices_precomputed_spline(qtbot) -> None:
    widget = PendulumWidget()
    qtbot.addWidget(widget)
    tips = _tips(50)
    sub = widget.SPLINE_SUBDIV

    widget._set_trail_source(tips)
    assert widget._trail_spline is not None
    assert len(widget._trail_spline) == (len(tips) - 1) * sub + 1

    widget._set_trail_frame(10)
    assert list(widget._trail) == [tuple(p) for p in tips[:11].tolist()]
    smooth = widget._smoothed_trail()
    assert len(smooth) == 10 * sub + 1
    np.testing.assert_allclose(smooth[0], tips[0])
    np.testing.assert_allclose(smooth[-1], tips[10])

    # Forward playback extends the window; a full window slides.
    widget.TRAIL_LENGTH = 8
    widget._set_trail_frame(11)
    assert list(widget._trail) == [tuple(p) for p in tips[4:12].tolist()]
    smooth = widget._smoothed_trail()
    assert len(smooth) == 7 * sub + 1
    np.testing.assert_allclose(smooth[0], tips[4])
    np.testing.assert_allclose(smooth[-1], tips[11])


def test_trail_paint_uses_bucketed_pens_and_no_respline(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = PendulumWidget()
    qtbot.addWidget(widget)
    widget.resize(400, 400)
    widget._set_trail_source(_tips(400))
    widget._set_trail_frame(399)
    assert len(widget._trail) == widget.TRAIL_LENGTH

    spline_calls: list[int] = []
    original = BasePendulumWidget._catmull_rom_smooth

    def _spy(points: list, n_sub: int = 4) -> list:
        spline_calls.append(len(points))
        return original(points, n_sub)

    monkeypatch.setattr(BasePendulumWidget, "_catmull_rom_smooth", staticmethod(_spy))

    painter = MagicMock()
    widget._draw_trail(painter)
    widget._draw_trail(painter)

    assert spline_calls == []
    assert painter.setPen.call_count == 2 * widget.TRAIL_ALPHA_BUCKETS
    assert painter.drawPolyline.call_count == 2 * widget.TRAIL_ALPHA_BUCKETS
    assert painter.drawLine.call_count == 0


def test_trail_fallback_caches_smoothing_by_contents(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trail populated directly (no tip cache) is smoothed once, not per paint."""
    widget = PendulumWidget()
    qtbot.addWidget(widget)
    widget.resize(400, 400)
    for x, y in _tips(12).tolist():
        widget._trail.append((x, y))

    spline_calls: list[int] = []
    original = BasePendulumWidget._catmull_rom_smooth

    def _spy(points: list, n_sub: int = 4) -> list:
        spline_calls.append(len(points))
        return original(points, n_sub)

    monkeypatch.setattr(BasePendulumWidget, "_catmull_rom_smooth", staticmethod(_spy))

    painter = MagicMock()
    widget._draw_trail(painter)
    widget._draw_trail(painter)
    assert spline_calls == [12]

    widget._trail.append((0.0, 0.0))
    widget._draw_trail(painter)
    assert spline_calls == [12, 13]
