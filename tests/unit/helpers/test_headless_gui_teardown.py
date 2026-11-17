"""Unit tests for :mod:`tests.helpers.headless_gui_teardown`."""

from __future__ import annotations

import time

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_headless_teardown_without_qt_or_matplotlib_does_not_raise() -> None:
    """Teardown is a no-op when neither Qt nor matplotlib was imported."""
    from tests.helpers.headless_gui_teardown import headless_test_teardown

    headless_test_teardown()


def test_defuse_qtagg_canvas_clears_pending_idle_draw() -> None:
    """The QtAgg idle-draw flag must clear without touching a deleted C++ object."""
    pytest.importorskip("PyQt6")
    from src.shared.python.plotting.mpl_canvas import MplCanvas

    from tests.helpers.headless_gui_teardown import _defuse_qtagg_canvas

    canvas = MplCanvas(width=4, height=3, dpi=80)
    canvas.draw_idle()
    assert canvas._draw_pending is True
    _defuse_qtagg_canvas(canvas)
    assert canvas._draw_pending is False
    canvas.close_canvas()


def test_stop_matplotlib_animations_halts_pyplot_funcanimation() -> None:
    """Pyplot figures must stop advancing after ``stop_matplotlib_animations``."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, _ax = plt.subplots()
    ticks = {"n": 0}

    def _tick(_frame: int) -> list[object]:
        ticks["n"] += 1
        return []

    anim = FuncAnimation(
        fig,
        _tick,
        interval=30,
        save_count=50,
        blit=False,
        cache_frame_data=False,
    )
    try:
        assert isinstance(fig._animated, bool), (
            "precondition: _animated is not the anim"
        )
        fig.canvas.draw()
        assert ticks["n"] >= 1, "precondition: timer must be running"

        from tests.helpers.headless_gui_teardown import stop_matplotlib_animations

        stop_matplotlib_animations()
        count_after_stop = ticks["n"]
        time.sleep(0.15)
        stop_matplotlib_animations()
        assert ticks["n"] == count_after_stop
    finally:
        plt.close(fig)


def test_headless_teardown_stops_embedded_mplcanvas_animation() -> None:
    """Embedded canvases are absent from ``Gcf`` but their timers must still halt."""
    pytest.importorskip("PyQt6")
    pytest.importorskip("matplotlib")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    import matplotlib

    matplotlib.use("QtAgg", force=True)
    from matplotlib._pylab_helpers import Gcf
    from matplotlib.animation import FuncAnimation
    from src.shared.python.plotting.mpl_canvas import MplCanvas

    canvas = MplCanvas(width=4, height=3, dpi=80)
    ticks = {"n": 0}

    def _tick(_frame: int) -> list[object]:
        ticks["n"] += 1
        return []

    anim = FuncAnimation(
        canvas.fig,
        _tick,
        interval=30,
        save_count=50,
        blit=False,
        cache_frame_data=False,
    )
    try:
        assert not list(Gcf.get_all_fig_managers()), "precondition: embed is not in Gcf"
        canvas.draw()
        assert ticks["n"] >= 1, "precondition: animation timer is live"

        from tests.helpers.headless_gui_teardown import headless_test_teardown

        headless_test_teardown()
        count_after_teardown = ticks["n"]
        app.processEvents()
        time.sleep(0.15)
        app.processEvents()
        assert ticks["n"] == count_after_teardown
    finally:
        if not canvas._canvas_closed:
            canvas.close_canvas()
