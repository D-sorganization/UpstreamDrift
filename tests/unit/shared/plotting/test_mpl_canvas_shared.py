"""Contract tests for the consolidated :mod:`src.shared.python.plotting.mpl_canvas`.

Covers the two properties that motivated the consolidation (#9474):

* a queued ``draw_idle`` must be defused before the C++ canvas dies, and
* the pyplot figure registry must be released via ``plt.close``.

Both live in one place now, so these tests pin the behaviour for every
consumer that previously carried its own ``MplCanvas`` copy.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

pytest.importorskip("PyQt6", reason="Qt canvas contract requires PyQt6")
pytest.importorskip("matplotlib", reason="Qt canvas contract requires matplotlib")


@pytest.fixture
def canvas():
    """Yield a live shared canvas, always torn down."""
    from src.shared.python.plotting.mpl_canvas import MplCanvas

    instance = MplCanvas(width=4, height=3, dpi=80)
    try:
        yield instance
    finally:
        instance.close_canvas()


def test_rejects_non_positive_geometry() -> None:
    """Precondition: figure geometry must be strictly positive."""
    from src.shared.python.contracts import PreconditionError
    from src.shared.python.plotting.mpl_canvas import MplCanvas

    for kwargs in ({"width": 0}, {"height": -1}, {"dpi": 0}):
        with pytest.raises((PreconditionError, ValueError)):
            MplCanvas(**kwargs)


def test_exposes_fig_attribute(canvas) -> None:
    """``fig`` is the documented handle used by embed adapters."""
    from matplotlib.figure import Figure

    assert isinstance(canvas.fig, Figure)
    assert canvas.figure is canvas.fig


def test_close_canvas_cancels_pending_draw(canvas) -> None:
    """A queued ``draw_idle`` must not survive teardown (#9474)."""
    canvas.draw_idle()
    assert canvas._draw_pending is True, "precondition: a draw is queued"

    canvas.close_canvas()

    assert canvas._draw_pending is False


def test_close_canvas_releases_pyplot_figure(canvas, monkeypatch) -> None:
    """Teardown hands the figure to ``plt.close`` exactly once."""
    import matplotlib.pyplot as plt

    closed: list[object] = []
    monkeypatch.setattr(plt, "close", closed.append)

    fig = canvas.fig
    canvas.close_canvas()
    canvas.close_canvas()

    assert closed == [fig], "figure released once, and only once"


def test_close_canvas_is_idempotent(canvas) -> None:
    """``cleanup`` contract requires repeat calls to be harmless."""
    canvas.close_canvas()
    canvas.close_canvas()
    canvas.close_canvas()


def test_close_canvas_survives_deleted_cpp_object(canvas) -> None:
    """Teardown must not raise once Qt has destroyed the C++ half."""
    from PyQt6 import sip

    canvas.draw_idle()
    sip.delete(canvas)

    canvas.close_canvas()


def test_draw_idle_after_close_is_a_no_op(canvas) -> None:
    """The defused callback stays defused if invoked by a live timer."""
    canvas.draw_idle()
    canvas.close_canvas()

    canvas._draw_idle()

    assert canvas._draw_pending is False


def test_parent_is_accepted_positionally() -> None:
    """Engine call sites pass ``parent`` positionally; keep that working."""
    from PyQt6.QtWidgets import QWidget

    from src.shared.python.plotting.mpl_canvas import MplCanvas

    parent = QWidget()
    instance = MplCanvas(parent, width=5, height=4, dpi=100)
    try:
        assert instance.parent() is parent
    finally:
        instance.close_canvas()


def test_add_subplot_and_clear_axes(canvas) -> None:
    """The engine-local helpers survive consolidation."""
    from matplotlib.axes import Axes

    axes = canvas.add_subplot(111)
    assert isinstance(axes, Axes)
    assert canvas.fig.axes

    canvas.clear_axes()
    assert not canvas.fig.axes
