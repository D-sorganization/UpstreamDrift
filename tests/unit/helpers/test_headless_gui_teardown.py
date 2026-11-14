"""Unit tests for :mod:`tests.helpers.headless_gui_teardown`."""

from __future__ import annotations

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
