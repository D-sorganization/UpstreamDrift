"""Headless-safe Qt/matplotlib teardown for the unit-test gate (#14310).

Pytest-xdist reuses worker processes across GUI and non-GUI tests. A widget
destroyed while a matplotlib ``draw_idle`` callback is still queued, or a
``mujoco.Renderer`` collected without an explicit ``close()``, can raise during
a later test's event processing and destabilize the merge queue.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "close_matplotlib_qt_canvases",
    "headless_test_teardown",
    "settle_qt_event_loop",
    "stop_matplotlib_animations",
]


def settle_qt_event_loop(*, passes: int = 5) -> None:
    """Pump the Qt event loop while a ``QApplication`` exists."""
    if "PyQt6.QtWidgets" not in sys.modules:
        return
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError):
        return
    instance = QApplication.instance()
    if instance is None:
        return
    for _ in range(max(1, passes)):
        instance.processEvents()


def _defuse_qtagg_canvas(canvas: Any) -> None:
    """Clear matplotlib idle-draw state on a QtAgg canvas without touching C++."""
    for attr in ("_draw_pending", "_is_drawing"):
        with _suppress_all():
            setattr(canvas, attr, False)


def _close_embedded_mpl_canvases() -> None:
    """Close embedded :class:`~src.shared.python.plotting.mpl_canvas.MplCanvas` widgets.

    These figures are absent from matplotlib's ``Gcf`` registry, but the
    ``WeakSet`` registry tracks them without a per-test ``gc.get_objects()``.
    """
    if "src.shared.python.plotting.mpl_canvas" not in sys.modules:
        return
    try:
        from src.shared.python.plotting.mpl_canvas import iter_live_mpl_canvases
    except (ImportError, OSError):
        return
    for canvas in iter_live_mpl_canvases():
        close_canvas = getattr(canvas, "close_canvas", None)
        if not callable(close_canvas):
            continue
        try:
            close_canvas()
        except Exception:  # noqa: BLE001 - best-effort teardown
            logger.debug("embedded close_canvas failed", exc_info=True)


def close_matplotlib_qt_canvases() -> None:
    """Defuse idle draws and close every managed matplotlib figure."""
    mpl = sys.modules.get("matplotlib")
    if mpl is None:
        return
    try:
        from matplotlib._pylab_helpers import Gcf
        import matplotlib.pyplot as plt
    except (ImportError, AttributeError):
        return

    for manager in list(Gcf.get_all_fig_managers()):
        canvas = getattr(manager, "canvas", None)
        if canvas is None:
            continue
        close_canvas = getattr(canvas, "close_canvas", None)
        if callable(close_canvas):
            try:
                close_canvas()
            except Exception:  # noqa: BLE001 - best-effort teardown
                logger.debug("close_canvas failed", exc_info=True)
            continue
        _defuse_qtagg_canvas(canvas)

    _close_embedded_mpl_canvases()

    try:
        plt.close("all")
    except Exception:  # noqa: BLE001
        logger.debug("plt.close failed", exc_info=True)


def stop_matplotlib_animations() -> None:
    """Stop timers on figures still registered with pyplot."""
    if "matplotlib.pyplot" not in sys.modules:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for num in list(plt.get_fignums()):
        with _suppress_all():
            plt.close(num)


def headless_test_teardown() -> None:
    """Run after each test on xdist workers to drain GUI/GL side effects."""
    if "matplotlib" not in sys.modules and "PyQt6.QtWidgets" not in sys.modules:
        return
    stop_matplotlib_animations()
    close_matplotlib_qt_canvases()
    settle_qt_event_loop()


class _suppress_all:
    """Context manager that swallows all exceptions."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> bool:
        return True
