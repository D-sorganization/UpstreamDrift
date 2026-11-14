"""Headless-safe Qt/matplotlib teardown for the unit-test gate (#14310).

Pytest-xdist reuses worker processes across GUI and non-GUI tests. A widget
destroyed while a matplotlib ``draw_idle`` callback is still queued, or a
``mujoco.Renderer`` collected without an explicit ``close()``, can raise during
a later test's event processing and destabilize the merge queue.
"""

from __future__ import annotations

import gc
import logging
import sys
from typing import Any, Iterator

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


def _iter_qt_figure_canvases() -> Iterator[Any]:
    """Yield QtAgg figure canvases, including embeds not registered in ``Gcf``."""
    mpl = sys.modules.get("matplotlib")
    if mpl is None:
        return
    try:
        from matplotlib._pylab_helpers import Gcf
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    except (ImportError, AttributeError):
        return

    seen: set[int] = set()
    for manager in list(Gcf.get_all_fig_managers()):
        canvas = getattr(manager, "canvas", None)
        if canvas is not None and id(canvas) not in seen:
            seen.add(id(canvas))
            yield canvas

    for obj in gc.get_objects():
        if isinstance(obj, FigureCanvasQTAgg) and id(obj) not in seen:
            seen.add(id(obj))
            yield obj


def close_matplotlib_qt_canvases() -> None:
    """Defuse idle draws and close every QtAgg canvas (``Gcf`` and embedded)."""
    mpl = sys.modules.get("matplotlib")
    if mpl is None:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for canvas in _iter_qt_figure_canvases():
        close_canvas = getattr(canvas, "close_canvas", None)
        if callable(close_canvas):
            try:
                close_canvas()
            except Exception:  # noqa: BLE001 - best-effort teardown
                logger.debug("close_canvas failed", exc_info=True)
            continue
        _defuse_qtagg_canvas(canvas)

    try:
        plt.close("all")
    except Exception:  # noqa: BLE001
        logger.debug("plt.close failed", exc_info=True)


def _defuse_qtagg_canvas(canvas: Any) -> None:
    """Clear matplotlib idle-draw state on a QtAgg canvas without touching C++."""
    for attr in ("_draw_pending", "_is_drawing"):
        with _suppress_all():
            setattr(canvas, attr, False)


def _iter_live_animations() -> Iterator[Any]:
    """Yield live :class:`~matplotlib.animation.Animation` instances via the GC."""
    if "matplotlib.animation" not in sys.modules:
        return
    try:
        from matplotlib.animation import Animation
    except ImportError:
        return
    for obj in gc.get_objects():
        if isinstance(obj, Animation):
            yield obj


def _halt_animation(anim: Any) -> None:
    """Stop one animation's timer and disconnect its canvas callbacks."""
    for method_name in ("pause", "_stop"):
        method = getattr(anim, method_name, None)
        if callable(method):
            with _suppress_all():
                method()
    event_source = getattr(anim, "event_source", None)
    if event_source is not None:
        with _suppress_all():
            event_source.stop()


def stop_matplotlib_animations() -> None:
    """Stop running :class:`~matplotlib.animation.Animation` timers."""
    for anim in _iter_live_animations():
        _halt_animation(anim)


def headless_test_teardown() -> None:
    """Run after each test on xdist workers to drain GUI/GL side effects."""
    stop_matplotlib_animations()
    close_matplotlib_qt_canvases()
    settle_qt_event_loop()


class _suppress_all:
    """Context manager that swallows all exceptions."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> bool:
        return True
