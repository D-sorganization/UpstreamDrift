"""Canonical Qt matplotlib canvas with safe teardown.

Single definition of ``MplCanvas`` for UpstreamDrift-owned code. Before
#9474 the class was copy-pasted across the repository, so a fix to canvas
teardown had to be applied in every copy or silently drift.

Teardown (``close_canvas``) addresses two distinct leaks:

1. **A queued idle draw outliving the C++ widget.** ``draw_idle`` posts
   ``QTimer.singleShot(0, self._draw_idle)``. When Qt destroys the
   underlying C++ object before that timer fires, ``_draw_idle`` reaches
   ``self.height()`` on a dead wrapper and raises ``RuntimeError``. The
   timer cannot be cancelled (no handle is retained), but matplotlib's
   ``_draw_idle`` early-returns when ``_draw_pending`` is false *before*
   touching C++ state, so clearing the flag defuses the pending callback.
2. **A figure retained by the pyplot registry.** ``plt.close(fig)`` drops
   the figure so the Agg buffer and axes are collectable.

Ownership: UpstreamDrift. This module has no counterpart in the Tools
repository, so it is edited here (see
``docs/shared_tools/divergence_inventory.v1.json``).
"""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.shared.python.contracts import require

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:  # pragma: no cover - exercised only on old matplotlib
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    except ImportError as exc:
        # Re-raise so callers (e.g. plotting.base) can fall back to a
        # headless stub rather than crashing at import time.
        raise ImportError("Qt backend for matplotlib not available") from exc

__all__ = ["MplCanvas"]


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for embedding in PyQt6, with deterministic teardown.

    The signature accommodates both historical call conventions: plotting
    consumers pass geometry by keyword (``MplCanvas(width=5, height=4)``)
    while engine widgets pass the parent positionally
    (``MplCanvas(self, width=5, height=4, dpi=100)``).
    """

    def __init__(
        self,
        parent: Any | None = None,
        width: float = 8.0,
        height: float = 6.0,
        dpi: int = 100,
    ) -> None:
        """Initialise the canvas with a figure of the requested geometry.

        Args:
            parent: Optional Qt parent widget.
            width: Figure width in inches; must be positive.
            height: Figure height in inches; must be positive.
            dpi: Dots per inch for rendering; must be positive.

        Raises:
            PreconditionError: If any geometry value is not positive.
        """
        require(
            width is not None and width > 0,
            f"width must be a positive number of inches, got {width!r}",
            value=width,
        )
        require(
            height is not None and height > 0,
            f"height must be a positive number of inches, got {height!r}",
            value=height,
        )
        require(
            dpi is not None and dpi > 0,
            f"dpi must be a positive integer, got {dpi!r}",
            value=dpi,
        )

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self._canvas_closed = False
        if parent is not None:
            self.setParent(parent)

    def add_subplot(self, *args: Any, **kwargs: Any) -> Axes:
        """Add a subplot to the figure and return the new axes."""
        return self.fig.add_subplot(*args, **kwargs)

    def clear_axes(self) -> None:
        """Clear every axes from the figure and repaint.

        Uses a blocking ``draw()`` rather than ``draw_idle()`` to preserve
        the pre-consolidation behaviour of the engine-local canvas, whose
        callers replot immediately after clearing.
        """
        self.fig.clear()
        self.draw()

    def close_canvas(self) -> None:
        """Release the figure and defuse any queued idle draw.

        Idempotent and non-raising: embed adapters call this during host
        shutdown, where a failure must never block teardown, and Qt may
        already have destroyed the underlying C++ object.
        """
        if getattr(self, "_canvas_closed", False):
            return
        self._canvas_closed = True

        # Order matters: defuse the queued callback before releasing the
        # figure, so a timer firing mid-teardown finds nothing to draw.
        # Plain Python attribute assignment - safe on a deleted C++ object.
        self._draw_pending = False
        self._is_drawing = False

        figure = getattr(self, "fig", None)
        if figure is None:
            return
        try:
            import matplotlib.pyplot as plt

            plt.close(figure)
        except Exception:  # pragma: no cover - defensive  # noqa: BLE001
            # Teardown is best-effort; never propagate during shutdown.
            pass
