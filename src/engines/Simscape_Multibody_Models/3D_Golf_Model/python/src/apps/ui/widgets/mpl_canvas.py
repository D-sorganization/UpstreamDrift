"""Reusable Matplotlib canvas widget for embedding plots in Qt GUIs.

The implementation is the shared canonical canvas
(:mod:`src.shared.python.plotting.mpl_canvas`); this module only pins the
geometry defaults the 3D Golf Model tabs were written against. Keeping the
subclass preserves the ``from ..widgets.mpl_canvas import MplCanvas``
import used by the five plot tabs while removing the duplicated body
(#9474).
"""

from __future__ import annotations

from typing import Any

from src.shared.python.plotting.mpl_canvas import MplCanvas as _SharedMplCanvas

__all__ = ["MplCanvas"]


class MplCanvas(_SharedMplCanvas):
    """Matplotlib canvas widget for embedding plots in Qt.

    Behaviour is inherited from the shared canvas, including
    ``close_canvas()``, ``add_subplot()`` and ``clear_axes()``. Only the
    default figure geometry differs from the shared default.
    """

    def __init__(
        self,
        parent: Any | None = None,
        width: float = 5.0,
        height: float = 4.0,
        dpi: int = 100,
    ) -> None:
        """Initialize the matplotlib canvas with specified dimensions."""
        super().__init__(parent=parent, width=width, height=height, dpi=dpi)
