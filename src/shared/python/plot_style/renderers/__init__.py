"""Renderer implementations sub-package.

Concrete :class:`~src.shared.python.plot_style.contracts.MarkerRenderer`
backends. Currently bundled:

* :class:`MatplotlibMarkerRenderer` — 2D ``Axes`` and 3D ``Axes3D``.
* :class:`PyQtGLMarkerRenderer` — pyqtgraph / OpenGL 3D backend, gated
  behind the optional ``body-part-viz-gl`` extra. Importing the symbol
  directly from this package will raise ``ImportError`` if the extra is
  not installed; that import is lazy so ``import plot_style`` itself
  stays cheap and dependency-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .matplotlib import MatplotlibMarkerRenderer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pyqtgl import PyQtGLMarkerRenderer

__all__ = ["MatplotlibMarkerRenderer", "PyQtGLMarkerRenderer"]


def __getattr__(name: str) -> Any:
    """Lazy attribute access for optional renderers."""
    if name == "PyQtGLMarkerRenderer":
        from .pyqtgl import (  # noqa: PLC0415 - intentional lazy import
            PyQtGLMarkerRenderer,
        )

        return PyQtGLMarkerRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
