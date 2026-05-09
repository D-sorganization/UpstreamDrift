"""Renderer implementations sub-package.

Concrete :class:`~src.shared.python.plot_style.contracts.MarkerRenderer`
backends. Currently bundled:

* :class:`MatplotlibMarkerRenderer` — 2D ``Axes`` and 3D ``Axes3D``.
"""

from __future__ import annotations

from .matplotlib import MatplotlibMarkerRenderer

__all__ = ["MatplotlibMarkerRenderer"]
