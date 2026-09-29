"""Renderer implementations sub-package.

Concrete :class:`~body_part_viz.contracts.ShapeRenderer` backends
(matplotlib, pyqtgraph, ...). Additional backends land in follow-up
issues of EPIC #4755.
"""

from __future__ import annotations

from .matplotlib_renderer import MatplotlibRenderer

__all__ = ["MatplotlibRenderer"]
