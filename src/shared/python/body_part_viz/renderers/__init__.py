"""Renderer implementations sub-package.

Concrete :class:`~body_part_viz.contracts.ShapeRenderer` backends
(matplotlib, pyqtgraph, ...). Additional backends land in follow-up
issues of EPIC #4755.
"""

from __future__ import annotations

from .matplotlib_renderer import MatplotlibRenderer
from .projective_renderer import (
    SurfaceLayer,
    SurfaceMesh,
    composite_surface,
    render_surface_layer,
    validate_surface_size,
    validate_surface_source,
)

__all__ = [
    "MatplotlibRenderer",
    "SurfaceLayer",
    "SurfaceMesh",
    "composite_surface",
    "render_surface_layer",
    "validate_surface_size",
    "validate_surface_source",
]
