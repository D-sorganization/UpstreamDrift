"""Marker-shape primitive sub-package.

Each module exports a single class implementing the
:class:`plot_style.contracts.MarkerShapeRenderer` Protocol. The
:data:`SHAPE_REGISTRY` mapping links each :class:`MarkerShape` enum
member to the default factory for its primitive.
"""

from __future__ import annotations

from collections.abc import Callable

from ..contracts import MarkerShapeRenderer
from ..markers import MarkerShape
from ._cross_marker import CrossMarker
from ._cube_marker import CubeMarker
from ._custom_mesh_marker import CustomMeshMarker
from ._diamond_marker import DiamondMarker
from ._sphere_marker import SphereMarker
from ._star_marker import StarMarker

__all__ = [
    "SHAPE_REGISTRY",
    "CrossMarker",
    "CubeMarker",
    "CustomMeshMarker",
    "DiamondMarker",
    "SphereMarker",
    "StarMarker",
    "default_marker_for",
]


# Default factories keyed by MarkerShape. CUSTOM_MESH and PLUS / POINT
# require additional inputs and so are not in the default registry.
SHAPE_REGISTRY: dict[MarkerShape, Callable[[], MarkerShapeRenderer]] = {
    MarkerShape.SPHERE: SphereMarker,
    MarkerShape.CUBE: CubeMarker,
    MarkerShape.CROSS: CrossMarker,
    MarkerShape.STAR: StarMarker,
    MarkerShape.DIAMOND: DiamondMarker,
}


def default_marker_for(shape: MarkerShape) -> MarkerShapeRenderer:
    """Return a default-configured marker for ``shape``.

    Raises
    ------
    KeyError
        If ``shape`` is not in :data:`SHAPE_REGISTRY` (e.g.
        :attr:`MarkerShape.CUSTOM_MESH`, which requires explicit
        construction with a :class:`CustomMeshSpec` or asset path).
    """
    if not isinstance(shape, MarkerShape):
        raise TypeError(
            f"shape must be MarkerShape; got {type(shape).__name__}"
        )
    if shape not in SHAPE_REGISTRY:
        raise KeyError(
            f"no default factory for {shape!r}; construct the marker "
            "directly (e.g. CustomMeshMarker for CUSTOM_MESH)"
        )
    return SHAPE_REGISTRY[shape]()
