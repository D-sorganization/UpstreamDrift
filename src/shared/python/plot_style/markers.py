"""Marker shape enum and the :class:`MarkerStyle` dataclass.

A *marker style* captures every visual attribute of a marker except the
position itself (those are supplied per-frame at render time):

* shape — sphere, cube, custom mesh, ...
* size, edge color / width, opacity
* fill color (any :data:`ColorScale` variant)

Custom mesh markers attach a :class:`CustomMeshSpec` describing the
geometry the renderer should draw.

Design-by-Contract
------------------
Both dataclasses are frozen and validate every constraint in
``__post_init__``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from matplotlib.colors import is_color_like

from .colors import ColorScale, DataDrivenColor, PaletteColor, StaticColor

__all__ = ["CustomMeshSpec", "MarkerShape", "MarkerStyle"]


class MarkerShape(str, Enum):
    """Built-in marker shape identifiers.

    ``CUSTOM_MESH`` is paired with a :class:`CustomMeshSpec` on the
    enclosing :class:`MarkerStyle`.
    """

    SPHERE = "sphere"
    CUBE = "cube"
    CROSS = "cross"
    STAR = "star"
    DIAMOND = "diamond"
    PLUS = "plus"
    POINT = "point"
    CUSTOM_MESH = "custom_mesh"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class CustomMeshSpec:
    """Triangular mesh attached to a :class:`MarkerShape.CUSTOM_MESH`.

    Attributes
    ----------
    vertices:
        ``(V, 3)`` ``float`` ndarray of vertex positions in the marker's
        local frame.
    faces:
        ``(F, 3)`` integer ndarray of triangle indices into
        :attr:`vertices`.
    name:
        Non-empty identifier used when persisting / logging.
    """

    name: str
    vertices: np.ndarray = field(repr=False)
    faces: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string; got {self.name!r}")
        if not isinstance(self.vertices, np.ndarray):
            raise TypeError(
                f"vertices must be numpy.ndarray; got {type(self.vertices).__name__}"
            )
        if not isinstance(self.faces, np.ndarray):
            raise TypeError(
                f"faces must be numpy.ndarray; got {type(self.faces).__name__}"
            )
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape (V, 3); got shape={self.vertices.shape}"
            )
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(
                f"faces must have shape (F, 3); got shape={self.faces.shape}"
            )
        if self.vertices.dtype.kind not in ("f", "i", "u"):
            raise TypeError(
                f"vertices dtype must be numeric; got {self.vertices.dtype}"
            )
        if self.faces.dtype.kind not in ("i", "u"):
            raise TypeError(f"faces dtype must be integer; got {self.faces.dtype}")
        if self.faces.size > 0:
            max_idx = int(self.faces.max())
            if max_idx >= self.vertices.shape[0]:
                raise ValueError(
                    "faces index out of bounds for vertices: "
                    f"max index={max_idx}, n_vertices={self.vertices.shape[0]}"
                )
            if int(self.faces.min()) < 0:
                raise ValueError("faces indices must be non-negative")


def _default_fill_color() -> StaticColor:
    """Default :class:`StaticColor` used by :class:`MarkerStyle`."""
    return StaticColor("#1f77b4")


@dataclass(frozen=True)
class MarkerStyle:
    """All visual properties of a marker except its position.

    Attributes
    ----------
    shape:
        Marker shape identifier.
    size_px:
        Diameter in pixels. Strictly positive.
    edge_color:
        Edge color (any matplotlib-recognised color string).
    edge_width:
        Edge stroke width in pixels. Non-negative.
    fill_color:
        Any :data:`ColorScale` variant (static, palette, or data-driven).
    opacity:
        Alpha multiplier in ``[0, 1]``.
    custom_mesh:
        Required iff :attr:`shape` is :attr:`MarkerShape.CUSTOM_MESH`.
    """

    shape: MarkerShape = MarkerShape.SPHERE
    size_px: float = 6.0
    edge_color: str = "#000000"
    edge_width: float = 0.5
    fill_color: ColorScale = field(default_factory=_default_fill_color)
    opacity: float = 1.0
    custom_mesh: CustomMeshSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.shape, MarkerShape):
            raise TypeError(
                f"shape must be MarkerShape; got {type(self.shape).__name__}"
            )
        if not isinstance(self.size_px, (int, float)) or isinstance(self.size_px, bool):
            raise TypeError(
                f"size_px must be numeric; got {type(self.size_px).__name__}"
            )
        size_f = float(self.size_px)
        if not math.isfinite(size_f) or size_f <= 0.0:
            raise ValueError(f"size_px must be finite and > 0; got {self.size_px!r}")

        if not isinstance(self.edge_width, (int, float)) or isinstance(
            self.edge_width, bool
        ):
            raise TypeError(
                f"edge_width must be numeric; got {type(self.edge_width).__name__}"
            )
        edge_f = float(self.edge_width)
        if not math.isfinite(edge_f) or edge_f < 0.0:
            raise ValueError(
                f"edge_width must be finite and >= 0; got {self.edge_width!r}"
            )

        if not isinstance(self.opacity, (int, float)) or isinstance(self.opacity, bool):
            raise TypeError(
                f"opacity must be numeric; got {type(self.opacity).__name__}"
            )
        opacity_f = float(self.opacity)
        if not math.isfinite(opacity_f) or not 0.0 <= opacity_f <= 1.0:
            raise ValueError(f"opacity must lie in [0, 1]; got {self.opacity!r}")

        if not isinstance(self.edge_color, str) or not self.edge_color:
            raise ValueError(
                f"edge_color must be a non-empty string; got {self.edge_color!r}"
            )
        if not is_color_like(self.edge_color):
            raise ValueError(
                f"edge_color {self.edge_color!r} is not a parseable matplotlib color"
            )

        if not isinstance(
            self.fill_color, (StaticColor, PaletteColor, DataDrivenColor)
        ):
            raise TypeError(
                "fill_color must be a ColorScale instance "
                "(StaticColor / PaletteColor / DataDrivenColor); "
                f"got {type(self.fill_color).__name__}"
            )

        is_custom_mesh = self.shape is MarkerShape.CUSTOM_MESH
        has_mesh = self.custom_mesh is not None
        if is_custom_mesh and not has_mesh:
            raise ValueError(
                "MarkerStyle.shape == CUSTOM_MESH requires custom_mesh to be set"
            )
        if has_mesh and not is_custom_mesh:
            raise ValueError(
                "custom_mesh is only allowed when shape == CUSTOM_MESH; "
                f"got shape={self.shape}"
            )
        if has_mesh and not isinstance(self.custom_mesh, CustomMeshSpec):
            raise TypeError(
                "custom_mesh must be CustomMeshSpec; "
                f"got {type(self.custom_mesh).__name__}"
            )
