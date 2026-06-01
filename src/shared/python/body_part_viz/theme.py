"""Shape theme dataclass — colour, opacity, edge styling.

Frozen dataclass; validates colour strings via matplotlib's
``is_color_like`` so the renderers can rely on well-formed input.

Relationship to the versioned theme contract (``theme/v1``)
-----------------------------------------------------------
``ShapeTheme`` describes *per-shape geometry styling* — a single shape's fill
colour, opacity, edge colour/width and shading mode. This is deliberately
*outside* the ``theme/v1`` colour-role contract
(:mod:`src.shared.python.theme.v1`), which governs application *palette role
tokens* (``bg``, ``accent``, ``text`` …). The two concerns are orthogonal:
a ``ShapeTheme`` colour is a free matplotlib colour for one mesh, not a named
role in an app palette. Unified under #6566: the palette/role contract lives
in ``theme/v1``; shape styling stays here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from matplotlib.colors import is_color_like

__all__ = ["ShapeTheme"]


@dataclass(frozen=True)
class ShapeTheme:
    """Visual styling for a single body-part shape.

    Attributes
    ----------
    color:
        Any matplotlib-recognised colour string (e.g. ``"#1f77b4"``,
        ``"red"``, ``"C0"``).
    opacity:
        Alpha in ``[0.0, 1.0]``.
    edge_color:
        Edge colour string (matplotlib-recognised).
    edge_width:
        Edge line width in points; must be ``>= 0``.
    flat_shaded:
        If ``True`` use flat (per-face) shading; else smooth.
    group:
        Logical group name used by themed colour palettes.
    """

    color: str = "#1f77b4"
    opacity: float = 0.8
    edge_color: str = "#000000"
    edge_width: float = 0.5
    flat_shaded: bool = True
    group: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.color, str):
            raise TypeError(f"color must be str; got {type(self.color).__name__}")
        if not self.color:
            raise ValueError("color must be non-empty")
        if not is_color_like(self.color):
            raise ValueError(
                f"color {self.color!r} is not a recognised matplotlib colour"
            )

        if not isinstance(self.edge_color, str):
            raise TypeError(
                f"edge_color must be str; got {type(self.edge_color).__name__}"
            )
        if not self.edge_color:
            raise ValueError("edge_color must be non-empty")
        if not is_color_like(self.edge_color):
            raise ValueError(
                f"edge_color {self.edge_color!r} is not a recognised matplotlib colour"
            )

        if not isinstance(self.opacity, (int, float)) or isinstance(self.opacity, bool):
            raise TypeError(
                f"opacity must be numeric; got {type(self.opacity).__name__}"
            )
        opacity = float(self.opacity)
        if not math.isfinite(opacity) or opacity < 0.0 or opacity > 1.0:
            raise ValueError(
                f"opacity must be a finite value in [0, 1]; got {self.opacity!r}"
            )

        if not isinstance(self.edge_width, (int, float)) or isinstance(
            self.edge_width, bool
        ):
            raise TypeError(
                f"edge_width must be numeric; got {type(self.edge_width).__name__}"
            )
        edge_width = float(self.edge_width)
        if not math.isfinite(edge_width) or edge_width < 0.0:
            raise ValueError(
                f"edge_width must be a finite, non-negative number; "
                f"got {self.edge_width!r}"
            )

        if not isinstance(self.flat_shaded, bool):
            raise TypeError(
                f"flat_shaded must be bool; got {type(self.flat_shaded).__name__}"
            )

        if not isinstance(self.group, str):
            raise TypeError(f"group must be str; got {type(self.group).__name__}")
        if not self.group:
            raise ValueError("group must be non-empty")
