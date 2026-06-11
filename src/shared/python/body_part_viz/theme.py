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
import re
from dataclasses import dataclass

from matplotlib.colors import is_color_like

__all__ = ["ShapeTheme"]

_HEX_COLOR_RE = re.compile(
    r"^#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{4}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$"
)


def _validate_color(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    if value.startswith("#") and not _HEX_COLOR_RE.fullmatch(value):
        raise ValueError(f"{field_name} is not a valid hex color")
    if not is_color_like(value):
        raise ValueError(f"{field_name} is not a valid color or matplotlib colour")


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
        _validate_color(self.color, "color")
        _validate_color(self.edge_color, "edge_color")

        if not isinstance(self.opacity, (int, float)) or isinstance(self.opacity, bool):
            raise TypeError("opacity must be float")
        opacity = float(self.opacity)
        if not math.isfinite(opacity):
            raise ValueError("opacity must be finite")
        if opacity < 0.0 or opacity > 1.0:
            raise ValueError("opacity must be in [0.0, 1.0]")

        if not isinstance(self.edge_width, (int, float)) or isinstance(
            self.edge_width, bool
        ):
            raise TypeError("edge_width must be float")
        edge_width = float(self.edge_width)
        if not math.isfinite(edge_width):
            raise ValueError("edge_width must be finite")
        if edge_width < 0.0:
            raise ValueError("edge_width must be >= 0")

        if not isinstance(self.flat_shaded, bool):
            raise TypeError(
                f"flat_shaded must be bool; got {type(self.flat_shaded).__name__}"
            )

        if not isinstance(self.group, str):
            raise TypeError("group must be str")
        if not self.group:
            raise ValueError("group must be non-empty")
