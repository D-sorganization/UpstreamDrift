"""Body-part visualisation contracts and dataclasses.

This package defines the abstract surface (Protocols + frozen dataclasses)
that every shape, fitter, and renderer implementation talks across.

Implementations of shapes, fitters, and rendering backends live in the
``shapes``, ``fitters``, and ``renderers`` sub-packages and are added in
follow-up issues of EPIC #4755.
"""

from __future__ import annotations

from ._types import FittedShape
from .bindings import BindingKind, MarkerBinding
from .contracts import BodyPartShape, ShapeFitter, ShapeRenderer
from .theme import ShapeTheme

__all__ = [
    "BindingKind",
    "BodyPartShape",
    "FittedShape",
    "MarkerBinding",
    "ShapeFitter",
    "ShapeRenderer",
    "ShapeTheme",
]
