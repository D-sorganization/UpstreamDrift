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
from .persistence import (
    SCHEMA_VERSION,
    SegmentVizSet,
    SegmentVizSpec,
    VALID_FITTER_KINDS,
    VALID_SHAPE_KINDS,
    migrate_v1_to_v2,
)
from .theme import ShapeTheme

__all__ = [
    "SCHEMA_VERSION",
    "VALID_FITTER_KINDS",
    "VALID_SHAPE_KINDS",
    "BindingKind",
    "BodyPartShape",
    "FittedShape",
    "MarkerBinding",
    "SegmentVizSet",
    "SegmentVizSpec",
    "ShapeFitter",
    "ShapeRenderer",
    "ShapeTheme",
    "migrate_v1_to_v2",
]
# Issue 6180 referenced this path
