"""Body-part visualisation contracts and dataclasses.

This package defines the abstract surface (Protocols + frozen dataclasses)
that every shape, fitter, and renderer implementation talks across.

The package contains:
- ``contracts``: Core interfaces for body part shapes, shape fitters, and shape renderers.
- ``shapes``: Concrete body part shape definitions and primitives.
- ``fitters``: Algorithms to fit shapes to marker or mocap data.
- ``renderers``: Matplotlib and PyQtGL rendering backends for shape visualization.
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
