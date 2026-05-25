"""Body-part visualisation contracts, dataclasses, and implementations.

This package defines the abstract surface (Protocols + frozen dataclasses)
that every shape, fitter, and renderer implementation talks across, and
provides their canonical implementations:

- ``shapes`` — capsule, ellipsoid, cylinder, and mesh shapes.
- ``fitters`` — Kabsch, Procrustes, and cluster fitters.
- ``renderers`` — matplotlib and PyQtGL rendering backends.
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
