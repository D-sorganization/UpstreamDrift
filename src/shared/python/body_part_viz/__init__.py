"""Body-part visualisation contracts, dataclasses, and implementations.

This package owns the canonical body-part visualisation stack:

- ``contracts`` — runtime-checkable Protocols (``BodyPartShape``,
  ``ShapeFitter``, ``ShapeRenderer``).
- ``shapes`` — primitive shapes (capsule, ellipsoid, cylinder, mesh).
- ``fitters`` — fitting strategies (Kabsch, Procrustes, cluster).
- ``renderers`` — rendering backends (Matplotlib, PyQtGL).
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
