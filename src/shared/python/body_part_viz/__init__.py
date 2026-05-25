"""Body-part visualisation contracts, dataclasses, and implementations.

This package provides the full body-part visualisation stack:

- ``contracts`` — runtime-checkable Protocols (``BodyPartShape``,
  ``ShapeFitter``, ``ShapeRenderer``).
- ``_types`` — ``FittedShape`` dataclass.
- ``shapes`` — capsule, ellipsoid, cylinder, mesh, composite, and line
  shape implementations.
- ``fitters`` — Kabsch (pairwise and cluster) and anisotropic Procrustes
  fitters.
- ``renderers`` — Matplotlib and PyQtGL rendering backends.
- ``bindings`` — ``MarkerBinding`` / ``BindingKind`` for attaching
  markers to body-part shapes.
- ``persistence`` — ``SegmentVizSpec`` / ``SegmentVizSet`` serialisation
  and v1→v2 migration.
- ``theme`` — ``ShapeTheme`` for unified visual styling.
- ``asset_library`` — built-in shape asset registry.
- ``urdf_bridge`` — helpers for importing shapes from URDF descriptions.

See ADR-0011 for the shared-style design rationale.
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
