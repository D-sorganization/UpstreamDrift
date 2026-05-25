"""Body-part visualisation contracts, dataclasses, and implementations.

This package owns the canonical body-part visualisation stack used
across UpstreamDrift:

- ``contracts`` — abstract surface (Protocols) that every shape,
  fitter, and renderer implementation talks across.
- ``bindings`` — marker bindings and configuration.
- ``shapes`` — built-in body-part shape definitions.
- ``fitters`` — shape fitting implementations.
- ``renderers`` — rendering backend implementations.
- ``asset_library`` — reusable visual assets.
- ``persistence`` — segment visualisation schema and persistence.
- ``theme`` — shape styling themes.
- ``urdf_bridge`` — URDF export and bridging.

Epic #4755 shipped the full stack.
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
