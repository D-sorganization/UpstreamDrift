"""Body-part visualisation contracts, dataclasses, and implementations.

This package is the single source of truth for "what does a body
segment look like" across the C3D Viewer Segments tab, the Motion-Match
Preview live view, and the URDF generator's visual links:

- ``contracts`` — runtime-checkable Protocols
  (``BodyPartShape``, ``ShapeFitter``, ``ShapeRenderer``) plus the
  ``FittedShape`` dataclass re-exported from ``_types``.
- ``bindings`` — ``MarkerBinding`` / ``BindingKind`` for attaching a
  shape to one or more mocap markers.
- ``shapes`` — built-in primitives (line, cylinder, capsule,
  ellipsoid, composite) and ``MeshShape`` with STL/OBJ/PLY/GLB loaders
  via trimesh.
- ``fitters`` — ``BetweenTwoMarkers``, ``ClusterKabsch``, and
  ``ProcrustesAnisotropic`` fitters.
- ``renderers`` — ``MatplotlibRenderer`` and ``PyQtGLRenderer``
  backends.
- ``asset_library`` — curated default body-part mesh manifest.
- ``persistence`` — ``SegmentVizSet`` / ``SegmentVizSpec`` JSON v2
  schema with the v1-to-v2 migration helper and the
  ``VALID_SHAPE_KINDS`` / ``VALID_FITTER_KINDS`` whitelists.
- ``theme`` — ``ShapeTheme`` color / style tokens.
- ``urdf_bridge`` — adapter that maps URDF link visuals onto shapes.

See AGENTS.md §B for the shipped renderer / shape / fitter inventory.
Epic #4755 shipped the full toolkit.
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
