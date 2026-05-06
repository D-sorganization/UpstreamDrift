"""Drake motion-matching parity package.

This package implements the Drake-side of the cross-engine motion-matching
contract defined in
``src/engines/CROSS_ENGINE_PARITY_SPEC.md`` and elaborated in
``src/engines/physics_engines/drake/DRAKE_PARITY_SPEC.md``.

Issue #4108 (DRAKE-1) lands the URDF generator + loader; downstream issues
(DRAKE-2..7) consume them.
"""

from __future__ import annotations

from .humanoid_urdf import (
    CANONICAL_URDF,
    SHARED_DIMENSIONS_YAML,
    build_humanoid_urdf,
    load_humanoid_dimensions,
    load_humanoid_into_plant,
)

__all__ = [
    "CANONICAL_URDF",
    "SHARED_DIMENSIONS_YAML",
    "build_humanoid_urdf",
    "load_humanoid_dimensions",
    "load_humanoid_into_plant",
]
