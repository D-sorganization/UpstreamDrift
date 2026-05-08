"""Public re-export of the ``ClubTarget`` dataclass.

Issue #4095 promotes the loader/oracle/cost surface to a top-level package.
``target`` is the canonical module name (matching ``target.m`` in MATLAB
conceptual layout); historical code imports from ``club_target`` and that
import path is preserved.

Public API:
    ClubTarget       -- frozen dataclass for a measured club swing.
    AlignOptions     -- resampling and impact-alignment options.
    SourceProvenance -- file-level provenance metadata.
"""

from __future__ import annotations

from .body_target import (
    BODY_TARGET_SCHEMA_VERSION,
    MAX_BODY_POSITION_NORM_M,
    BodyEvent,
    BodyTarget,
)
from .club_target import (
    QUAT_NORM_TOL,
    TIME_EPS,
    AlignOptions,
    ClubTarget,
    SourceProvenance,
    ValidAlignment,
)

__all__ = [
    "AlignOptions",
    "BODY_TARGET_SCHEMA_VERSION",
    "BodyEvent",
    "BodyTarget",
    "ClubTarget",
    "MAX_BODY_POSITION_NORM_M",
    "QUAT_NORM_TOL",
    "SourceProvenance",
    "TIME_EPS",
    "ValidAlignment",
]
