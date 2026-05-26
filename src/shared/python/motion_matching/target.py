"""Public re-export of the ``ClubTarget`` and ``ClubBallTarget`` dataclasses.

The loader/oracle/cost surface is promoted to a top-level package.
``target`` is the canonical module name (matching ``target.m`` in MATLAB
conceptual layout); historical code imports from ``club_target`` and that
import path is preserved.

Public API:
    ClubTarget       -- frozen dataclass for a measured club swing.
    AlignOptions     -- resampling and impact-alignment options.
    SourceProvenance -- file-level provenance metadata.
    BallImpactState  -- ball boundary condition at impact.
    ClubBallTarget   -- club + ball composite target.
    extract_ball_impact_from_clubtarget -- club-state-only extractor.
"""

from __future__ import annotations

from .body_target import (
    BODY_TARGET_SCHEMA_VERSION,
    MAX_BODY_POSITION_NORM_M,
    BodyEvent,
    BodyTarget,
)
from .club_ball_target import (
    CLUB_BALL_TARGET_SCHEMA_VERSION,
    DEFAULT_ELASTICITY_FACTOR,
    LAUNCH_DIR_NORM_TOL,
    MAX_LAUNCH_SPEED_MPS,
    MAX_SPIN_RPM,
    BallImpactState,
    ClubBallTarget,
    extract_ball_impact_from_clubtarget,
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
    "BallImpactState",
    "BodyEvent",
    "BodyTarget",
    "CLUB_BALL_TARGET_SCHEMA_VERSION",
    "ClubBallTarget",
    "ClubTarget",
    "DEFAULT_ELASTICITY_FACTOR",
    "LAUNCH_DIR_NORM_TOL",
    "MAX_BODY_POSITION_NORM_M",
    "MAX_LAUNCH_SPEED_MPS",
    "MAX_SPIN_RPM",
    "QUAT_NORM_TOL",
    "SourceProvenance",
    "TIME_EPS",
    "ValidAlignment",
    "extract_ball_impact_from_clubtarget",
]
