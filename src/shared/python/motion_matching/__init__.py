"""Motion matching: shared loaders, oracle, cost, and visualisation.

This package is promoted from buried-under-Simscape to the
canonical Python entry point every engine imports from. The top-level
surface mirrors the MATLAB ``motion_matching/shared/`` layout one-to-one.
"""

from __future__ import annotations

from .align_to_simulation_grid import (
    AlignedTrajectory,
    align_to_simulation_grid,
    detect_impact_index,
)
from .body_skeleton import (
    BodySegment,
    BodySegmentGroup,
    default_body_segments,
)
from .compute_total_work import compute_total_work
from .cost import (
    CostBreakdown,
    CostOptions,
    SimOutput,
    compute_cost,
)
from .fit_result import CanonicalFitResult
from .body_target import (
    BODY_TARGET_SCHEMA_VERSION,
    MAX_BODY_POSITION_NORM_M,
    BodyEvent,
    BodyTarget,
)
from .load_body_target import (
    load_body_target,
    load_body_target_c3d,
)
from .load_club_target import (
    ALLOWED_SHEETS,
    load_club_target,
    load_club_target_c3d,
    load_club_target_excel,
    load_club_target_mat,
)
from .multi_source_target import MultiSourceTarget
from .plot_error_timecourse import plot_error_timecourse
from .plot_fit_quality_card import (
    FitQualityScalars,
    fit_quality_summary,
    plot_fit_quality_card,
)
from .plot_trajectory_overlay import plot_trajectory_overlay
from .sim_out import FitResult, SimOut
from .synthesize_target_from_coefficients import (
    THETA_BOUNDS,
    EngineSimulator,
    SynthOptions,
    synthesize_target_from_coefficients,
)
from .target import (
    AlignOptions,
    BallImpactState,
    ClubBallTarget,
    ClubTarget,
    SourceProvenance,
    extract_ball_impact_from_clubtarget,
)
from .validate_theta import (
    COEFFS_PER_JOINT,
    DEFAULT_THETA_BOUND_TABLE,
    validate_theta,
)
from .validators import (
    REGULARIZER_KINDS,
    must_be_finite_vector,
    must_be_monotonic_time,
    must_be_regularizer_kind,
    must_be_unit_quaternion_rows,
    must_have_fields,
)

__all__ = [
    "ALLOWED_SHEETS",
    "AlignOptions",
    "AlignedTrajectory",
    "BODY_TARGET_SCHEMA_VERSION",
    "BallImpactState",
    "BodyEvent",
    "BodySegment",
    "BodySegmentGroup",
    "BodyTarget",
    "COEFFS_PER_JOINT",
    "CanonicalFitResult",
    "ClubBallTarget",
    "ClubTarget",
    "CostBreakdown",
    "CostOptions",
    "DEFAULT_THETA_BOUND_TABLE",
    "EngineSimulator",
    "FitQualityScalars",
    "FitResult",
    "MAX_BODY_POSITION_NORM_M",
    "MultiSourceTarget",
    "REGULARIZER_KINDS",
    "SimOut",
    "SimOutput",
    "SourceProvenance",
    "SynthOptions",
    "THETA_BOUNDS",
    "align_to_simulation_grid",
    "compute_cost",
    "compute_total_work",
    "default_body_segments",
    "detect_impact_index",
    "extract_ball_impact_from_clubtarget",
    "fit_quality_summary",
    "load_body_target",
    "load_body_target_c3d",
    "load_club_target",
    "load_club_target_c3d",
    "load_club_target_excel",
    "load_club_target_mat",
    "must_be_finite_vector",
    "must_be_monotonic_time",
    "must_be_regularizer_kind",
    "must_be_unit_quaternion_rows",
    "must_have_fields",
    "plot_error_timecourse",
    "plot_fit_quality_card",
    "plot_trajectory_overlay",
    "synthesize_target_from_coefficients",
    "validate_theta",
]
