"""Motion matching: shared loaders, oracle, cost, and visualisation.

This package is promoted from buried-under-Simscape to the
canonical Python entry point every engine imports from. The top-level
surface mirrors the MATLAB ``motion_matching/shared/`` layout one-to-one.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_LAZY_EXPORTS = {
    "AlignedTrajectory": ".align_to_simulation_grid",
    "align_to_simulation_grid": ".align_to_simulation_grid",
    "detect_impact_index": ".align_to_simulation_grid",
    "BodySegment": ".body_skeleton",
    "BodySegmentGroup": ".body_skeleton",
    "default_body_segments": ".body_skeleton",
    "compute_total_work": ".compute_total_work",
    "CostBreakdown": ".cost",
    "CostOptions": ".cost",
    "SimOutput": ".cost",
    "compute_cost": ".cost",
    "CanonicalFitResult": ".fit_result",
    "BODY_TARGET_SCHEMA_VERSION": ".body_target",
    "MAX_BODY_POSITION_NORM_M": ".body_target",
    "BodyEvent": ".body_target",
    "BodyTarget": ".body_target",
    "load_body_target": ".load_body_target",
    "load_body_target_c3d": ".load_body_target",
    "ALLOWED_SHEETS": ".load_club_target",
    "load_club_target": ".load_club_target",
    "load_club_target_c3d": ".load_club_target",
    "load_club_target_excel": ".load_club_target",
    "load_club_target_mat": ".load_club_target",
    "MultiSourceTarget": ".multi_source_target",
    "plot_error_timecourse": ".plot_error_timecourse",
    "FitQualityScalars": ".plot_fit_quality_card",
    "fit_quality_summary": ".plot_fit_quality_card",
    "plot_fit_quality_card": ".plot_fit_quality_card",
    "plot_trajectory_overlay": ".plot_trajectory_overlay",
    "FitResult": ".sim_out",
    "SimOut": ".sim_out",
    "THETA_BOUNDS": ".synthesize_target_from_coefficients",
    "EngineSimulator": ".synthesize_target_from_coefficients",
    "SynthOptions": ".synthesize_target_from_coefficients",
    "synthesize_target_from_coefficients": ".synthesize_target_from_coefficients",
    "AlignOptions": ".target",
    "BallImpactState": ".target",
    "ClubBallTarget": ".target",
    "ClubTarget": ".target",
    "SourceProvenance": ".target",
    "extract_ball_impact_from_clubtarget": ".target",
    "COEFFS_PER_JOINT": ".validate_theta",
    "DEFAULT_THETA_BOUND_TABLE": ".validate_theta",
    "validate_theta": ".validate_theta",
    "REGULARIZER_KINDS": ".validators",
    "must_be_finite_vector": ".validators",
    "must_be_monotonic_time": ".validators",
    "must_be_regularizer_kind": ".validators",
    "must_be_unit_quaternion_rows": ".validators",
    "must_have_fields": ".validators",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(_LAZY_EXPORTS.keys())
