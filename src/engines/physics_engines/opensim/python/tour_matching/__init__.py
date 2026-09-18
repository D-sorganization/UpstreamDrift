"""OpenSim tour-average marker matching (epic #10003, OS-1 onward).

Pure-Python contracts that run without the OpenSim bindings: capture label
to body mapping, TRC export/import and MarkerSet authoring. Runtime-bound
steps (IK, Moco tracking) live in drivers under
docs/development/opensim_tour_matching and skip without ``opensim``.
"""

from src.engines.physics_engines.opensim.python.tour_matching.cli import (
    CheckpointManifest,
    RunConfig,
    build_parser,
    compute_run_hash,
)
from src.engines.physics_engines.opensim.python.tour_matching.marker_calibration import (
    CalibrationResult,
    calibrate_marker_offsets,
)
from src.engines.physics_engines.opensim.python.tour_matching.marker_map import (
    GOLF_HUMANOID_MARKER_BODIES,
    body_for,
    labels_per_body,
)
from src.engines.physics_engines.opensim.python.tour_matching.marker_set import (
    MarkerPlacement,
    attach_marker_set,
    locked_coordinates,
    unlock_coordinates,
    write_model,
)
from src.engines.physics_engines.opensim.python.tour_matching.metrics import (
    SharedMetrics,
    compute_shared_metrics,
)
from src.engines.physics_engines.opensim.python.tour_matching.moco_tracking import (
    MocoTrackingConfig,
    MocoTrackingResult,
    build_moco_study,
    sanitize_trc_for_horizon,
)
from src.engines.physics_engines.opensim.python.tour_matching.polynomial_profile import (
    Degree6PolynomialCoefficients,
    PolynomialTorqueProfile,
    check_effort_and_rate_bounds,
    create_polynomial_prescribed_controller,
    fit_degree6_from_discrete_controls,
    load_controls_from_sto,
)
from src.engines.physics_engines.opensim.python.tour_matching.scale import (
    DEFAULT_NOMINAL_LENGTHS_M,
    SegmentScaleResult,
    estimate_segment_scales,
)
from src.engines.physics_engines.opensim.python.tour_matching.trc import (
    read_trc,
    write_trc,
)
from src.engines.physics_engines.opensim.python.tour_matching.model_audit import (
    BASELINE_MODEL_SHA256,
    BASELINE_OS3B_INPUT_SHA256,
    ModelGeometryAuditResult,
    audit_model_geometry,
    verify_model_qualification,
)
from src.engines.physics_engines.opensim.python.tour_matching.club_geometry import (
    attach_visual_club,
    get_club_frame_offsets,
    has_visual_club,
)
from src.engines.physics_engines.opensim.python.tour_matching.segment_scaling import (
    ScalingPolicy,
    SegmentScalingReport,
    apply_segment_scaling,
)

from src.engines.physics_engines.opensim.python.tour_matching.visualization import (
    plot_3d_trajectory_overlay,
    plot_effort_and_rates,
    plot_marker_error_timecourse,
)

__all__ = [
    "BASELINE_MODEL_SHA256",
    "BASELINE_OS3B_INPUT_SHA256",
    "DEFAULT_NOMINAL_LENGTHS_M",
    "CalibrationResult",
    "CheckpointManifest",
    "Degree6PolynomialCoefficients",
    "GOLF_HUMANOID_MARKER_BODIES",
    "MarkerPlacement",
    "MocoTrackingConfig",
    "MocoTrackingResult",
    "ModelGeometryAuditResult",
    "PolynomialTorqueProfile",
    "RunConfig",
    "ScalingPolicy",
    "SegmentScaleResult",
    "SegmentScalingReport",
    "SharedMetrics",
    "apply_segment_scaling",
    "attach_marker_set",
    "attach_visual_club",
    "audit_model_geometry",
    "body_for",
    "build_moco_study",
    "build_parser",
    "calibrate_marker_offsets",
    "check_effort_and_rate_bounds",
    "compute_run_hash",
    "compute_shared_metrics",
    "create_polynomial_prescribed_controller",
    "estimate_segment_scales",
    "fit_degree6_from_discrete_controls",
    "get_club_frame_offsets",
    "has_visual_club",
    "labels_per_body",
    "load_controls_from_sto",
    "locked_coordinates",
    "plot_3d_trajectory_overlay",
    "plot_effort_and_rates",
    "plot_marker_error_timecourse",
    "read_trc",
    "sanitize_trc_for_horizon",
    "unlock_coordinates",
    "verify_model_qualification",
    "write_model",
    "write_trc",
]
