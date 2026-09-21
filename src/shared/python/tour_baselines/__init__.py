"""Tour Baselines module — single source of truth for golf model identities and coverage.

Part of the Matched Swing Program (#10363, #10584, #10585, #10586).
"""

from __future__ import annotations

from .audit import (
    AUDIT_SCHEMA,
    MarkerAuditDetail,
    TargetAuditReceipt,
    audit_tour_target,
)
from .canonical_targets import (
    CanonicalTourTarget,
    evaluate_target_tracking_error,
    load_canonical_tour_target,
)
from .coverage import (
    CoverageCell,
    ToolExclusion,
    generate_coverage_matrix,
    list_excluded_tools,
    render_coverage_markdown,
)
from .events import (
    SWING_EVENTS_DRIVER,
    SWING_EVENTS_IRON,
    BiomechanicalEvent,
    DetectionMethod,
    TourSwingEvents,
    detect_tour_events,
)
from .measurement_map import (
    MEASUREMENT_MAP_DRIVER,
    MEASUREMENT_MAP_IRON,
    MEASUREMENT_MAP_VERSION,
    MarkerMeasurementSemantics,
    MeasurementClass,
    get_measurement_map,
)
from .models import (
    BackendType,
    EvidenceStatus,
    FitMode,
    GolfModelIdentity,
    ModelTopology,
    SourceOwner,
)
from .provenance import (
    PROVENANCE_DRIVER,
    PROVENANCE_IRON,
    SHARED_PLAYER_ID,
    TourProvenance,
    get_tour_provenance,
)
from .reconciliation import (
    GoverningEpic,
    ReconciliationRecord,
    ToolsRevisionStatus,
    get_governing_epics,
    get_historical_reconciliation,
    get_tools_revision_status,
    render_reconciliation_markdown,
)
from .registry import (
    AmbiguousModelError,
    clear_golf_model_registry,
    detect_provider_mismatch,
    get_golf_model,
    init_default_registry,
    list_golf_models,
    register_golf_model,
    resolve_model_alias,
)

__all__ = [
    "AUDIT_SCHEMA",
    "AmbiguousModelError",
    "BackendType",
    "BiomechanicalEvent",
    "CanonicalTourTarget",
    "CoverageCell",
    "DetectionMethod",
    "EvidenceStatus",
    "FitMode",
    "GolfModelIdentity",
    "GoverningEpic",
    "MEASUREMENT_MAP_DRIVER",
    "MEASUREMENT_MAP_IRON",
    "MEASUREMENT_MAP_VERSION",
    "MarkerAuditDetail",
    "MarkerMeasurementSemantics",
    "MeasurementClass",
    "ModelTopology",
    "PROVENANCE_DRIVER",
    "PROVENANCE_IRON",
    "ReconciliationRecord",
    "SHARED_PLAYER_ID",
    "SWING_EVENTS_DRIVER",
    "SWING_EVENTS_IRON",
    "SourceOwner",
    "TargetAuditReceipt",
    "ToolExclusion",
    "ToolsRevisionStatus",
    "TourProvenance",
    "TourSwingEvents",
    "audit_tour_target",
    "clear_golf_model_registry",
    "detect_provider_mismatch",
    "detect_tour_events",
    "evaluate_target_tracking_error",
    "generate_coverage_matrix",
    "get_golf_model",
    "get_governing_epics",
    "get_historical_reconciliation",
    "get_measurement_map",
    "get_tools_revision_status",
    "get_tour_provenance",
    "init_default_registry",
    "list_excluded_tools",
    "list_golf_models",
    "load_canonical_tour_target",
    "register_golf_model",
    "render_coverage_markdown",
    "render_reconciliation_markdown",
    "resolve_model_alias",
]
