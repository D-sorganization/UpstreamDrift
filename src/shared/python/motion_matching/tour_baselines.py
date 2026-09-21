"""Re-export of Tour Baselines infrastructure under motion_matching.

Enables seamless discovery from src.shared.python.motion_matching.tour_baselines.
"""

from __future__ import annotations

from src.shared.python.tour_baselines import (
    AmbiguousModelError,
    BackendType,
    CoverageCell,
    EvidenceStatus,
    FitMode,
    GolfModelIdentity,
    GoverningEpic,
    ModelTopology,
    ReconciliationRecord,
    SourceOwner,
    ToolExclusion,
    ToolsRevisionStatus,
    clear_golf_model_registry,
    detect_provider_mismatch,
    generate_coverage_matrix,
    get_golf_model,
    get_governing_epics,
    get_historical_reconciliation,
    get_tools_revision_status,
    init_default_registry,
    list_excluded_tools,
    list_golf_models,
    register_golf_model,
    render_coverage_markdown,
    render_reconciliation_markdown,
    resolve_model_alias,
)

__all__ = [
    "AmbiguousModelError",
    "BackendType",
    "CoverageCell",
    "EvidenceStatus",
    "FitMode",
    "GolfModelIdentity",
    "GoverningEpic",
    "ModelTopology",
    "ReconciliationRecord",
    "SourceOwner",
    "ToolExclusion",
    "ToolsRevisionStatus",
    "clear_golf_model_registry",
    "detect_provider_mismatch",
    "generate_coverage_matrix",
    "get_golf_model",
    "get_governing_epics",
    "get_historical_reconciliation",
    "get_tools_revision_status",
    "init_default_registry",
    "list_excluded_tools",
    "list_golf_models",
    "register_golf_model",
    "render_coverage_markdown",
    "render_reconciliation_markdown",
    "resolve_model_alias",
]
