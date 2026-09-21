"""Tour Baselines module — single source of truth for golf model identities and coverage.

Part of the Matched Swing Program (#10363, #10584, #10585).
"""

from __future__ import annotations

from .coverage import (
    CoverageCell,
    ToolExclusion,
    generate_coverage_matrix,
    list_excluded_tools,
    render_coverage_markdown,
)
from .models import (
    BackendType,
    EvidenceStatus,
    FitMode,
    GolfModelIdentity,
    ModelTopology,
    SourceOwner,
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
