"""Workspace project/session metadata and result browsing APIs."""

from __future__ import annotations

from .artifact_handoff import (
    ArtifactKind,
    ArtifactReference,
    WorkspaceHandoff,
    compute_file_sha256,
    convert_artifact,
    register_artifact_adapter,
)
from .project_store import (
    DatasetMetadata,
    ProjectMetadata,
    SessionMetadata,
    SessionProjectStore,
    SubjectMetadata,
)
from .results_browser import ResultArtifact, ResultFilter, ResultsBrowser
from .results_workspace import (
    ActionAvailability,
    ComparisonResult,
    HandoffDispatchPayload,
    MissingAssetDiagnosticError,
    ResultArtifactItem,
    ResultCategory,
    ResultsWorkspaceCoordinator,
    UnitMismatchDiagnosticError,
    WorkspaceActionType,
)

__all__ = [
    "ActionAvailability",
    "ArtifactKind",
    "ArtifactReference",
    "ComparisonResult",
    "DatasetMetadata",
    "HandoffDispatchPayload",
    "MissingAssetDiagnosticError",
    "ProjectMetadata",
    "ResultArtifact",
    "ResultArtifactItem",
    "ResultCategory",
    "ResultFilter",
    "ResultsBrowser",
    "ResultsWorkspaceCoordinator",
    "SessionMetadata",
    "SessionProjectStore",
    "SubjectMetadata",
    "UnitMismatchDiagnosticError",
    "WorkspaceActionType",
    "WorkspaceHandoff",
    "compute_file_sha256",
    "convert_artifact",
    "register_artifact_adapter",
]
