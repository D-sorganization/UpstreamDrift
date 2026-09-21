"""Neural motion matching support package (epic #10603).

NM-00 (#10615) owns the fail-closed dataset/checkpoint/training-claim audit.
Later NM children extend this package; do not invent parallel trainers here.
"""

from __future__ import annotations

from .audit import (
    AUDIT_SCHEMA,
    ArtifactAuditReceipt,
    ParquetInspectResult,
    audit_neural_artifacts,
    inspect_parquet_bounded,
)
from .claims import classify_training_claim
from .coverage import NeuralCoverageCell, generate_neural_coverage_matrix
from .types import (
    ArtifactIdentity,
    ArtifactKind,
    ArtifactRole,
    ClaimStatus,
    Disposition,
)

__all__ = [
    "AUDIT_SCHEMA",
    "ArtifactAuditReceipt",
    "ArtifactIdentity",
    "ArtifactKind",
    "ArtifactRole",
    "ClaimStatus",
    "Disposition",
    "NeuralCoverageCell",
    "ParquetInspectResult",
    "audit_neural_artifacts",
    "classify_training_claim",
    "generate_neural_coverage_matrix",
    "inspect_parquet_bounded",
]
