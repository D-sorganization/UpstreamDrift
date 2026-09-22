"""Discovery pointer: NM-00 dataset/checkpoint audit lives in ``neural_motion``.

Issue #10615 lists ``motion_matching/surrogate/`` among the owned paths. The
fail-closed audit implementation is intentionally shared under
``src/shared/python/neural_motion/`` so later NM epic children (#10603) can
extend one package. This module re-exports the public audit surface so
surrogate callers and path-membership checks resolve honestly without
duplicating the audit logic (DRY).
"""

from __future__ import annotations

from src.shared.python.neural_motion import (
    AUDIT_SCHEMA,
    ArtifactAuditReceipt,
    ParquetInspectResult,
    audit_neural_artifacts,
    inspect_parquet_bounded,
)

__all__ = [
    "AUDIT_SCHEMA",
    "ArtifactAuditReceipt",
    "ParquetInspectResult",
    "audit_neural_artifacts",
    "inspect_parquet_bounded",
]
