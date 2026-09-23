"""NM-08: Verified inference orchestration, distribution checks and safe fallback."""

from __future__ import annotations

from .distribution import DistributionBounds, check_target_distribution
from .orchestration import VerifiedInferenceOrchestrator
from .types import (
    INFERENCE_SCHEMA,
    AttemptRecord,
    DomainCheckResult,
    InferenceBudget,
    InferenceStatus,
    VerifiedInferenceReport,
)

__all__ = [
    "INFERENCE_SCHEMA",
    "AttemptRecord",
    "DistributionBounds",
    "DomainCheckResult",
    "InferenceBudget",
    "InferenceStatus",
    "VerifiedInferenceOrchestrator",
    "VerifiedInferenceReport",
    "check_target_distribution",
]
