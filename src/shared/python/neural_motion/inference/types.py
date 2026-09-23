"""Typed contracts for NM-08 verified inference orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "INFERENCE_SCHEMA",
    "AttemptRecord",
    "DomainCheckResult",
    "InferenceBudget",
    "InferenceStatus",
    "VerifiedInferenceReport",
]

INFERENCE_SCHEMA = "neural-verified-inference/1.0.0"


class InferenceStatus(str, Enum):
    """Auditable lifecycle status for verified inference."""

    NEURAL_ACCEPTED = "neural_accepted"
    NEURAL_REFINED = "neural_refined"
    CLASSICAL_FALLBACK = "classical_fallback"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class DomainCheckResult:
    """Empirical training coverage and domain diagnostic result.

    Attributes:
        is_in_distribution: Whether the target lies within training coverage.
        domain_distance: Normalized geometric/kinematic distance metric.
        empirical_confidence: Domain-support score [0.0, 1.0].
            NOTE: This is domain density information, NOT the probability
            that an inferred body is the true golfer.
        diagnostics: List of diagnostic failure/warning strings.
        contact_regime_supported: Whether contact regime is compatible.
        duration_supported: Whether motion duration is in bounds.
    """

    is_in_distribution: bool
    domain_distance: float
    empirical_confidence: float
    diagnostics: tuple[str, ...] = field(default_factory=tuple)
    contact_regime_supported: bool = True
    duration_supported: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.empirical_confidence <= 1.0):
            raise ValueError(
                f"empirical_confidence must be in [0, 1], got {self.empirical_confidence}"
            )
        if self.domain_distance < 0.0:
            raise ValueError("domain_distance must be non-negative")


@dataclass(slots=True)
class InferenceBudget:
    """Shared wall-clock compute budget across inference attempts."""

    total_budget_s: float
    t_start: float = field(default_factory=time.perf_counter)

    def __post_init__(self) -> None:
        if self.total_budget_s <= 0.0:
            raise ValueError("total_budget_s must be positive")

    def elapsed_s(self) -> float:
        return time.perf_counter() - self.t_start

    def remaining_s(self) -> float:
        return max(0.0, self.total_budget_s - self.elapsed_s())

    def is_exhausted(self) -> bool:
        return self.remaining_s() <= 0.0


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """Audit record for a single solver/proposal attempt."""

    phase: str
    controls: np.ndarray | None
    cost: float | None
    independent_replay: bool
    acceptance_status: str
    duration_s: float
    rejection_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "has_controls": self.controls is not None,
            "cost": self.cost,
            "independent_replay": self.independent_replay,
            "acceptance_status": self.acceptance_status,
            "duration_s": self.duration_s,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True, slots=True)
class VerifiedInferenceReport:
    """Consolidated report across all attempts with auditable statuses."""

    schema: str
    status: InferenceStatus
    selected_controls: np.ndarray | None
    attempts: tuple[AttemptRecord, ...]
    domain_check: DomainCheckResult
    acceptance_verdict: dict[str, Any] | None
    is_preview_only: bool
    duration_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status.value,
            "has_selected_controls": self.selected_controls is not None,
            "attempts": [a.as_dict() for a in self.attempts],
            "domain_check": {
                "is_in_distribution": self.domain_check.is_in_distribution,
                "domain_distance": self.domain_check.domain_distance,
                "empirical_confidence": self.domain_check.empirical_confidence,
                "diagnostics": list(self.domain_check.diagnostics),
                "contact_regime_supported": self.domain_check.contact_regime_supported,
                "duration_supported": self.domain_check.duration_supported,
            },
            "acceptance_verdict": self.acceptance_verdict,
            "is_preview_only": self.is_preview_only,
            "duration_s": self.duration_s,
        }
