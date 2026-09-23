"""Types and schemas for NM-07 forward surrogate comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "SURROGATE_COMPARISON_SCHEMA",
    "SurrogateAblationResult",
    "SurrogateCandidateKind",
    "SurrogateComparisonConfig",
    "SurrogateComparisonReport",
]

SURROGATE_COMPARISON_SCHEMA = "neural-surrogate-comparison/1.0.0"


class SurrogateCandidateKind(str, Enum):
    """Ablation candidate families evaluated under NM-07."""

    FORWARD_SURROGATE_INVERT = "forward_surrogate_invert"
    FORWARD_SURROGATE_POLISH = "forward_surrogate_polish"
    PHYSICS_STRUCTURED_RESIDUAL = "physics_structured_residual"
    MASKED_PROPOSAL = "masked_proposal"
    CVAE_MIXTURE = "cvae_mixture"
    DIFFUSION_FALLBACK = "diffusion_fallback"


@dataclass(frozen=True, slots=True)
class SurrogateComparisonConfig:
    """Pre-registered configuration for forward surrogate and alternative comparison."""

    model_id: str
    trust_radius: float = 2.0
    gradient_cos_sim_threshold: float = 0.3
    max_accepted_query_latency_s: float = 1.0
    max_iterations: int = 100
    stopping_criterion_tol: float = 1e-4
    enforce_measured_units: bool = True
    max_clubhead_rmse_m: float = 0.05
    max_orientation_rmse_rad: float = 0.1

    def __post_init__(self) -> None:
        """Design by Contract validation."""
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not (np.isfinite(self.trust_radius) and self.trust_radius > 0.0):
            raise ValueError("trust_radius must be positive")
        if not (0.0 < self.gradient_cos_sim_threshold <= 1.0):
            raise ValueError("gradient_cos_sim_threshold must be in (0, 1]")
        if not (
            np.isfinite(self.max_accepted_query_latency_s)
            and self.max_accepted_query_latency_s > 0.0
        ):
            raise ValueError("max_accepted_query_latency_s must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if not (
            np.isfinite(self.stopping_criterion_tol)
            and self.stopping_criterion_tol > 0.0
        ):
            raise ValueError("stopping_criterion_tol must be positive")


@dataclass(frozen=True, slots=True)
class SurrogateAblationResult:
    """Outcome and cost metrics for one ablation candidate."""

    candidate_kind: str
    sample_efficiency: float
    accepted_query_latency_s: float
    clubhead_rmse_m: float
    butt_rmse_m: float
    orientation_rmse_rad: float
    trust_region_rejected: bool
    gradient_fidelity_rejected: bool
    contact_boundary_failure: bool
    converged: bool
    n_evaluations: int
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_kind": self.candidate_kind,
            "sample_efficiency": float(self.sample_efficiency),
            "accepted_query_latency_s": float(self.accepted_query_latency_s),
            "clubhead_rmse_m": float(self.clubhead_rmse_m),
            "butt_rmse_m": float(self.butt_rmse_m),
            "orientation_rmse_rad": float(self.orientation_rmse_rad),
            "trust_region_rejected": bool(self.trust_region_rejected),
            "gradient_fidelity_rejected": bool(self.gradient_fidelity_rejected),
            "contact_boundary_failure": bool(self.contact_boundary_failure),
            "converged": bool(self.converged),
            "n_evaluations": int(self.n_evaluations),
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True, slots=True)
class SurrogateComparisonReport:
    """Evidence bundle comparing forward surrogates and alternatives."""

    schema_version: str
    model_id: str
    candidates: dict[str, SurrogateAblationResult]
    selected_approach: str
    selection_rationale: str
    timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "candidates": {k: v.as_dict() for k, v in self.candidates.items()},
            "selected_approach": self.selected_approach,
            "selection_rationale": self.selection_rationale,
            "timestamp": self.timestamp,
        }
