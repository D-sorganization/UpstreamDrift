"""Frozen benefit-experiment registration for neural motion matching (NM-01 #10616).

Pre-registers baselines, pilot scales, promotion gates, latency phases and the
break-even formula before any production generation or training. A negative
benefit result retains research checkpoints but blocks accelerated-product
promotion; it does not cancel required per-model deliverables.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.shared.python.tour_baselines.qualification_profiles import (
    QUALIFICATION_PROFILE_VERSION,
)

__all__ = [
    "BENEFIT_EXPERIMENT_SCHEMA",
    "REQUIRED_BASELINES",
    "REQUIRED_LATENCY_PHASES",
    "BaselineKind",
    "BenefitExperimentSpec",
    "BreakEvenResult",
    "LatencyPhase",
    "PilotScale",
    "PromotionGates",
    "compute_break_even",
    "default_benefit_experiment",
    "freeze_digest",
]

BENEFIT_EXPERIMENT_SCHEMA = "neural-benefit-experiment/1.0.0"


class BaselineKind(str, Enum):
    """Pre-registered comparison baselines for the benefit experiment."""

    COLD_CLASSICAL = "cold_classical"
    RETRIEVAL_PLUS_SOLVE = "retrieval_plus_solve"
    EXISTING_NETWORKS_PLUS_SOLVE = "existing_networks_plus_solve"
    FORWARD_SURROGATE_PLUS_POLISH = "forward_surrogate_plus_polish"
    PROPOSED_INVERSE_PLUS_POLISH = "proposed_inverse_plus_polish"


class LatencyPhase(str, Enum):
    """All-phase wall-time accounting buckets (aligned with CO-02/#10587 intent)."""

    STARTUP = "startup"
    PREPROCESSING = "preprocessing"
    CANDIDATE_GENERATION = "candidate_generation"
    NEURAL_INFERENCE = "neural_inference"
    PHYSICAL_REFINEMENT = "physical_refinement"
    REJECTED_ATTEMPTS = "rejected_attempts"
    INDEPENDENT_REPLAY = "independent_replay"


REQUIRED_BASELINES: tuple[BaselineKind, ...] = (
    BaselineKind.COLD_CLASSICAL,
    BaselineKind.RETRIEVAL_PLUS_SOLVE,
    BaselineKind.EXISTING_NETWORKS_PLUS_SOLVE,
    BaselineKind.FORWARD_SURROGATE_PLUS_POLISH,
    BaselineKind.PROPOSED_INVERSE_PLUS_POLISH,
)

REQUIRED_LATENCY_PHASES: tuple[LatencyPhase, ...] = (
    LatencyPhase.STARTUP,
    LatencyPhase.PREPROCESSING,
    LatencyPhase.CANDIDATE_GENERATION,
    LatencyPhase.NEURAL_INFERENCE,
    LatencyPhase.PHYSICAL_REFINEMENT,
    LatencyPhase.REJECTED_ATTEMPTS,
    LatencyPhase.INDEPENDENT_REPLAY,
)


@dataclass(frozen=True, slots=True)
class PilotScale:
    """Nested pilot episode stages and held-out query minima."""

    episode_stages: tuple[int, ...] = (100, 500, 2000)
    n_seeds: int = 3
    min_synthetic_queries_per_stratum: int = 30
    n_workbook_trials: int = 4
    n_c3d_tests: int = 2
    statistical_limitations: str = (
        "Four workbook trials and two C3D-derived tests are a tiny empirical "
        "set; do not report population confidence. Synthetic strata validate "
        "software contracts and within-model generalization only."
    )

    def __post_init__(self) -> None:
        if self.episode_stages != (100, 500, 2000):
            raise ValueError("episode_stages must be the frozen (100, 500, 2000) nest")
        if self.n_seeds != 3:
            raise ValueError("n_seeds must be the frozen value 3")
        if self.min_synthetic_queries_per_stratum < 30:
            raise ValueError("min_synthetic_queries_per_stratum must be >= 30")
        if self.n_workbook_trials != 4:
            raise ValueError("n_workbook_trials must be 4")
        if self.n_c3d_tests != 2:
            raise ValueError("n_c3d_tests must be 2")
        if not self.statistical_limitations.strip():
            raise ValueError("statistical_limitations must be recorded")

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_stages": list(self.episode_stages),
            "n_seeds": self.n_seeds,
            "min_synthetic_queries_per_stratum": (
                self.min_synthetic_queries_per_stratum
            ),
            "n_workbook_trials": self.n_workbook_trials,
            "n_c3d_tests": self.n_c3d_tests,
            "statistical_limitations": self.statistical_limitations,
        }


@dataclass(frozen=True, slots=True)
class PromotionGates:
    """Proposed acceleration promotion criteria (not measured outcomes)."""

    median_speedup_min: float = 2.0
    p95_nonworse: bool = True
    accepted_quality_rate_nonworse: bool = True
    is_measured_outcome: bool = False

    def __post_init__(self) -> None:
        if self.median_speedup_min != 2.0:
            raise ValueError("median_speedup_min is frozen at 2.0 for NM-01")
        if self.is_measured_outcome:
            raise ValueError(
                "promotion gates are proposed experiment criteria; "
                "is_measured_outcome must remain False until NM-10 evidence"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "median_speedup_min": self.median_speedup_min,
            "p95_nonworse": self.p95_nonworse,
            "accepted_quality_rate_nonworse": self.accepted_quality_rate_nonworse,
            "is_measured_outcome": self.is_measured_outcome,
        }


@dataclass(frozen=True, slots=True)
class BreakEvenResult:
    """Break-even evaluation: offline cost divided by per-query savings."""

    offline_cost_s: float
    per_query_savings_s: float
    queries_to_break_even: float | None
    has_positive_break_even: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "offline_cost_s": self.offline_cost_s,
            "per_query_savings_s": self.per_query_savings_s,
            "queries_to_break_even": self.queries_to_break_even,
            "has_positive_break_even": self.has_positive_break_even,
        }


def compute_break_even(
    offline_cost_s: float,
    per_query_savings_s: float,
) -> BreakEvenResult | None:
    """Return break-even queries, or ``None`` when savings are nonpositive.

    Formula: ``queries = offline_cost / per_query_savings`` (same units).
    When ``per_query_savings_s <= 0`` there is no positive break-even.
    """
    if offline_cost_s < 0.0:
        raise ValueError("offline_cost_s must be non-negative")
    if per_query_savings_s <= 0.0:
        return None
    queries = offline_cost_s / per_query_savings_s
    return BreakEvenResult(
        offline_cost_s=float(offline_cost_s),
        per_query_savings_s=float(per_query_savings_s),
        queries_to_break_even=float(queries),
        has_positive_break_even=True,
    )


@dataclass(frozen=True, slots=True)
class BenefitExperimentSpec:
    """Immutable registration of the NM-01 benefit experiment.

    Design by Contract:
    - ``baselines`` must equal :data:`REQUIRED_BASELINES`.
    - ``latency_phases`` must equal :data:`REQUIRED_LATENCY_PHASES`.
    - Digests are computed from the frozen payload (excluding the digests).
    """

    schema: str
    baselines: tuple[BaselineKind, ...]
    latency_phases: tuple[LatencyPhase, ...]
    pilot_scale: PilotScale
    promotion_gates: PromotionGates
    split_policy: str
    hardware_targets: tuple[str, ...]
    qualification_profile_version: str
    qualification_alignment_note: str
    compute_caps_pending_timing_probe: bool = True
    governing_issue: str = "#10616"
    split_digest: str = field(default="", compare=False)
    gate_digest: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.schema != BENEFIT_EXPERIMENT_SCHEMA:
            raise ValueError(
                f"schema must be {BENEFIT_EXPERIMENT_SCHEMA!r}, got {self.schema!r}"
            )
        if tuple(self.baselines) != REQUIRED_BASELINES:
            raise ValueError(
                "baselines must exactly match the pre-registered REQUIRED_BASELINES set"
            )
        if tuple(self.latency_phases) != REQUIRED_LATENCY_PHASES:
            raise ValueError(
                "latency_phases must exactly match REQUIRED_LATENCY_PHASES "
                "(includes replay, refinement and rejected attempts)"
            )
        if not self.split_policy:
            raise ValueError("split_policy must be non-empty")
        if not self.hardware_targets:
            raise ValueError("hardware_targets must be non-empty")
        if not self.qualification_profile_version:
            raise ValueError("qualification_profile_version must be non-empty")
        if not self.qualification_alignment_note:
            raise ValueError("qualification_alignment_note must be non-empty")
        digest = freeze_digest(self)
        object.__setattr__(self, "split_digest", digest)
        object.__setattr__(self, "gate_digest", digest)

    def evaluate_break_even(
        self,
        offline_cost_s: float,
        per_query_savings_s: float,
    ) -> BreakEvenResult:
        """Evaluate break-even under this frozen experiment registration."""
        result = compute_break_even(offline_cost_s, per_query_savings_s)
        if result is None:
            return BreakEvenResult(
                offline_cost_s=float(offline_cost_s),
                per_query_savings_s=float(per_query_savings_s),
                queries_to_break_even=None,
                has_positive_break_even=False,
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "governing_issue": self.governing_issue,
            "baselines": [b.value for b in self.baselines],
            "latency_phases": [p.value for p in self.latency_phases],
            "pilot_scale": self.pilot_scale.as_dict(),
            "promotion_gates": self.promotion_gates.as_dict(),
            "split_policy": self.split_policy,
            "hardware_targets": list(self.hardware_targets),
            "qualification_profile_version": self.qualification_profile_version,
            "qualification_alignment_note": self.qualification_alignment_note,
            "compute_caps_pending_timing_probe": (
                self.compute_caps_pending_timing_probe
            ),
            "split_digest": self.split_digest,
            "gate_digest": self.gate_digest,
        }


def _payload_for_digest(spec: BenefitExperimentSpec) -> dict[str, Any]:
    """Public payload used for digests (digests themselves excluded)."""
    return {
        "schema": spec.schema,
        "governing_issue": spec.governing_issue,
        "baselines": [b.value for b in spec.baselines],
        "latency_phases": [p.value for p in spec.latency_phases],
        "pilot_scale": spec.pilot_scale.as_dict(),
        "promotion_gates": spec.promotion_gates.as_dict(),
        "split_policy": spec.split_policy,
        "hardware_targets": list(spec.hardware_targets),
        "qualification_profile_version": spec.qualification_profile_version,
        "qualification_alignment_note": spec.qualification_alignment_note,
        "compute_caps_pending_timing_probe": spec.compute_caps_pending_timing_probe,
    }


def freeze_digest(spec: BenefitExperimentSpec) -> str:
    """SHA-256 of the sorted JSON payload (excluding digest fields)."""
    payload = json.dumps(
        _payload_for_digest(spec),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_benefit_experiment() -> BenefitExperimentSpec:
    """Return the frozen NM-01 benefit experiment registration."""
    return BenefitExperimentSpec(
        schema=BENEFIT_EXPERIMENT_SCHEMA,
        baselines=REQUIRED_BASELINES,
        latency_phases=REQUIRED_LATENCY_PHASES,
        pilot_scale=PilotScale(),
        promotion_gates=PromotionGates(),
        split_policy="by_source_trial_seed_family_geometry_and_contact_regime",
        hardware_targets=("cpu_cold", "cpu_warm", "gpu_if_available"),
        qualification_profile_version=QUALIFICATION_PROFILE_VERSION,
        qualification_alignment_note=(
            "Native accuracy and feasibility gates reuse TB-02/#10587 "
            f"qualification profile version {QUALIFICATION_PROFILE_VERSION}; "
            "model-specific reduced profiles must not silently inherit "
            "full-body G3 thresholds when body markers are unobserved."
        ),
        compute_caps_pending_timing_probe=True,
    )
