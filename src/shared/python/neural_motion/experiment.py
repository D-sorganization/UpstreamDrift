"""Benefit experiment freeze for neural motion matching (NM-01 #10616).

Pre-registers nested episode stages, baselines, latency phases, promotion
gates, compute caps and the break-even formula before any production
generation or training. Negative benefit retains research checkpoints but
blocks accelerated-product promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
)
from src.shared.python.motion_matching.fit_result import CanonicalFitResult
from src.shared.python.training.config import TrainingConfig

__all__ = [
    "BASELINE_METHODS",
    "EXPERIMENT_SCHEMA",
    "LATENCY_PHASES",
    "NESTED_EPISODE_STAGES",
    "BenefitExperimentReceipt",
    "BenefitExperimentSpec",
    "BenchmarkCase",
    "BenchmarkCaseKind",
    "ComputeBudgetCaps",
    "PromotionGates",
    "break_even_queries",
    "classical_baseline_latency_s",
    "default_benefit_experiment",
    "freeze_benefit_experiment",
    "pilot_training_config",
]

EXPERIMENT_SCHEMA = "neural-benefit-experiment/1.0.0"

NESTED_EPISODE_STAGES: tuple[int, ...] = (100, 500, 2000)

BASELINE_METHODS: tuple[str, ...] = (
    "cold_classical",
    "retrieval_plus_solve",
    "existing_networks_plus_solve",
    "forward_surrogate_plus_polish",
    "inverse_proposal_plus_polish",
)

LATENCY_PHASES: tuple[str, ...] = (
    "startup",
    "preprocessing",
    "candidate_generation",
    "neural_inference",
    "physical_refinement",
    "rejected_attempts",
    "independent_replay",
)

DEFAULT_TRAINING_SEEDS: tuple[int, ...] = (0, 1, 2)

DEFAULT_C3D_DERIVED_TESTS: tuple[str, ...] = (
    "c3d.club_only.control_a",
    "c3d.club_only.control_b",
)

STATISTICAL_LIMITATIONS = (
    "Tiny empirical set: four workbook trials and two C3D-derived club-only "
    "controls do not support population confidence intervals. Synthetic query "
    "episodes validate software/stratum contracts only and are not native "
    "physical evidence. Report uncertainty and limitations with every speed "
    "or quality claim."
)


class BenchmarkCaseKind(str, Enum):
    """Kinds of cases the frozen benchmark must include."""

    REPLAY = "replay"
    REFINEMENT = "refinement"
    FAILURE = "failure"
    SYNTHETIC_QUERY = "synthetic_query"
    WORKBOOK_TARGET = "workbook_target"
    C3D_TARGET = "c3d_target"


@dataclass(frozen=True)
class PromotionGates:
    """Reviewed product-promotion gates (proposed experiment criteria)."""

    min_median_speedup: float = 2.0
    p95_must_not_worsen: bool = True
    accepted_quality_rate_must_not_degrade: bool = True
    negative_result_blocks_product_promotion: bool = True
    negative_result_retains_research_checkpoint: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_median_speedup) or self.min_median_speedup <= 0.0:
            raise ValueError("min_median_speedup must be a positive finite value")
        if not self.negative_result_blocks_product_promotion:
            raise ValueError(
                "negative_result_blocks_product_promotion must remain True"
            )
        if not self.negative_result_retains_research_checkpoint:
            raise ValueError(
                "negative_result_retains_research_checkpoint must remain True"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_median_speedup": self.min_median_speedup,
            "p95_must_not_worsen": self.p95_must_not_worsen,
            "accepted_quality_rate_must_not_degrade": (
                self.accepted_quality_rate_must_not_degrade
            ),
            "negative_result_blocks_product_promotion": (
                self.negative_result_blocks_product_promotion
            ),
            "negative_result_retains_research_checkpoint": (
                self.negative_result_retains_research_checkpoint
            ),
        }


@dataclass(frozen=True)
class ComputeBudgetCaps:
    """Hard compute/storage/wall-time caps, optionally after a timing probe.

    When ``provisional=True``, caps are declared placeholders pending a real
    timing probe. Later stages must not treat provisional caps as empirically
    derived limits.
    """

    max_cpu_core_hours: float
    max_gpu_hours: float
    max_storage_gib: float
    max_wall_time_hours: float
    timing_probe_completed: bool
    probe_receipt_id: str | None
    provisional: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("max_cpu_core_hours", self.max_cpu_core_hours),
            ("max_gpu_hours", self.max_gpu_hours),
            ("max_storage_gib", self.max_storage_gib),
            ("max_wall_time_hours", self.max_wall_time_hours),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite value")
        if self.provisional:
            if self.timing_probe_completed:
                raise ValueError(
                    "provisional caps must have timing_probe_completed=False"
                )
            if self.probe_receipt_id is not None:
                raise ValueError("provisional caps must have probe_receipt_id=None")
        else:
            if not self.timing_probe_completed:
                raise ValueError(
                    "timing_probe must be completed before freezing compute caps"
                )
            if not self.probe_receipt_id:
                raise ValueError("probe_receipt_id is required after timing_probe")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_cpu_core_hours": self.max_cpu_core_hours,
            "max_gpu_hours": self.max_gpu_hours,
            "max_storage_gib": self.max_storage_gib,
            "max_wall_time_hours": self.max_wall_time_hours,
            "timing_probe_completed": self.timing_probe_completed,
            "probe_receipt_id": self.probe_receipt_id,
            "provisional": self.provisional,
        }


@dataclass(frozen=True)
class BenchmarkCase:
    """One pre-registered benchmark case slot."""

    case_id: str
    kind: BenchmarkCaseKind
    stratum: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if not self.stratum:
            raise ValueError("stratum must be non-empty")
        if not isinstance(self.kind, BenchmarkCaseKind):
            raise ValueError("kind must be a BenchmarkCaseKind")

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "kind": self.kind.value,
            "stratum": self.stratum,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class BenefitExperimentSpec:
    """Frozen benefit-experiment registration before training dispatch."""

    nested_episode_stages: tuple[int, ...]
    training_seeds: tuple[int, ...]
    min_synthetic_queries_per_stratum: int
    workbook_trials: tuple[str, ...]
    c3d_derived_tests: tuple[str, ...]
    baseline_methods: tuple[str, ...]
    latency_phases: tuple[str, ...]
    promotion_gates: PromotionGates
    compute_caps: ComputeBudgetCaps
    benchmark_cases: tuple[BenchmarkCase, ...]
    statistical_limitations: str
    hardware_profile: str
    related_contracts: tuple[str, ...] = (
        "#10587",
        "#10606",
        "#10616",
    )

    def __post_init__(self) -> None:
        if self.nested_episode_stages != NESTED_EPISODE_STAGES:
            raise ValueError(
                f"nested_episode_stages must equal {NESTED_EPISODE_STAGES}"
            )
        if len(self.training_seeds) != 3:
            raise ValueError("training_seeds must contain exactly three seeds")
        if any(seed < 0 for seed in self.training_seeds):
            raise ValueError("training_seeds must be non-negative")
        if self.min_synthetic_queries_per_stratum < 30:
            raise ValueError("min_synthetic_queries_per_stratum must be >= 30")
        if tuple(self.workbook_trials) != CANONICAL_TRIAL_SHEETS:
            raise ValueError("workbook_trials must equal CO-00 CANONICAL_TRIAL_SHEETS")
        if len(self.c3d_derived_tests) != 2:
            raise ValueError("exactly two C3D-derived tests are required")
        if set(self.baseline_methods) != set(BASELINE_METHODS):
            raise ValueError("baseline_methods must match BASELINE_METHODS")
        if set(self.latency_phases) != set(LATENCY_PHASES):
            raise ValueError("latency_phases must match LATENCY_PHASES")
        if not self.statistical_limitations.strip():
            raise ValueError("statistical_limitations must be recorded")
        if not self.hardware_profile:
            raise ValueError("hardware_profile must be non-empty")
        kinds = {case.kind for case in self.benchmark_cases}
        required = {
            BenchmarkCaseKind.REPLAY,
            BenchmarkCaseKind.REFINEMENT,
            BenchmarkCaseKind.FAILURE,
        }
        if not required.issubset(kinds):
            raise ValueError(
                "benchmark_cases must include replay, refinement and failure"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "nested_episode_stages": list(self.nested_episode_stages),
            "training_seeds": list(self.training_seeds),
            "min_synthetic_queries_per_stratum": (
                self.min_synthetic_queries_per_stratum
            ),
            "workbook_trials": list(self.workbook_trials),
            "c3d_derived_tests": list(self.c3d_derived_tests),
            "baseline_methods": list(self.baseline_methods),
            "latency_phases": list(self.latency_phases),
            "promotion_gates": self.promotion_gates.as_dict(),
            "compute_caps": self.compute_caps.as_dict(),
            "benchmark_cases": [case.as_dict() for case in self.benchmark_cases],
            "statistical_limitations": self.statistical_limitations,
            "hardware_profile": self.hardware_profile,
            "related_contracts": list(self.related_contracts),
        }


@dataclass(frozen=True)
class BenefitExperimentReceipt:
    """Content-addressed freeze receipt for splits and gates."""

    schema: str
    split_digest: str
    gate_digest: str
    content_digest: str
    experiment: Mapping[str, Any] = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "split_digest": self.split_digest,
            "gate_digest": self.gate_digest,
            "content_digest": self.content_digest,
            "experiment": dict(self.experiment),
        }

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def break_even_queries(
    offline_cost_s: float,
    per_query_savings_s: float,
) -> float | None:
    """Return queries to break even, or None when savings are non-positive.

    break_even = offline_generation_training_teacher_cost / per_query_savings.
    If savings <= 0 there is no positive break-even.
    """
    if not math.isfinite(offline_cost_s) or offline_cost_s < 0.0:
        raise ValueError("offline_cost_s must be a finite non-negative value")
    if not math.isfinite(per_query_savings_s):
        raise ValueError("per_query_savings_s must be finite")
    if per_query_savings_s <= 0.0:
        return None
    return offline_cost_s / per_query_savings_s


def classical_baseline_latency_s(result: CanonicalFitResult) -> float:
    """Map a classical CanonicalFitResult into latency accounting."""
    return result.latency_wall_clock_s()


def pilot_training_config(
    *,
    output_dir: Path,
    model_id: str,
    seed: int,
) -> TrainingConfig:
    """Build a shared TrainingConfig for an NM-01 pilot stage."""
    return TrainingConfig.for_neural_motion_pilot(
        output_dir=output_dir,
        model_id=model_id,
        seed=seed,
    )


def _default_benchmark_cases() -> tuple[BenchmarkCase, ...]:
    cases: list[BenchmarkCase] = [
        BenchmarkCase(
            case_id="bench.replay.continuous",
            kind=BenchmarkCaseKind.REPLAY,
            stratum="native_replay",
            notes="Independent continuous replay from one initial state",
        ),
        BenchmarkCase(
            case_id="bench.refinement.polish",
            kind=BenchmarkCaseKind.REFINEMENT,
            stratum="physical_polish",
            notes="Native physical refinement after neural proposal",
        ),
        BenchmarkCase(
            case_id="bench.failure.feasibility",
            kind=BenchmarkCaseKind.FAILURE,
            stratum="feasibility_reject",
            notes="Rejected attempts retained with reasons",
        ),
        BenchmarkCase(
            case_id="bench.synthetic.stratum_min",
            kind=BenchmarkCaseKind.SYNTHETIC_QUERY,
            stratum="synthetic_held_out",
            notes=">=30 independent synthetic queries per declared stratum",
        ),
    ]
    for sheet in CANONICAL_TRIAL_SHEETS:
        cases.append(
            BenchmarkCase(
                case_id=f"bench.workbook.{sheet}",
                kind=BenchmarkCaseKind.WORKBOOK_TARGET,
                stratum="workbook_external",
                notes=f"External workbook target {sheet}",
            )
        )
    for test_id in DEFAULT_C3D_DERIVED_TESTS:
        cases.append(
            BenchmarkCase(
                case_id=f"bench.{test_id}",
                kind=BenchmarkCaseKind.C3D_TARGET,
                stratum="c3d_external",
                notes="C3D-derived club-only control",
            )
        )
    return tuple(cases)


def default_benefit_experiment() -> BenefitExperimentSpec:
    """Return the frozen default benefit-experiment registration."""
    return BenefitExperimentSpec(
        nested_episode_stages=NESTED_EPISODE_STAGES,
        training_seeds=DEFAULT_TRAINING_SEEDS,
        min_synthetic_queries_per_stratum=30,
        workbook_trials=CANONICAL_TRIAL_SHEETS,
        c3d_derived_tests=DEFAULT_C3D_DERIVED_TESTS,
        baseline_methods=BASELINE_METHODS,
        latency_phases=LATENCY_PHASES,
        promotion_gates=PromotionGates(),
        compute_caps=ComputeBudgetCaps(
            max_cpu_core_hours=48.0,
            max_gpu_hours=8.0,
            max_storage_gib=64.0,
            max_wall_time_hours=24.0,
            timing_probe_completed=False,
            probe_receipt_id=None,
            provisional=True,
        ),
        benchmark_cases=_default_benchmark_cases(),
        statistical_limitations=STATISTICAL_LIMITATIONS,
        hardware_profile=(
            "cpu_required; gpu_optional; report cold/warm CPU and available "
            "GPU median/p95 separately"
        ),
    )


def freeze_benefit_experiment(
    spec: BenefitExperimentSpec | None = None,
) -> BenefitExperimentReceipt:
    """Freeze split and gate digests for the benefit experiment."""
    experiment = spec if spec is not None else default_benefit_experiment()
    payload = experiment.as_dict()
    split_payload = {
        "nested_episode_stages": payload["nested_episode_stages"],
        "training_seeds": payload["training_seeds"],
        "min_synthetic_queries_per_stratum": payload[
            "min_synthetic_queries_per_stratum"
        ],
        "workbook_trials": payload["workbook_trials"],
        "c3d_derived_tests": payload["c3d_derived_tests"],
        "benchmark_cases": payload["benchmark_cases"],
    }
    gate_payload = {
        "promotion_gates": payload["promotion_gates"],
        "compute_caps": payload["compute_caps"],
        "baseline_methods": payload["baseline_methods"],
        "latency_phases": payload["latency_phases"],
        "hardware_profile": payload["hardware_profile"],
    }
    return BenefitExperimentReceipt(
        schema=EXPERIMENT_SCHEMA,
        split_digest=_sha256_payload(split_payload),
        gate_digest=_sha256_payload(gate_payload),
        content_digest=_sha256_payload(payload),
        experiment=payload,
    )
