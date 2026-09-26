"""Types and data contracts for NM-10 Benchmark suite.

Governing Issue: #10625
Parent Epic: #10603
"""

from __future__ import annotations

import dataclasses
from enum import Enum
import hashlib
import json
import math
from typing import Any, Final, Sequence


class BenchmarkMethod(str, Enum):
    """The 5 comparative methods evaluated under NM-10."""

    COLD_SOLVER = "cold_solver"
    RETRIEVAL_SOLVER = "retrieval_solver"
    EXISTING_NEURAL = "existing_neural"
    FORWARD_SURROGATE_POLISH = "forward_surrogate_polish"
    LEARNED_PROPOSAL_POLISH = "learned_proposal_polish"


class PromotionDecision(str, Enum):
    """Outcome of frozen promotion gate evaluation."""

    PROMOTED = "PROMOTED"
    RESEARCH_ONLY = "RESEARCH_ONLY"


@dataclasses.dataclass(frozen=True, slots=True)
class TimingBreakdown:
    """Detailed latency decomposition across pipeline stages.

    Design by Contract:
        All durations must be non-negative and finite.
    """

    preprocessing_s: float
    native_initialization_s: float
    proposal_inference_s: float
    rejected_attempts_s: float
    native_polish_s: float
    verification_replay_s: float
    io_overhead_s: float
    cold_startup_s: float = 0.0
    warm_startup_s: float = 0.0

    def __post_init__(self) -> None:
        for field in (
            "preprocessing_s",
            "native_initialization_s",
            "proposal_inference_s",
            "rejected_attempts_s",
            "native_polish_s",
            "verification_replay_s",
            "io_overhead_s",
            "cold_startup_s",
            "warm_startup_s",
        ):
            val = getattr(self, field)
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{field} must be non-negative and finite, got {val}")

    @property
    def interactive_latency_s(self) -> float:
        """Wall-clock time excluding one-time cold startup initialization."""
        return (
            self.preprocessing_s
            + self.native_initialization_s
            + self.proposal_inference_s
            + self.rejected_attempts_s
            + self.native_polish_s
            + self.verification_replay_s
            + self.io_overhead_s
        )

    @property
    def total_cold_latency_s(self) -> float:
        """Full cold-start latency including process/engine initialization."""
        return self.interactive_latency_s + self.cold_startup_s


@dataclasses.dataclass(frozen=True, slots=True)
class LatencySummary:
    """Empirical latency distribution metrics across queries."""

    median_s: float
    p95_s: float
    min_s: float
    max_s: float
    mean_s: float
    std_s: float
    sample_count: int
    is_gpu: bool = False

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError(
                f"sample_count must be non-negative, got {self.sample_count}"
            )
        for field in ("median_s", "p95_s", "min_s", "max_s", "mean_s", "std_s"):
            val = getattr(self, field)
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{field} must be non-negative and finite, got {val}")

    @classmethod
    def from_samples(
        cls, samples: Sequence[float], is_gpu: bool = False
    ) -> LatencySummary:
        """Compute statistical summary from a sequence of latency observations."""
        if not samples:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, is_gpu=is_gpu)
        sorted_s = sorted(samples)
        n = len(sorted_s)
        mean_val = sum(sorted_s) / n
        variance = sum((x - mean_val) ** 2 for x in sorted_s) / n
        std_val = math.sqrt(variance)

        # Median
        if n % 2 == 1:
            median_val = sorted_s[n // 2]
        else:
            median_val = 0.5 * (sorted_s[n // 2 - 1] + sorted_s[n // 2])

        # p95 (nearest rank or linear interpolation)
        p95_idx = int(math.ceil(0.95 * n)) - 1
        p95_idx = min(max(0, p95_idx), n - 1)
        p95_val = sorted_s[p95_idx]

        return cls(
            median_s=float(median_val),
            p95_s=float(p95_val),
            min_s=float(sorted_s[0]),
            max_s=float(sorted_s[-1]),
            mean_s=float(mean_val),
            std_s=float(std_val),
            sample_count=n,
            is_gpu=is_gpu,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class AcceptedMatchMetrics:
    """Physics-verified quality and throughput metrics.

    Design by Contract:
        Acceptance rate calculation strictly accounts for rejected attempts in denominator.
    """

    accepted_count: int
    rejected_count: int
    mean_residual_norm: float
    p95_residual_norm: float
    coverage_ratio: float
    native_calls_per_query: float
    peak_memory_mb: float

    def __post_init__(self) -> None:
        if self.accepted_count < 0 or self.rejected_count < 0:
            raise ValueError("Counts must be non-negative")
        for field in (
            "mean_residual_norm",
            "p95_residual_norm",
            "coverage_ratio",
            "native_calls_per_query",
            "peak_memory_mb",
        ):
            val = getattr(self, field)
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{field} must be non-negative and finite, got {val}")

    @property
    def total_attempts(self) -> int:
        return self.accepted_count + self.rejected_count

    @property
    def acceptance_rate(self) -> float:
        """Fraction of total attempts accepted by independent dynamic replay."""
        if self.total_attempts == 0:
            return 0.0
        return self.accepted_count / self.total_attempts


@dataclasses.dataclass(frozen=True, slots=True)
class BreakEvenReceipt:
    """Auditable economic and wall-clock break-even calculation.

    If candidate savings <= 0, has_break_even is strictly False.
    """

    training_wall_time_s: float
    training_cost_usd: float
    baseline_query_latency_s: float
    candidate_query_latency_s: float
    per_query_latency_savings_s: float
    baseline_query_cost_usd: float
    candidate_query_cost_usd: float
    per_query_dollar_savings_usd: float
    has_break_even: bool
    wall_time_break_even_queries: int | None
    dollar_break_even_queries: int | None
    explanation: str


@dataclasses.dataclass(frozen=True, slots=True)
class PromotionGateResult:
    """Evaluation of frozen promotion gate."""

    decision: PromotionDecision
    passed: bool
    median_speedup: float
    p95_delta_s: float
    quality_rate_delta: float
    failure_reasons: tuple[str, ...] = ()
    follow_up_recommendation: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class DataEfficiencyCurve:
    """Sample efficiency trajectory comparing active versus random acquisition."""

    budget_points: tuple[int, ...]
    active_acceptance_curve: tuple[float, ...]
    random_acceptance_curve: tuple[float, ...]
    sample_efficiency_multiplier: float
    active_superiority_confirmed: bool


UNMEASURED_BASELINE_MESSAGE: Final[str] = (
    "baseline must be measured, not derived from the candidate (#10960 P0-5)"
)


@dataclasses.dataclass(frozen=True, slots=True)
class MeasuredBaseline:
    """Measured baseline and cost inputs for a comparative benchmark (#10960 P0-5).

    Precondition: every field is measured; ``None`` raises ``ValueError`` so a
    baseline can never be derived from the candidate.
    """

    latency: LatencySummary
    metrics: AcceptedMatchMetrics
    training_wall_time_s: float
    training_cost_usd: float
    baseline_query_cost_usd: float
    candidate_query_cost_usd: float

    def __post_init__(self) -> None:
        if any(getattr(self, f.name) is None for f in dataclasses.fields(self)):
            raise ValueError(UNMEASURED_BASELINE_MESSAGE)


@dataclasses.dataclass(frozen=True, slots=True)
class ModelBenchmarkCard:
    """Comprehensive benchmark record for a specific model under NM-10."""

    model_id: str
    method: BenchmarkMethod
    timing: TimingBreakdown
    latency: LatencySummary
    metrics: AcceptedMatchMetrics
    break_even: BreakEvenReceipt
    promotion: PromotionDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "method": self.method.value,
            "timing": {
                "interactive_latency_s": self.timing.interactive_latency_s,
                "preprocessing_s": self.timing.preprocessing_s,
                "native_initialization_s": self.timing.native_initialization_s,
                "proposal_inference_s": self.timing.proposal_inference_s,
                "rejected_attempts_s": self.timing.rejected_attempts_s,
                "native_polish_s": self.timing.native_polish_s,
                "verification_replay_s": self.timing.verification_replay_s,
                "io_overhead_s": self.timing.io_overhead_s,
            },
            "latency": {
                "median_s": self.latency.median_s,
                "p95_s": self.latency.p95_s,
                "min_s": self.latency.min_s,
                "max_s": self.latency.max_s,
                "mean_s": self.latency.mean_s,
                "std_s": self.latency.std_s,
                "sample_count": self.latency.sample_count,
                "is_gpu": self.latency.is_gpu,
            },
            "metrics": {
                "accepted_count": self.metrics.accepted_count,
                "rejected_count": self.metrics.rejected_count,
                "acceptance_rate": self.metrics.acceptance_rate,
                "mean_residual_norm": self.metrics.mean_residual_norm,
                "p95_residual_norm": self.metrics.p95_residual_norm,
                "coverage_ratio": self.metrics.coverage_ratio,
                "native_calls_per_query": self.metrics.native_calls_per_query,
                "peak_memory_mb": self.metrics.peak_memory_mb,
            },
            "break_even": {
                "has_break_even": self.break_even.has_break_even,
                "wall_time_break_even_queries": self.break_even.wall_time_break_even_queries,
                "dollar_break_even_queries": self.break_even.dollar_break_even_queries,
                "per_query_latency_savings_s": self.break_even.per_query_latency_savings_s,
                "per_query_dollar_savings_usd": self.break_even.per_query_dollar_savings_usd,
            },
            "promotion": self.promotion.value,
        }

    def compute_digest(self) -> str:
        """Compute deterministic SHA-256 digest of benchmark record."""
        canonical_json = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
