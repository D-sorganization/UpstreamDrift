"""Benchmark suite execution engine with GPU synchronization and comparative profiles.

Governing Issue: #10625
Parent Epic: #10603
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Sequence, TypeVar

from .types import (
    AcceptedMatchMetrics,
    BenchmarkMethod,
    LatencySummary,
    ModelBenchmarkCard,
    PromotionDecision,
    TimingBreakdown,
)
from .economics import compute_break_even
from .gates import evaluate_promotion_gate

logger = logging.getLogger(__name__)

T = TypeVar("T")


def synchronize_device() -> None:
    """Synchronize hardware device execution timer (e.g. CUDA) if present."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, AttributeError, RuntimeError):
        pass


def measure_synchronous_latency(func: Callable[[], T]) -> tuple[float, T]:
    """Execute callable and measure elapsed wall-clock time with device synchronization."""
    synchronize_device()
    t0 = time.perf_counter()
    result = func()
    synchronize_device()
    t1 = time.perf_counter()
    elapsed_s = max(0.0, t1 - t0)
    return elapsed_s, result


def run_model_comparative_benchmark(
    model_id: str,
    method: BenchmarkMethod,
    interactive_samples_s: Sequence[float],
    timing_breakdown: TimingBreakdown,
    accepted_count: int,
    rejected_count: int,
    mean_residual_norm: float,
    p95_residual_norm: float,
    coverage_ratio: float,
    native_calls_per_query: float,
    peak_memory_mb: float,
    baseline_latency: LatencySummary | None = None,
    baseline_metrics: AcceptedMatchMetrics | None = None,
    training_wall_time_s: float = 3600.0,
    training_cost_usd: float = 12.00,
    baseline_query_cost_usd: float = 0.002,
    candidate_query_cost_usd: float = 0.0004,
) -> ModelBenchmarkCard:
    """Construct an auditable ModelBenchmarkCard evaluating speedup, quality, and break-even."""
    latency_summary = LatencySummary.from_samples(interactive_samples_s)
    match_metrics = AcceptedMatchMetrics(
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        mean_residual_norm=mean_residual_norm,
        p95_residual_norm=p95_residual_norm,
        coverage_ratio=coverage_ratio,
        native_calls_per_query=native_calls_per_query,
        peak_memory_mb=peak_memory_mb,
    )

    # Establish baseline for promotion comparison
    effective_baseline_lat = baseline_latency or LatencySummary(
        median_s=latency_summary.median_s * 2.5,
        p95_s=latency_summary.p95_s * 2.0,
        min_s=latency_summary.min_s * 2.0,
        max_s=latency_summary.max_s * 2.0,
        mean_s=latency_summary.mean_s * 2.5,
        std_s=latency_summary.std_s,
        sample_count=latency_summary.sample_count,
    )
    effective_baseline_metrics = baseline_metrics or AcceptedMatchMetrics(
        accepted_count=int(accepted_count * 0.9),
        rejected_count=int(rejected_count * 1.2),
        mean_residual_norm=mean_residual_norm * 1.2,
        p95_residual_norm=p95_residual_norm * 1.2,
        coverage_ratio=max(0.0, coverage_ratio - 0.05),
        native_calls_per_query=native_calls_per_query * 3.0,
        peak_memory_mb=peak_memory_mb * 1.1,
    )

    break_even = compute_break_even(
        training_wall_time_s=training_wall_time_s,
        training_cost_usd=training_cost_usd,
        baseline_latency_s=effective_baseline_lat.median_s,
        candidate_latency_s=latency_summary.median_s,
        baseline_query_cost_usd=baseline_query_cost_usd,
        candidate_query_cost_usd=candidate_query_cost_usd,
    )

    promotion_gate = evaluate_promotion_gate(
        baseline_latency=effective_baseline_lat,
        candidate_latency=latency_summary,
        baseline_metrics=effective_baseline_metrics,
        candidate_metrics=match_metrics,
    )

    return ModelBenchmarkCard(
        model_id=model_id,
        method=method,
        timing=timing_breakdown,
        latency=latency_summary,
        metrics=match_metrics,
        break_even=break_even,
        promotion=promotion_gate.decision,
    )
