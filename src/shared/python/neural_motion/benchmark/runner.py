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
    MeasuredBaseline,
    ModelBenchmarkCard,
    PromotionDecision,
    TimingBreakdown,
    UNMEASURED_BASELINE_MESSAGE,
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
    metrics: AcceptedMatchMetrics,
    baseline: MeasuredBaseline | None = None,
) -> ModelBenchmarkCard:
    """Construct an auditable ModelBenchmarkCard evaluating speedup, quality, and break-even.

    Precondition: ``baseline`` is a ``MeasuredBaseline``; ``None`` raises
    ``ValueError`` (#10960 P0-5).
    """
    if baseline is None:
        raise ValueError(UNMEASURED_BASELINE_MESSAGE)

    latency_summary = LatencySummary.from_samples(interactive_samples_s)

    break_even = compute_break_even(
        training_wall_time_s=baseline.training_wall_time_s,
        training_cost_usd=baseline.training_cost_usd,
        baseline_latency_s=baseline.latency.median_s,
        candidate_latency_s=latency_summary.median_s,
        baseline_query_cost_usd=baseline.baseline_query_cost_usd,
        candidate_query_cost_usd=baseline.candidate_query_cost_usd,
    )

    promotion_gate = evaluate_promotion_gate(
        baseline_latency=baseline.latency,
        candidate_latency=latency_summary,
        baseline_metrics=baseline.metrics,
        candidate_metrics=metrics,
    )

    return ModelBenchmarkCard(
        model_id=model_id,
        method=method,
        timing=timing_breakdown,
        latency=latency_summary,
        metrics=metrics,
        break_even=break_even,
        promotion=promotion_gate.decision,
    )
