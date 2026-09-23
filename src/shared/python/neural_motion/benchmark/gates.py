"""Frozen promotion gates for NM-10 benchmark.

Governing Issue: #10625
Parent Epic: #10603
"""

from __future__ import annotations

import math
from typing import Final

from .types import (
    AcceptedMatchMetrics,
    LatencySummary,
    PromotionDecision,
    PromotionGateResult,
)


def evaluate_promotion_gate(
    baseline_latency: LatencySummary,
    candidate_latency: LatencySummary,
    baseline_metrics: AcceptedMatchMetrics,
    candidate_metrics: AcceptedMatchMetrics,
    min_median_speedup: float = 2.0,
) -> PromotionGateResult:
    """Evaluate frozen promotion gate.

    Promotion Criteria:
        1. Median interactive speedup >= min_median_speedup (default 2.0x).
        2. Non-worse p95 latency: candidate p95 <= baseline p95 + 1e-4.
        3. Non-worse quality: candidate acceptance_rate >= baseline acceptance_rate - 1e-4.

    If any condition fails, the candidate is designated RESEARCH_ONLY with an explicit follow-up recommendation.
    """
    if baseline_latency.median_s <= 0.0 or candidate_latency.median_s <= 0.0:
        raise ValueError("Median latencies must be strictly positive")

    median_speedup = baseline_latency.median_s / candidate_latency.median_s
    p95_delta_s = candidate_latency.p95_s - baseline_latency.p95_s
    quality_rate_delta = (
        candidate_metrics.acceptance_rate - baseline_metrics.acceptance_rate
    )

    failure_reasons: list[str] = []

    # Gate 1: Median Speedup
    if median_speedup < min_median_speedup:
        failure_reasons.append(
            f"Insufficient median speedup: achieved {median_speedup:.2f}x, required >={min_median_speedup:.2f}x"
        )

    # Gate 2: Non-worse p95 latency
    if p95_delta_s > 1e-4:
        failure_reasons.append(
            f"p95 latency regressed: candidate ({candidate_latency.p95_s:.4f}s) > baseline ({baseline_latency.p95_s:.4f}s) by {p95_delta_s:.4f}s"
        )

    # Gate 3: Non-worse quality acceptance rate
    if quality_rate_delta < -1e-4:
        failure_reasons.append(
            f"Acceptance rate regressed: candidate ({candidate_metrics.acceptance_rate:.4f}) < baseline ({baseline_metrics.acceptance_rate:.4f}) by {abs(quality_rate_delta):.4f}"
        )

    passed = len(failure_reasons) == 0

    if passed:
        decision = PromotionDecision.PROMOTED
        recommendation = (
            f"Candidate qualified for production deployment with {median_speedup:.2f}x median acceleration "
            f"and non-worse tail latency / quality."
        )
    else:
        decision = PromotionDecision.RESEARCH_ONLY
        recommendation = (
            f"Candidate remains RESEARCH_ONLY due to failing promotion gate ({'; '.join(failure_reasons)}). "
            f"Requires explicit follow-up investigation under research track."
        )

    return PromotionGateResult(
        decision=decision,
        passed=passed,
        median_speedup=float(median_speedup),
        p95_delta_s=float(p95_delta_s),
        quality_rate_delta=float(quality_rate_delta),
        failure_reasons=tuple(failure_reasons),
        follow_up_recommendation=recommendation,
    )
