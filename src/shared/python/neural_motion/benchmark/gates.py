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


def _evidence_refusals(
    baseline_latency: LatencySummary | None,
    candidate_latency: LatencySummary | None,
    baseline_metrics: AcceptedMatchMetrics | None,
    candidate_metrics: AcceptedMatchMetrics | None,
) -> list[str]:
    """Return reasons the evidence cannot support promotion (fail closed, #12320).

    Evidence must be present, non-zero and comparable. No sample-size
    thresholds are applied here; those await the statistical review
    (Repository_Management#1840).
    """
    named: dict[str, object] = {
        "baseline latency": baseline_latency,
        "candidate latency": candidate_latency,
        "baseline metrics": baseline_metrics,
        "candidate metrics": candidate_metrics,
    }
    reasons = [
        f"Missing evidence: {name} not provided"
        for name, value in named.items()
        if value is None
    ]
    if reasons:
        return reasons
    assert baseline_latency is not None and candidate_latency is not None
    assert baseline_metrics is not None and candidate_metrics is not None

    for name, lat in (
        ("baseline", baseline_latency),
        ("candidate", candidate_latency),
    ):
        if lat.sample_count == 0:
            reasons.append(f"Zero latency samples in {name} evidence")
    for name, met in (
        ("baseline", baseline_metrics),
        ("candidate", candidate_metrics),
    ):
        if met.total_attempts == 0:
            reasons.append(f"Zero attempts in {name} validation evidence")
        elif met.accepted_count == 0:
            reasons.append(f"Zero accepted examples in {name} validation evidence")
    if reasons:
        return reasons

    if baseline_latency.sample_count != candidate_latency.sample_count:
        reasons.append(
            "Non-comparable evidence: latency sample counts differ "
            f"(baseline {baseline_latency.sample_count}, "
            f"candidate {candidate_latency.sample_count})"
        )
    if baseline_latency.is_gpu != candidate_latency.is_gpu:
        reasons.append("Non-comparable evidence: GPU/CPU timing modes differ")
    if baseline_metrics.total_attempts != candidate_metrics.total_attempts:
        reasons.append(
            "Non-comparable evidence: validation attempt counts differ "
            f"(baseline {baseline_metrics.total_attempts}, "
            f"candidate {candidate_metrics.total_attempts})"
        )
    return reasons


def _refused(reasons: list[str]) -> PromotionGateResult:
    return PromotionGateResult(
        decision=PromotionDecision.RESEARCH_ONLY,
        passed=False,
        median_speedup=0.0,
        p95_delta_s=0.0,
        quality_rate_delta=0.0,
        failure_reasons=tuple(reasons),
        follow_up_recommendation=(
            f"Promotion refused, evidence insufficient ({'; '.join(reasons)}). "
            f"Candidate remains RESEARCH_ONLY until comparable, non-zero evidence exists."
        ),
    )


def evaluate_promotion_gate(
    baseline_latency: LatencySummary | None,
    candidate_latency: LatencySummary | None,
    baseline_metrics: AcceptedMatchMetrics | None,
    candidate_metrics: AcceptedMatchMetrics | None,
    min_median_speedup: float = 2.0,
) -> PromotionGateResult:
    """Evaluate frozen promotion gate.

    Promotion Criteria:
        1. Median interactive speedup >= min_median_speedup (default 2.0x).
        2. Non-worse p95 latency: candidate p95 <= baseline p95 + 1e-4.
        3. Non-worse quality: candidate acceptance_rate >= baseline acceptance_rate - 1e-4.

    Fails closed: missing, zero or non-comparable evidence is refused before any
    arithmetic (#12320).

    If any condition fails, the candidate is designated RESEARCH_ONLY with an explicit follow-up recommendation.
    """
    evidence_reasons = _evidence_refusals(
        baseline_latency, candidate_latency, baseline_metrics, candidate_metrics
    )
    if evidence_reasons:
        return _refused(evidence_reasons)
    assert baseline_latency is not None and candidate_latency is not None
    assert baseline_metrics is not None and candidate_metrics is not None

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
