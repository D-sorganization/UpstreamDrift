"""Unit tests for NM-10: Benchmark Accepted-Match Speed, Data Efficiency and Break-Even.

Governing Issue: #10625
Parent Epic: #10603
"""

from __future__ import annotations

import math
from typing import Final
import pytest

pytestmark = pytest.mark.unit

from src.shared.python.neural_motion.benchmark.types import (
    BenchmarkMethod,
    TimingBreakdown,
    LatencySummary,
    AcceptedMatchMetrics,
    DataEfficiencyCurve,
    BreakEvenReceipt,
    PromotionDecision,
    MeasuredBaseline,
    ModelBenchmarkCard,
)
from src.shared.python.neural_motion.benchmark.economics import (
    compute_break_even,
)
from src.shared.python.neural_motion.benchmark.efficiency import (
    evaluate_data_efficiency,
)
from src.shared.python.neural_motion.benchmark.gates import (
    evaluate_promotion_gate,
)
from src.shared.python.neural_motion.benchmark.runner import (
    measure_synchronous_latency,
    run_model_comparative_benchmark,
)
from src.shared.python.motion_matching.performance_baseline import (
    PerformanceBaseline,
    record_neural_motion_benchmark,
)


def test_benchmark_method_enum_values() -> None:
    """BenchmarkMethod must enumerate all 5 required comparative techniques."""
    methods = {m.value for m in BenchmarkMethod}
    expected = {
        "cold_solver",
        "retrieval_solver",
        "existing_neural",
        "forward_surrogate_polish",
        "learned_proposal_polish",
    }
    assert methods == expected


def test_timing_breakdown_total_and_validation() -> None:
    """TimingBreakdown accounts for all pipeline stages and validates non-negative finite values."""
    tb = TimingBreakdown(
        preprocessing_s=0.005,
        native_initialization_s=0.020,
        proposal_inference_s=0.012,
        rejected_attempts_s=0.008,
        native_polish_s=0.045,
        verification_replay_s=0.015,
        io_overhead_s=0.002,
        cold_startup_s=0.100,
        warm_startup_s=0.001,
    )
    # Interactive latency excludes one-time cold startup
    expected_interactive = 0.005 + 0.020 + 0.012 + 0.008 + 0.045 + 0.015 + 0.002
    assert math.isclose(tb.interactive_latency_s, expected_interactive, rel_tol=1e-6)
    assert math.isclose(
        tb.total_cold_latency_s, expected_interactive + 0.100, rel_tol=1e-6
    )

    # Rejection of non-finite or negative timing
    with pytest.raises(ValueError, match="must be non-negative"):
        TimingBreakdown(
            preprocessing_s=-0.001,
            native_initialization_s=0.01,
            proposal_inference_s=0.01,
            rejected_attempts_s=0.0,
            native_polish_s=0.01,
            verification_replay_s=0.01,
            io_overhead_s=0.0,
        )


def test_latency_summary_percentiles() -> None:
    """LatencySummary correctly evaluates median, p95, min, max, and std."""
    samples = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    summary = LatencySummary.from_samples(samples, is_gpu=False)
    assert math.isclose(summary.median_s, 0.055, rel_tol=1e-4)
    assert summary.p95_s >= 0.09
    assert summary.min_s == 0.01
    assert summary.max_s == 0.10
    assert summary.sample_count == 10
    assert not summary.is_gpu


def test_acceptance_rate_includes_rejected_attempts() -> None:
    """Acceptance rate denominator strictly includes rejected attempts."""
    metrics = AcceptedMatchMetrics(
        accepted_count=80,
        rejected_count=20,
        mean_residual_norm=0.00042,
        p95_residual_norm=0.00085,
        coverage_ratio=0.96,
        native_calls_per_query=4.2,
        peak_memory_mb=145.0,
    )
    # 80 / (80 + 20) = 0.80
    assert math.isclose(metrics.acceptance_rate, 0.80, rel_tol=1e-6)

    # Empty denominator must be 0.0, not ZeroDivisionError
    empty_metrics = AcceptedMatchMetrics(
        accepted_count=0,
        rejected_count=0,
        mean_residual_norm=0.0,
        p95_residual_norm=0.0,
        coverage_ratio=0.0,
        native_calls_per_query=0.0,
        peak_memory_mb=0.0,
    )
    assert empty_metrics.acceptance_rate == 0.0


def test_measure_synchronous_latency_context() -> None:
    """measure_synchronous_latency executes operation and synchronizes timer accurately."""

    def dummy_op() -> int:
        acc = sum(i * i for i in range(1000))
        return acc

    latency_s, result = measure_synchronous_latency(dummy_op)
    assert latency_s > 0.0
    assert result == 332833500


def test_break_even_positive_savings() -> None:
    """Break-even queries computed accurately when candidate yields positive per-query savings."""
    receipt = compute_break_even(
        training_wall_time_s=3600.0,
        training_cost_usd=15.00,
        baseline_latency_s=0.500,
        candidate_latency_s=0.100,
        baseline_query_cost_usd=0.0025,
        candidate_query_cost_usd=0.0005,
    )
    assert receipt.has_break_even is True
    # Wall time savings: 0.500 - 0.100 = 0.400 s per query
    # N_wall = 3600 / 0.400 = 9000 queries
    assert receipt.wall_time_break_even_queries == 9000
    # Dollar savings: 0.0025 - 0.0005 = 0.0020 USD per query
    # N_dollar = 15.00 / 0.0020 = 7500 queries
    assert receipt.dollar_break_even_queries == 7500


def test_break_even_zero_or_negative_savings() -> None:
    """Zero or negative savings strictly yield has_break_even=False and None query counts."""
    receipt_zero = compute_break_even(
        training_wall_time_s=3600.0,
        training_cost_usd=15.00,
        baseline_latency_s=0.200,
        candidate_latency_s=0.200,
        baseline_query_cost_usd=0.0010,
        candidate_query_cost_usd=0.0010,
    )
    assert receipt_zero.has_break_even is False
    assert receipt_zero.wall_time_break_even_queries is None
    assert receipt_zero.dollar_break_even_queries is None
    assert "no break-even" in receipt_zero.explanation.lower()

    receipt_negative = compute_break_even(
        training_wall_time_s=3600.0,
        training_cost_usd=15.00,
        baseline_latency_s=0.150,
        candidate_latency_s=0.300,
        baseline_query_cost_usd=0.0010,
        candidate_query_cost_usd=0.0025,
    )
    assert receipt_negative.has_break_even is False
    assert receipt_negative.wall_time_break_even_queries is None
    assert receipt_negative.dollar_break_even_queries is None


def test_promotion_gate_success() -> None:
    """Promotion gate approves candidate with >=2x median speedup, non-worse p95, and non-worse quality."""
    baseline = AcceptedMatchMetrics(
        accepted_count=70,
        rejected_count=30,
        mean_residual_norm=0.0010,
        p95_residual_norm=0.0020,
        coverage_ratio=0.90,
        native_calls_per_query=10.0,
        peak_memory_mb=120.0,
    )
    candidate = AcceptedMatchMetrics(
        accepted_count=85,
        rejected_count=15,
        mean_residual_norm=0.0008,
        p95_residual_norm=0.0015,
        coverage_ratio=0.95,
        native_calls_per_query=2.5,
        peak_memory_mb=130.0,
    )
    base_latency = LatencySummary(
        median_s=0.400,
        p95_s=0.800,
        min_s=0.200,
        max_s=1.200,
        mean_s=0.450,
        std_s=0.100,
        sample_count=50,
    )
    cand_latency = LatencySummary(
        median_s=0.150,
        p95_s=0.400,
        min_s=0.080,
        max_s=0.600,
        mean_s=0.180,
        std_s=0.050,
        sample_count=50,
    )

    decision = evaluate_promotion_gate(
        baseline_latency=base_latency,
        candidate_latency=cand_latency,
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        min_median_speedup=2.0,
    )
    assert decision.decision == PromotionDecision.PROMOTED
    assert decision.passed is True
    # 0.400 / 0.150 = 2.6667x
    assert decision.median_speedup > 2.0
    assert decision.p95_delta_s < 0.0  # Better (lower) latency
    assert decision.quality_rate_delta >= 0.0  # Non-worse acceptance


def test_promotion_gate_failures() -> None:
    """Promotion gate rejects candidate failing speedup, p95, or quality requirements."""
    base_metrics = AcceptedMatchMetrics(80, 20, 0.001, 0.002, 0.95, 5.0, 100.0)
    cand_metrics = AcceptedMatchMetrics(80, 20, 0.001, 0.002, 0.95, 5.0, 100.0)

    # 1. Speedup too low (1.5x instead of 2.0x)
    base_lat = LatencySummary(0.300, 0.600, 0.1, 1.0, 0.35, 0.05, 50)
    cand_lat_slow = LatencySummary(0.200, 0.500, 0.1, 0.8, 0.25, 0.04, 50)
    decision1 = evaluate_promotion_gate(
        base_lat, cand_lat_slow, base_metrics, cand_metrics
    )
    assert decision1.decision == PromotionDecision.RESEARCH_ONLY
    assert "insufficient median speedup" in decision1.failure_reasons[0].lower()

    # 2. Worse p95 latency
    cand_lat_bad_tail = LatencySummary(0.100, 0.800, 0.05, 1.5, 0.20, 0.15, 50)
    decision2 = evaluate_promotion_gate(
        base_lat, cand_lat_bad_tail, base_metrics, cand_metrics
    )
    assert decision2.decision == PromotionDecision.RESEARCH_ONLY
    assert any("p95 latency regressed" in r.lower() for r in decision2.failure_reasons)

    # 3. Worse accepted quality rate
    cand_metrics_low_acc = AcceptedMatchMetrics(50, 50, 0.001, 0.002, 0.95, 2.0, 100.0)
    cand_lat_fast = LatencySummary(0.100, 0.250, 0.05, 0.4, 0.12, 0.02, 50)
    decision3 = evaluate_promotion_gate(
        base_lat, cand_lat_fast, base_metrics, cand_metrics_low_acc
    )
    assert decision3.decision == PromotionDecision.RESEARCH_ONLY
    assert any(
        "acceptance rate regressed" in r.lower() for r in decision3.failure_reasons
    )


def test_data_efficiency_active_vs_random() -> None:
    """evaluate_data_efficiency compares active vs random acquisition at matched budgets."""
    budgets = [100, 250, 500, 1000]
    active_acc = [0.45, 0.65, 0.82, 0.92]
    random_acc = [0.30, 0.48, 0.62, 0.74]

    curve = evaluate_data_efficiency(
        native_simulation_budgets=budgets,
        active_acquisition_acceptance=active_acc,
        random_acquisition_acceptance=random_acc,
    )
    assert len(curve.budget_points) == 4
    assert curve.sample_efficiency_multiplier > 1.0
    assert curve.active_superiority_confirmed is True


def test_three_seed_uncertainty_aggregation() -> None:
    """Aggregates latency and acceptance variance across three distinct seeds."""
    seed_results = [
        AcceptedMatchMetrics(82, 18, 0.00045, 0.00090, 0.96, 3.2, 140.0),
        AcceptedMatchMetrics(80, 20, 0.00048, 0.00092, 0.95, 3.4, 142.0),
        AcceptedMatchMetrics(84, 16, 0.00043, 0.00088, 0.97, 3.1, 139.0),
    ]
    mean_rate = sum(m.acceptance_rate for m in seed_results) / 3.0
    assert math.isclose(mean_rate, 0.82, rel_tol=1e-2)


def test_model_benchmark_card_cryptographic_digest() -> None:
    """ModelBenchmarkCard produces immutable hash digest identifying the complete test run."""
    timing = TimingBreakdown(0.005, 0.020, 0.010, 0.005, 0.040, 0.015, 0.002)
    card = ModelBenchmarkCard(
        model_id="pendulum_2dof",
        method=BenchmarkMethod.LEARNED_PROPOSAL_POLISH,
        timing=timing,
        latency=LatencySummary(0.097, 0.180, 0.070, 0.250, 0.105, 0.020, 50),
        metrics=AcceptedMatchMetrics(90, 10, 0.0003, 0.0006, 0.98, 2.8, 110.0),
        break_even=compute_break_even(1800.0, 10.0, 0.450, 0.097, 0.002, 0.0005),
        promotion=PromotionDecision.PROMOTED,
    )
    digest = card.compute_digest()
    assert isinstance(digest, str)
    assert len(digest) == 64  # SHA-256 hex string


def test_performance_baseline_integration() -> None:
    """record_neural_motion_benchmark registers neural metrics with PerformanceBaseline."""
    baseline = PerformanceBaseline()
    card = ModelBenchmarkCard(
        model_id="pendulum_2dof",
        method=BenchmarkMethod.LEARNED_PROPOSAL_POLISH,
        timing=TimingBreakdown(0.005, 0.020, 0.010, 0.005, 0.040, 0.015, 0.002),
        latency=LatencySummary(0.097, 0.180, 0.070, 0.250, 0.105, 0.020, 50),
        metrics=AcceptedMatchMetrics(90, 10, 0.0003, 0.0006, 0.98, 2.8, 110.0),
        break_even=compute_break_even(1800.0, 10.0, 0.450, 0.097, 0.002, 0.0005),
        promotion=PromotionDecision.PROMOTED,
    )
    record = record_neural_motion_benchmark(baseline, card)
    assert record["model_id"] == "pendulum_2dof"
    assert record["speedup"] > 1.0
    assert record["status"] == "PROMOTED"


@pytest.mark.unit
def test_runner_without_measured_baseline_or_costs_raises() -> None:
    """Runner must reject missing/derived baselines and unmeasured costs (#10960 P0-5)."""
    samples = [0.01, 0.02, 0.03]
    timing = TimingBreakdown(0.001, 0.002, 0.003, 0.0, 0.004, 0.002, 0.001)
    metrics = AcceptedMatchMetrics(90, 10, 0.001, 0.002, 0.95, 2.0, 100.0)
    base_lat = LatencySummary(0.05, 0.10, 0.02, 0.15, 0.06, 0.01, 50)
    base_met = AcceptedMatchMetrics(80, 20, 0.002, 0.004, 0.90, 3.0, 110.0)

    # Calling with no baseline latency/metrics/costs raises ValueError
    with pytest.raises(
        ValueError,
        match=r"baseline must be measured, not derived from the candidate \(#10960 P0-5\)",
    ):
        run_model_comparative_benchmark(
            model_id="pendulum_2dof",
            method=BenchmarkMethod.LEARNED_PROPOSAL_POLISH,
            interactive_samples_s=samples,
            timing_breakdown=timing,
            metrics=metrics,
        )

    # A baseline with any unmeasured cost also raises ValueError
    with pytest.raises(
        ValueError,
        match=r"baseline must be measured, not derived from the candidate \(#10960 P0-5\)",
    ):
        MeasuredBaseline(
            latency=base_lat,
            metrics=base_met,
            training_wall_time_s=1800.0,
            training_cost_usd=None,  # type: ignore[arg-type]
            baseline_query_cost_usd=0.002,
            candidate_query_cost_usd=0.0005,
        )


@pytest.mark.unit
def test_runner_with_measured_baseline_and_costs_succeeds() -> None:
    """Runner constructs valid ModelBenchmarkCard with measured baseline and costs."""
    samples = [0.05, 0.05, 0.05]
    timing = TimingBreakdown(0.001, 0.002, 0.003, 0.0, 0.004, 0.002, 0.001)
    metrics = AcceptedMatchMetrics(90, 10, 0.0005, 0.001, 0.98, 2.0, 100.0)
    base_lat = LatencySummary(0.20, 0.40, 0.10, 0.60, 0.22, 0.05, 50)
    base_met = AcceptedMatchMetrics(70, 30, 0.001, 0.002, 0.90, 4.0, 120.0)

    card = run_model_comparative_benchmark(
        model_id="pendulum_2dof",
        method=BenchmarkMethod.LEARNED_PROPOSAL_POLISH,
        interactive_samples_s=samples,
        timing_breakdown=timing,
        metrics=metrics,
        baseline=MeasuredBaseline(
            latency=base_lat,
            metrics=base_met,
            training_wall_time_s=1800.0,
            training_cost_usd=10.0,
            baseline_query_cost_usd=0.002,
            candidate_query_cost_usd=0.0005,
        ),
    )
    assert card.model_id == "pendulum_2dof"
    assert card.promotion == PromotionDecision.PROMOTED
    assert card.break_even.has_break_even is True
