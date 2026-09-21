"""NM-01 (#10616): benefit experiment freeze, gates and break-even."""

from __future__ import annotations

import math

import pytest

from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
)
from src.shared.python.neural_motion.experiment import (
    BASELINE_METHODS,
    LATENCY_PHASES,
    NESTED_EPISODE_STAGES,
    BenefitExperimentSpec,
    BenchmarkCaseKind,
    ComputeBudgetCaps,
    PromotionGates,
    break_even_queries,
    freeze_benefit_experiment,
    default_benefit_experiment,
)

pytestmark = pytest.mark.unit


def test_default_experiment_preregisters_baselines_and_nested_stages() -> None:
    spec = default_benefit_experiment()
    assert isinstance(spec, BenefitExperimentSpec)
    assert spec.nested_episode_stages == NESTED_EPISODE_STAGES
    assert NESTED_EPISODE_STAGES == (100, 500, 2000)
    assert len(spec.training_seeds) == 3
    assert set(spec.baseline_methods) == set(BASELINE_METHODS)
    assert {
        "cold_classical",
        "retrieval_plus_solve",
        "existing_networks_plus_solve",
        "forward_surrogate_plus_polish",
        "inverse_proposal_plus_polish",
    } <= set(spec.baseline_methods)


def test_experiment_requires_synthetic_stratum_coverage_and_external_targets() -> None:
    spec = default_benefit_experiment()
    assert spec.min_synthetic_queries_per_stratum >= 30
    assert tuple(spec.workbook_trials) == CANONICAL_TRIAL_SHEETS
    assert len(spec.workbook_trials) == 4
    assert len(spec.c3d_derived_tests) == 2
    assert "tiny empirical set" in spec.statistical_limitations.lower()


def test_latency_accounting_covers_all_phases_including_failures() -> None:
    spec = default_benefit_experiment()
    assert set(spec.latency_phases) == set(LATENCY_PHASES)
    assert "independent_replay" in spec.latency_phases
    assert "physical_refinement" in spec.latency_phases
    assert "rejected_attempts" in spec.latency_phases
    kinds = {case.kind for case in spec.benchmark_cases}
    assert BenchmarkCaseKind.REPLAY in kinds
    assert BenchmarkCaseKind.REFINEMENT in kinds
    assert BenchmarkCaseKind.FAILURE in kinds


def test_promotion_gates_match_reviewed_2x_median_and_nonworse_p95() -> None:
    gates = default_benefit_experiment().promotion_gates
    assert isinstance(gates, PromotionGates)
    assert gates.min_median_speedup == 2.0
    assert gates.p95_must_not_worsen is True
    assert gates.accepted_quality_rate_must_not_degrade is True
    assert gates.negative_result_blocks_product_promotion is True
    assert gates.negative_result_retains_research_checkpoint is True


def test_compute_caps_require_timing_probe_before_training() -> None:
    with pytest.raises(ValueError, match="timing_probe"):
        ComputeBudgetCaps(
            max_cpu_core_hours=10.0,
            max_gpu_hours=1.0,
            max_storage_gib=20.0,
            max_wall_time_hours=8.0,
            timing_probe_completed=False,
            probe_receipt_id=None,
        )
    caps = ComputeBudgetCaps(
        max_cpu_core_hours=10.0,
        max_gpu_hours=1.0,
        max_storage_gib=20.0,
        max_wall_time_hours=8.0,
        timing_probe_completed=True,
        probe_receipt_id="probe.nm01.v1",
    )
    assert caps.timing_probe_completed is True


def test_break_even_rejects_nonpositive_savings() -> None:
    assert break_even_queries(offline_cost_s=100.0, per_query_savings_s=0.0) is None
    assert break_even_queries(offline_cost_s=100.0, per_query_savings_s=-1.0) is None
    with pytest.raises(ValueError, match="finite|non-negative|offline"):
        break_even_queries(offline_cost_s=math.nan, per_query_savings_s=1.0)
    n = break_even_queries(offline_cost_s=100.0, per_query_savings_s=2.0)
    assert n == pytest.approx(50.0)


def test_freeze_emits_stable_split_and_gate_digests() -> None:
    receipt = freeze_benefit_experiment(default_benefit_experiment())
    assert receipt.schema == "neural-benefit-experiment/1.0.0"
    assert len(receipt.split_digest) == 64
    assert len(receipt.gate_digest) == 64
    assert receipt.split_digest != receipt.gate_digest
    again = freeze_benefit_experiment(default_benefit_experiment())
    assert again.split_digest == receipt.split_digest
    assert again.gate_digest == receipt.gate_digest
    assert again.content_digest == receipt.content_digest
