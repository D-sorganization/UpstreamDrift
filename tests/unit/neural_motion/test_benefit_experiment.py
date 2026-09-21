"""NM-01 (#10616): frozen benefit experiment, splits, gates and break-even.

Acceptance cases from the issue:
- freeze benchmark splits and gate digests
- native accuracy/feasibility gates align with TB-02/#10587 profiles where declared
- hardware and all-phase latency accounting
- pre-register classical / retrieval / existing-network / forward-surrogate /
  proposed-inverse baselines
- pilot scales: 100/500/2000 episodes, three seeds, ≥30 synthetic queries per
  stratum, four workbook trials, two C3D tests, statistical limitations recorded
- no positive break-even if per-query savings <= 0
- benchmark accounting includes replay, refinement and failures
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.shared.python.neural_motion.experiment import (
    BENEFIT_EXPERIMENT_SCHEMA,
    REQUIRED_BASELINES,
    REQUIRED_LATENCY_PHASES,
    BaselineKind,
    BenefitExperimentSpec,
    BreakEvenResult,
    LatencyPhase,
    PilotScale,
    PromotionGates,
    compute_break_even,
    default_benefit_experiment,
    freeze_digest,
)
from src.shared.python.tour_baselines.qualification_profiles import (
    QUALIFICATION_PROFILE_VERSION,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "neural_motion_matching"
    / "evidence"
    / "nm01_benefit_experiment.json"
)


def test_required_baselines_are_pre_registered() -> None:
    expected = {
        BaselineKind.COLD_CLASSICAL,
        BaselineKind.RETRIEVAL_PLUS_SOLVE,
        BaselineKind.EXISTING_NETWORKS_PLUS_SOLVE,
        BaselineKind.FORWARD_SURROGATE_PLUS_POLISH,
        BaselineKind.PROPOSED_INVERSE_PLUS_POLISH,
    }
    assert set(REQUIRED_BASELINES) == expected
    spec = default_benefit_experiment()
    assert set(spec.baselines) == expected


def test_all_phase_latency_accounting_includes_replay_refinement_failures() -> None:
    required = {
        LatencyPhase.STARTUP,
        LatencyPhase.PREPROCESSING,
        LatencyPhase.CANDIDATE_GENERATION,
        LatencyPhase.NEURAL_INFERENCE,
        LatencyPhase.PHYSICAL_REFINEMENT,
        LatencyPhase.REJECTED_ATTEMPTS,
        LatencyPhase.INDEPENDENT_REPLAY,
    }
    assert set(REQUIRED_LATENCY_PHASES) == required
    spec = default_benefit_experiment()
    assert set(spec.latency_phases) == required


def test_pilot_scale_and_statistical_limitations_frozen() -> None:
    scale = PilotScale()
    assert scale.episode_stages == (100, 500, 2000)
    assert scale.n_seeds == 3
    assert scale.min_synthetic_queries_per_stratum == 30
    assert scale.n_workbook_trials == 4
    assert scale.n_c3d_tests == 2
    assert "tiny empirical" in scale.statistical_limitations.lower()


def test_promotion_gates_are_proposed_not_measured() -> None:
    gates = PromotionGates()
    assert gates.median_speedup_min == 2.0
    assert gates.p95_nonworse is True
    assert gates.accepted_quality_rate_nonworse is True
    assert gates.is_measured_outcome is False


def test_break_even_none_when_savings_nonpositive() -> None:
    assert compute_break_even(offline_cost_s=3600.0, per_query_savings_s=0.0) is None
    assert compute_break_even(offline_cost_s=3600.0, per_query_savings_s=-1.0) is None
    result = compute_break_even(offline_cost_s=3600.0, per_query_savings_s=12.0)
    assert isinstance(result, BreakEvenResult)
    assert result.queries_to_break_even == pytest.approx(300.0)
    assert result.has_positive_break_even is True


def test_no_positive_break_even_flag_when_savings_le_zero() -> None:
    spec = default_benefit_experiment()
    verdict = spec.evaluate_break_even(offline_cost_s=100.0, per_query_savings_s=0.0)
    assert verdict.has_positive_break_even is False
    assert verdict.queries_to_break_even is None


def test_frozen_split_and_gate_digest_stable() -> None:
    spec = default_benefit_experiment()
    digest = freeze_digest(spec)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
    # Digests are over the payload excluding digest fields themselves.
    payload_body = {
        key: value
        for key, value in spec.as_dict().items()
        if key not in {"split_digest", "gate_digest"}
    }
    expected = hashlib.sha256(
        json.dumps(payload_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert digest == expected
    assert spec.split_digest == digest
    assert spec.gate_digest == digest


def test_qualification_gate_alignment_declares_tb02_profile_version() -> None:
    spec = default_benefit_experiment()
    assert spec.qualification_profile_version == QUALIFICATION_PROFILE_VERSION
    assert "10587" in spec.qualification_alignment_note or "TB-02" in (
        spec.qualification_alignment_note
    )


def test_benefit_experiment_schema_and_evidence_receipt() -> None:
    spec = default_benefit_experiment()
    assert spec.schema == BENEFIT_EXPERIMENT_SCHEMA
    assert EVIDENCE.is_file(), "NM-01 must freeze a versioned evidence receipt"
    on_disk = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert on_disk["schema"] == BENEFIT_EXPERIMENT_SCHEMA
    assert on_disk["split_digest"] == spec.split_digest
    assert set(on_disk["baselines"]) == {b.value for b in REQUIRED_BASELINES}


def test_incomplete_baselines_rejected() -> None:
    with pytest.raises(ValueError, match="baselines"):
        BenefitExperimentSpec(
            schema=BENEFIT_EXPERIMENT_SCHEMA,
            baselines=(BaselineKind.COLD_CLASSICAL,),
            latency_phases=REQUIRED_LATENCY_PHASES,
            pilot_scale=PilotScale(),
            promotion_gates=PromotionGates(),
            split_policy="by_source_trial_seed_geometry_contact",
            hardware_targets=("cpu_cold", "cpu_warm", "gpu_if_available"),
            qualification_profile_version=QUALIFICATION_PROFILE_VERSION,
            qualification_alignment_note="TB-02/#10587",
            compute_caps_pending_timing_probe=True,
        )
