"""Inference, evaluation, and promotion contracts for participant twin ensembles."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.participant_twin_calibration import (
    ALIASED_PARAMETERS,
    CALIBRATION_TRAJECTORIES,
    build_benchmark,
    build_synthetic_cohort,
    fit_population,
    screen_identifiability,
)
from scripts.research.proximal_distal_energy.participant_twin_evaluation import (
    build_evidence_record,
    prior_predictive_check,
    run_workflow,
)
from scripts.research.proximal_distal_energy.participant_twin_provenance import (
    build_holdout_barrier,
)

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.scientific
EVIDENCE = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/"
    "participant_twin_calibration.json"
)


@pytest.fixture(scope="module")
def record() -> dict[str, object]:
    return build_evidence_record()


def test_committed_evidence_bundle_reproduces() -> None:
    committed = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert committed == build_evidence_record()


def test_screen_detects_the_structural_alias_before_any_fit() -> None:
    cohort = build_synthetic_cohort()
    benchmark = build_benchmark(cohort)

    screen = screen_identifiability(
        benchmark.design,
        benchmark.discrepancy,
        benchmark.noise_sd,
        calibration_trajectory_count=CALIBRATION_TRAJECTORIES,
    )

    assert screen.structural_rank < screen.parameter_count
    assert set(ALIASED_PARAMETERS) in [
        set(group) for group in screen.aliased_directions
    ]
    assert set(ALIASED_PARAMETERS).isdisjoint(screen.estimable_parameters)


def test_fit_refuses_a_screen_from_a_different_design() -> None:
    cohort = build_synthetic_cohort()
    benchmark = build_benchmark(cohort)
    other = build_benchmark(cohort, include_activation=False)
    foreign = screen_identifiability(
        other.design,
        other.discrepancy,
        other.noise_sd,
        calibration_trajectory_count=CALIBRATION_TRAJECTORIES,
    )

    with pytest.raises(ValueError, match="does not match this design"):
        fit_population(benchmark, cohort, build_holdout_barrier(cohort), foreign)


def test_calibration_never_reads_a_held_out_outcome(
    record: dict[str, object],
) -> None:
    ledger = record["calibration_access_ledger"]

    cohort = build_synthetic_cohort()
    calibration_total = sum(
        role == "calibration" for role in cohort["split"]["trajectory_roles"].values()
    )

    assert ledger["held_out_outcomes_read"] == 0
    assert ledger["calibration_outcomes_read"] == calibration_total
    assert (
        record["evaluation_ledger"]["gates"][
            "calibration_never_reads_held_out_outcomes"
        ]
        == "satisfied"
    )


@pytest.mark.parametrize("protocol", ["trajectory_held_out", "participant_held_out"])
def test_calibrated_twin_beats_both_declared_baselines(
    record: dict[str, object], protocol: str
) -> None:
    arms = record["held_out_evaluation"][protocol]

    assert (
        arms["calibrated_twin"]["normalized_rmse"]
        < arms["population_baseline"]["normalized_rmse"]
    )
    assert (
        arms["calibrated_twin"]["normalized_rmse"]
        < arms["uncalibrated_baseline"]["normalized_rmse"]
    )
    assert arms["calibrated_twin"]["observation_count"] > 0


def test_contraction_is_reported_separately_and_stays_low_for_the_alias(
    record: dict[str, object],
) -> None:
    contraction = record["posterior_contraction"]

    assert "prediction" not in contraction
    assert contraction["participant_twin"]["grip_compliance_alias"] < 0.2
    assert contraction["participant_twin"]["shaft_stiffness_scale"] > 0.5
    assert (
        "not evidence of out-of-sample predictive skill"
        in (contraction["reporting_boundary"])
    )


def test_only_screened_parameters_are_reported_as_recovered(
    record: dict[str, object],
) -> None:
    recovery = record["parameter_recovery"]

    assert set(ALIASED_PARAMETERS).issubset(recovery["withheld_parameters"])
    assert set(recovery["absolute_error_rms"]) == set(recovery["estimable_parameters"])


def test_omitting_the_discrepancy_term_biases_the_twin_parameters(
    record: dict[str, object],
) -> None:
    ablation = record["discrepancy_ablation"]
    included = ablation["with_discrepancy_term"]["parameter_absolute_error_rms"]
    omitted = ablation["without_discrepancy_term"]["parameter_absolute_error_rms"]

    assert set(included) == set(omitted)
    assert any(omitted[name] > 2.0 * included[name] for name in included)
    assert (
        ablation["without_discrepancy_term"]["posterior_predictive_coverage_90"]
        < ablation["with_discrepancy_term"]["posterior_predictive_coverage_90"]
    )


def test_dropping_the_optional_activation_channel_removes_its_identifiability() -> None:
    without = run_workflow(include_activation=False)

    screen = without["identifiability_screen"]
    assert "activation_gain" not in screen["structurally_identifiable"]
    assert "activation_gain" not in screen["estimable_parameters"]
    assert "activation" not in [channel["name"] for channel in without["channels"]]


def test_prior_predictive_check_fails_when_the_prior_cannot_generate_the_data() -> None:
    cohort = build_synthetic_cohort()
    benchmark = build_benchmark(cohort)
    inflated = dataclasses.replace(
        benchmark,
        observations={
            key: 10.0 * value for key, value in benchmark.observations.items()
        },
    )

    assert prior_predictive_check(benchmark, cohort)["status"] == "passed"
    assert prior_predictive_check(inflated, cohort)["status"] == "failed"


def test_transport_audit_retains_nontransportable_and_small_cells(
    record: dict[str, object],
) -> None:
    audit = record["transport_audit"]

    assert audit["nontransportable_cells"]
    statuses = {
        cell["transport_status"]
        for levels in audit["strata"].values()
        for cell in levels.values()
    }
    assert "nontransportable_unrepresented_in_training" in statuses
    assert "retained" in audit["retention_policy"]


def test_ledger_never_promotes_a_human_or_coaching_conclusion(
    record: dict[str, object],
) -> None:
    ledger = record["evaluation_ledger"]

    assert ledger["gates"]["private_governed_authority_contract"] == (
        "blocked_no_governed_participant_data"
    )
    assert ledger["promotion_decision"] == "synthetic_benchmark_qualified"
    assert ledger["human_promotion"] == "blocked_no_governed_participant_data"
    assert ledger["boundaries"]["human_digital_twin"] == "untested"
    assert ledger["boundaries"]["coaching_inference"] == "unsupported"
    assert ledger["boundaries"]["anatomical_or_muscle_strategy"] == "not_identified"
    assert "cannot" in record["inference_boundary"].lower()


def test_reviewer_documentation_lists_every_promotion_gate(
    record: dict[str, object],
) -> None:
    contract = (
        ROOT / "docs/research/proximal_distal_energy_transfer/"
        "PARTICIPANT_TWIN_CALIBRATION.md"
    ).read_text(encoding="utf-8")

    for gate in record["evaluation_ledger"]["gates"]:
        assert f"`{gate}`" in contract, gate
