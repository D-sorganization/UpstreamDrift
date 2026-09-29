"""Evidence contracts for the executable transmission-robustness study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_transmission_robustness_study import (
    PERTURBATION_NAMES,
    PROGRAM_NAMES,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _load_generated_study() -> tuple[dict, dict[str, np.ndarray]]:
    """Load the deterministic artifact produced by the separately tested runner."""

    record = json.loads((DATA_DIR / "transmission_robustness_study.json").read_text())
    with np.load(DATA_DIR / record["array_artifact"], allow_pickle=False) as artifact:
        arrays = {name: artifact[name].copy() for name in artifact.files}
    return record, arrays


def test_study_is_paired_held_out_and_reports_all_registered_programs() -> None:
    record, arrays = _load_generated_study()

    assert record["registered_before_preferred_result"] is True
    assert record["design"]["paired_common_random_numbers"] is True
    assert record["design"]["held_out_perturbations"] > 0
    assert tuple(record["programs"]) == PROGRAM_NAMES
    assert tuple(record["perturbation_names"]) == PERTURBATION_NAMES
    assert arrays["training_perturbations"].shape[1] == len(PERTURBATION_NAMES)
    assert arrays["held_out_perturbations"].shape[1] == len(PERTURBATION_NAMES)


def test_study_exposes_speed_variability_and_no_universal_optimum() -> None:
    record, arrays = _load_generated_study()

    assert len(record["held_out_pareto_programs"]) >= 2
    assert record["claim_status"]["universal_optimum"] == "rejected_by_tradeoffs"
    assert record["claim_status"]["human_self_stabilization"] == "untested"
    assert record["claim_status"]["nominal_speed_implies_repeatability"] == "rejected"
    assert np.all(np.isfinite(arrays["held_out_outcomes"]))
    assert arrays["held_out_outcomes"].shape[0] == len(PROGRAM_NAMES)


def test_study_has_pathway_closure_and_local_task_variance_partition() -> None:
    record, arrays = _load_generated_study()

    # The projected 4 ms integrator carries a declared numerical work--energy
    # residual; only the algebraic contact-power identity is machine-precision.
    assert record["closure"]["maximum_normalized_pathway_residual"] < 0.05
    assert record["closure"]["maximum_contact_power_residual_w"] < 1e-10
    assert record["local_task_map"]["nullity"] >= 1
    assert -1.0 <= record["local_task_map"]["synergy_index"] <= 1.0
    assert arrays["local_outcome_jacobian"].shape[0] >= 2
    assert arrays["local_outcome_jacobian"].shape[1] >= 3


def test_adversarial_register_contains_counterexamples_and_required_next_tests() -> (
    None
):
    record, _ = _load_generated_study()

    gaps = record["adversarial_gap_register"]
    assert len(gaps) >= 10
    assert {gap["status"] for gap in gaps} >= {"confirmed", "open"}
    assert all(gap["falsifier"] and gap["path_forward"] for gap in gaps)
    assert any("negative torque" in gap["counterexample"].lower() for gap in gaps)


pytestmark = pytest.mark.scientific
