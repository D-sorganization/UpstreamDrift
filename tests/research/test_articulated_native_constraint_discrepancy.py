from __future__ import annotations

import copy
import hashlib
import json
import numpy as np
import pytest
from pathlib import Path

from scripts.research.proximal_distal_energy.articulated_native_constraint_discrepancy import (
    NativeConstraintDiscrepancyConfig,
    run_native_constraint_discrepancy,
)
from scripts.research.proximal_distal_energy import (
    register_articulated_native_constraint_discrepancy_claims as claim_registration,
)

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def test_native_claim_registration_is_idempotent() -> None:
    registry = json.loads((DATA / "claim_audit_registry.json").read_text("utf-8"))
    inventory = json.loads((DATA / "claim_candidate_inventory.json").read_text("utf-8"))

    for _ in range(2):
        claims, selected = claim_registration._build_claims(inventory["candidates"])
        claim_registration._reconcile(
            registry,
            inventory,
            copy.deepcopy(claims),
            selected,
        )

    for review in registry["candidate_reviews"]:
        assert len(review["claim_ids"]) == len(set(review["claim_ids"]))
    reviews = {
        review["candidate_id"]: review for review in registry["candidate_reviews"]
    }
    candidates = {
        candidate["candidate_id"]: candidate for candidate in inventory["candidates"]
    }
    for claim in registry["claims"]:
        assert len(claim["candidate_ids"]) == len(claim["source_locations"])
        for candidate_id, source_location in zip(
            claim["candidate_ids"], claim["source_locations"], strict=True
        ):
            assert claim["claim_id"] in reviews[candidate_id]["claim_ids"]
            candidate = candidates[candidate_id]
            assert source_location == (
                f"{candidate['source_path']}:{candidate['line_start']}"
            )


def test_native_constraint_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        NativeConstraintDiscrepancyConfig(duration_s=0.0)
    with pytest.raises(ValueError, match="time_steps_s"):
        NativeConstraintDiscrepancyConfig(time_steps_s=(0.00025, 0.0005))
    with pytest.raises(ValueError, match="initial_club_displacement_m"):
        NativeConstraintDiscrepancyConfig(initial_club_displacement_m=0.0)
    with pytest.raises(ValueError, match="contact_stiffness"):
        NativeConstraintDiscrepancyConfig(contact_stiffness=np.inf)


def test_native_branch_uses_mujoco_constraint_solver_and_integrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mujoco

    native_step = mujoco.mj_step
    call_count = 0

    def counted_step(model: object, data: object) -> None:
        nonlocal call_count
        call_count += 1
        native_step(model, data)

    monkeypatch.setattr(mujoco, "mj_step", counted_step)
    config = NativeConstraintDiscrepancyConfig(
        duration_s=0.0005,
        time_steps_s=(0.0005,),
    )
    record, arrays = run_native_constraint_discrepancy(config)

    assert call_count == 2  # active equality plus equality-disabled killswitch
    assert record["schema_version"] == "articulated-native-constraint-discrepancy/v1"
    assert record["native_branch"]["constraint_type"] == "connect"
    assert record["native_branch"]["constraint_count"] == 2
    assert record["native_branch"]["minimum_constraint_row_count"] >= 6
    assert record["native_branch"]["integrator_operator"] == "mj_step"
    assert arrays["native_generalized_constraint_force_n"].shape[-1] == 20


def test_native_and_projected_branches_are_nontrivial_and_distinct() -> None:
    record, arrays = run_native_constraint_discrepancy(
        NativeConstraintDiscrepancyConfig(
            duration_s=0.001,
            time_steps_s=(0.0005, 0.00025),
        )
    )
    results = record["results"]

    assert results["maximum_native_generalized_constraint_force"] > 0.0
    assert results["maximum_projected_generalized_contact_force"] > 0.0
    assert results["maximum_trajectory_absolute_discrepancy"] > 0.0
    assert results["maximum_killswitch_generalized_constraint_force"] <= 1.0e-12
    assert results["initial_state_absolute_discrepancy"] <= 1.0e-15
    assert results["native_initial_attachment_separation_m"] == pytest.approx(
        0.001, abs=1.0e-10
    )
    assert results["all_registered_gates_passed"] is True
    assert record["claim_boundary"]["human_transfer_or_strategy"] == "untested"
    assert record["claim_boundary"]["engine_equivalence"] == "not_claimed"

    for key, values in arrays.items():
        if values.dtype.kind in "fc":
            assert np.all(np.isfinite(values)), key


def test_committed_native_constraint_evidence_is_current_and_bounded() -> None:
    record = json.loads(
        (DATA / "articulated_native_constraint_discrepancy.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["schema_version"] == ("articulated-native-constraint-discrepancy/v1")
    assert record["classification"] == (
        "synthetic_formulation_discrepancy_not_human_or_engine_equivalence_evidence"
    )
    assert record["native_branch"]["constraint_count"] == 2
    assert record["native_branch"]["minimum_constraint_row_count"] >= 6
    assert record["results"]["all_registered_gates_passed"] is True
    assert record["results"]["maximum_trajectory_absolute_discrepancy"] > 0.0
    for relative_path, expected in record["source_sha256"].items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected

    with np.load(DATA / "articulated_native_constraint_discrepancy.npz") as arrays:
        assert arrays["native_q"].shape == arrays["projected_q"].shape
        assert arrays["native_generalized_constraint_force_n"].shape[1] == 20
        assert np.max(np.abs(arrays["native_q"] - arrays["projected_q"])) > 0.0
        for key in arrays.files:
            if arrays[key].dtype.kind in "fc":
                assert np.all(np.isfinite(arrays[key])), key
