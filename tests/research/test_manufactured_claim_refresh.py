"""Provenance refresh must preserve independently reviewed claim evidence."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research.proximal_distal_energy import (
    register_articulated_manufactured_solution_claims as registration,
)

pytestmark = [pytest.mark.unit]


def test_refresh_preserves_outcomes_numeric_evidence_order_and_inputs() -> None:
    reviewed = {
        "claim_id": "PD-CLAIM-299",
        "statement": "bounded manufactured conservation",
        "adjudication_outcome": "supported",
        "numeric_evidence": [{"path": "free_body.energy", "tolerance": 0.02}],
        "evidence_artifacts": ["old-record.json"],
        "last_verified_on": "2026-08-21",
    }
    unrelated = {"claim_id": "PD-CLAIM-1", "statement": "separate claim"}
    original = [reviewed, unrelated]
    snapshot = deepcopy(original)
    update = {
        "claim_id": "PD-CLAIM-299",
        "statement": reviewed["statement"],
        "evidence_artifacts": ["current-record.json"],
        "last_verified_on": "2026-09-08",
    }
    result = registration._refresh_claims(original, [update])
    assert [item["claim_id"] for item in result] == ["PD-CLAIM-299", "PD-CLAIM-1"]
    assert result[0]["adjudication_outcome"] == "supported"
    assert result[0]["numeric_evidence"] == reviewed["numeric_evidence"]
    assert result[0]["evidence_artifacts"] == ["current-record.json"]
    assert result[0]["last_verified_on"] == "2026-09-08"
    assert result[1] == unrelated
    assert original == snapshot


@pytest.mark.parametrize("field", ["statement", "model_domain", "falsifier"])
def test_changed_scientific_claim_requires_new_review(field: str) -> None:
    original = {"claim_id": "PD-CLAIM-297", field: "reviewed meaning"}
    update = {"claim_id": "PD-CLAIM-297", field: "different meaning"}
    with pytest.raises(ValueError, match="scientific.*review"):
        registration._refresh_claims([original], [update])


def test_initial_registration_never_invents_adjudication() -> None:
    original = [{"claim_id": "PD-CLAIM-1", "statement": "separate"}]
    new = {"claim_id": "PD-CLAIM-297", "statement": "new manufactured claim"}
    result = registration._refresh_claims(original, [new])
    assert result == [*original, new]
    assert "adjudication_outcome" not in result[1]


def _registration_inputs() -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]
]:
    registry = json.loads(registration.REGISTRY.read_text(encoding="utf-8"))
    inventory = json.loads(registration.INVENTORY.read_text(encoding="utf-8"))
    claims, selected = registration._build_claims(inventory["candidates"])
    return registry, inventory, claims, selected


def _use_temporary_registration(
    inputs: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    registry, inventory = inputs
    output, source = tmp_path / "registry.json", tmp_path / "inventory.json"
    output.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    source.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(registration, "REGISTRY", output)
    monkeypatch.setattr(registration, "INVENTORY", source)
    return output


def test_reconcile_preserves_reviewed_fields_and_registry_order() -> None:
    registry, inventory, claims, selected = _registration_inputs()
    before = deepcopy(registry)
    inputs = deepcopy((inventory, claims, selected))
    registration._reconcile(registry, inventory, claims, selected)
    assert [c["claim_id"] for c in registry["claims"]] == [
        c["claim_id"] for c in before["claims"]
    ]
    refreshed = {claim["claim_id"]: claim for claim in registry["claims"]}
    for original in before["claims"]:
        for key, value in original.items():
            if key not in {"evidence_artifacts", "last_verified_on"}:
                assert refreshed[original["claim_id"]][key] == value
    assert (inventory, claims, selected) == inputs


@pytest.mark.parametrize("field", ["statement", "model_domain", "falsifier"])
def test_actual_reconciliation_refuses_scientific_change_before_mutation(
    field: str,
) -> None:
    registry, inventory, claims, selected = _registration_inputs()
    before = deepcopy(registry)
    claims[0][field] = "a different unreviewed scientific assertion"
    with pytest.raises(ValueError, match="scientific.*review"):
        registration._reconcile(registry, inventory, claims, selected)
    assert registry == before


def test_registration_entry_point_preserves_evidence_and_is_byte_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, inventory, _, _ = _registration_inputs()
    output = _use_temporary_registration((registry, inventory), tmp_path, monkeypatch)
    registration.main()
    first = output.read_bytes()
    registration.main()
    assert output.read_bytes() == first
    actual = {c["claim_id"]: c for c in json.loads(first)["claims"]}
    for original in registry["claims"]:
        assert (
            actual[original["claim_id"]]["adjudication_outcome"]
            == original["adjudication_outcome"]
        )
        if "numeric_evidence" in original:
            assert (
                actual[original["claim_id"]]["numeric_evidence"]
                == original["numeric_evidence"]
            )


def test_registration_failure_does_not_overwrite_the_persisted_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, inventory, _, _ = _registration_inputs()
    target = next(c for c in registry["claims"] if c["claim_id"] == "PD-CLAIM-297")
    target["statement"] = "a reviewed statement that differs from the proposed update"
    output = _use_temporary_registration((registry, inventory), tmp_path, monkeypatch)
    before = output.read_bytes()
    with pytest.raises(ValueError, match="scientific.*review"):
        registration.main()
    assert output.read_bytes() == before
