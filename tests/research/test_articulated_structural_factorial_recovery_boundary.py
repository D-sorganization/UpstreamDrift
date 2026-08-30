"""The recovery boundary cross-binds every operational gate without outcomes."""

from __future__ import annotations

import copy
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.research.proximal_distal_energy.articulated_structural_factorial_recovery_boundary import (
    BOUNDARY_SCHEMA,
    audit_recovery_boundary,
)

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _authority() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return tuple(
        json.loads((DATA / name).read_text(encoding="utf-8"))
        for name in (
            "articulated_structural_factorial_plan.json",
            "articulated_structural_factorial_launch.json",
            "articulated_structural_factorial_recovery_registration.json",
        )
    )  # type: ignore[return-value]


def _evidence() -> dict[str, dict[str, object]]:
    _, launch, _ = _authority()
    plan_sha = launch["plan_sha256"]
    execution = launch["execution_revision"]
    receipt_sha = "a" * 64
    return {
        "runtime": {
            "schema_version": (
                "articulated-structural-factorial-runtime-replay-audit/1.0.0"
            ),
            "classification": "runtime_replay_contract_exact",
            "gates": {"passes": True},
            "claim_boundary": {"scientific_outcomes_inspected": False},
        },
        "receipt": {
            "schema_version": (
                "articulated-structural-factorial-artifact-receipt/1.5.0"
            ),
            "execution_revision": execution,
            "requested_case_range": [0, 20],
            "run": {
                "id": 123,
                "status": "completed",
                "conclusion": "success",
                "head_sha": "d" * 40,
            },
            "job": {"status": "completed", "conclusion": "success"},
            "runtime_replay_artifact": {
                "id": 201,
                "name": "structural-runtime-replay-123",
                "size_in_bytes": 200,
                "digest": f"sha256:{'f' * 64}",
                "workflow_run": {"id": 123, "head_sha": "d" * 40},
            },
            "runtime_replay_archive_sha256": "f" * 64,
        },
        "corruption": {
            "schema_version": (
                "articulated-structural-factorial-corruption-audit/1.0.0"
            ),
            "identity": {
                "plan_sha256": plan_sha,
                "execution_revision": execution,
            },
            "sentinel": {"passes": True, "source_checkpoint_unchanged": True},
        },
        "collection": {
            "schema_version": "articulated-structural-factorial-collection/1.4.0",
            "classification": "execution_collection_not_scientific_summary",
            "plan_sha256": plan_sha,
            "execution_revision": execution,
            "combined_checkpoint_count": 20,
            "next_missing_case_index": 20,
            "sources": [
                {
                    "run_id": 123,
                    "requested_case_range": [0, 20],
                    "observed_case_range": [0, 20],
                    "run_conclusion": "success",
                    "artifact_receipt_schema": (
                        "articulated-structural-factorial-artifact-receipt/1.5.0"
                    ),
                    "artifact_receipt_sha256": receipt_sha,
                    "runtime_replay_artifact": {
                        "id": 201,
                        "name": "structural-runtime-replay-123",
                        "size_in_bytes": 200,
                        "digest": f"sha256:{'f' * 64}",
                        "workflow_run": {"id": 123, "head_sha": "d" * 40},
                    },
                    "runtime_replay_archive_sha256": "f" * 64,
                }
            ],
        },
        "legacy": {
            "schema_version": ("articulated-structural-factorial-prefix-view/1.0.0"),
            "classification": "operational_prefix_view_not_scientific_summary",
            "prefix_case_stop_exclusive": 20,
            "source_checkpoint_count": 714,
        },
        "enrichment": {
            "schema_version": (
                "articulated-structural-factorial-enrichment-audit/1.0.0"
            ),
            "identity": {
                "enriched_plan_sha256": plan_sha,
                "enriched_execution_revision": execution,
                "legacy_prefix_count": 20,
            },
            "gates": {
                "passes": True,
                "status_reproduction_exact": True,
                "completed_json_reproduction_exact": True,
                "legacy_array_reproduction_exact": True,
                "complete_evidence_sidecars_valid": True,
            },
            "claim_boundary": {
                "scientific_outcomes_inspected": False,
                "human_or_coaching_inference": False,
            },
        },
    }


def _audit(evidence: dict[str, dict[str, object]]) -> dict[str, object]:
    plan, launch, registration = _authority()
    return audit_recovery_boundary(
        plan=plan,
        launch=launch,
        registration=registration,
        runtime_replay_audit=evidence["runtime"],
        artifact_receipt=evidence["receipt"],
        artifact_receipt_sha256="a" * 64,
        corruption_audit=evidence["corruption"],
        collection_manifest=evidence["collection"],
        legacy_prefix_manifest=evidence["legacy"],
        enrichment_audit=evidence["enrichment"],
        case_start=0,
        case_stop=20,
        input_sha256=dict.fromkeys(evidence, "b" * 64),
    )


def test_recovery_boundary_cross_binds_all_exact_gates() -> None:
    result = _audit(_evidence())

    gates = cast(dict[str, object], result["gates"])
    identity = cast(dict[str, object], result["identity"])
    next_slice = cast(dict[str, object], result["next_slice"])
    claim_boundary = cast(dict[str, object], result["claim_boundary"])
    assert result["schema_version"] == BOUNDARY_SCHEMA
    assert result["classification"] == "attested_prefix_boundary_exact"
    assert gates["passes"] is True
    assert identity["run_id"] == 123
    assert next_slice["authorized"] is False
    assert claim_boundary["scientific_outcomes_inspected"] is False


@pytest.mark.parametrize(
    ("section", "mutation", "message"),
    [
        (
            "receipt",
            lambda value: value.update(
                schema_version=(
                    "articulated-structural-factorial-artifact-receipt/1.3.0"
                )
            ),
            "receipt 1.5",
        ),
        (
            "collection",
            lambda value: value["sources"][0].update(requested_case_range=[1, 20]),
            "gap-free",
        ),
        (
            "collection",
            lambda value: value["sources"][0].update(
                runtime_replay_archive_sha256="e" * 64
            ),
            "runtime replay",
        ),
        (
            "enrichment",
            lambda value: value["gates"].update(passes=False),
            "enrichment audit",
        ),
        (
            "runtime",
            lambda value: value["claim_boundary"].update(
                scientific_outcomes_inspected=True
            ),
            "outcome-blind",
        ),
    ],
)
def test_recovery_boundary_rejects_any_weakened_evidence(
    section: str,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    evidence = copy.deepcopy(_evidence())
    mutation(evidence[section])

    with pytest.raises(ValueError, match=message):
        _audit(evidence)
