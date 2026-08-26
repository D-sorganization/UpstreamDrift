"""Contracts for the UpstreamDrift engineering design-manual authority."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.check_design_manual_governance import (
    DesignManualGovernanceError,
    verify_calculation_registry,
    verify_governance_policy,
    verify_repository,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "config" / "design_manual_governance.json"
REGISTRY_PATH = REPO_ROOT / "manuals" / "upstreamdrift" / "calculation-registry.json"


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_repository_adopts_one_qmd_authority_and_blocks_release() -> None:
    summary = verify_repository(REPO_ROOT)

    assert summary.manual_id == "upstreamdrift"
    assert summary.canonical_qmd_count == 2
    assert summary.calculation_count == 0
    assert summary.release_status == "blocked-inventory-required"
    assert summary.public_projection_allowed is False


def test_policy_reuses_program_contracts_without_copying_schemas() -> None:
    policy = _json(POLICY_PATH)
    verify_governance_policy(policy)

    assert policy["contracts"]["owner_repository"] == (
        "D-sorganization/Engineering-Design-Manuals"
    )
    assert not (REPO_ROOT / "schemas" / "calculation-registry.schema.json").exists()
    assert not (REPO_ROOT / "schemas" / "publication-projection.schema.json").exists()


def test_registry_is_fail_closed_until_up_d1_inventory() -> None:
    registry = _json(REGISTRY_PATH)

    assert verify_calculation_registry(registry) == 0
    assert registry["release_status"] == "blocked-inventory-required"
    assert registry["inventory_commit"] is None
    assert registry["blockers"][0]["id"] == "UP-D1-inventory-required"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda policy: policy["generated_outputs"].update(editable=True),
            "generated outputs must not be editable",
        ),
        (
            lambda policy: policy["publication"].update(public_projection_allowed=True),
            "public projection must remain blocked",
        ),
        (
            lambda policy: policy["freshness"].update(enforcement="advisory"),
            "freshness enforcement",
        ),
        (
            lambda policy: policy["quality"].update(python_formatter="black"),
            "python formatter",
        ),
    ],
)
def test_policy_rejects_authority_drift(mutation: Any, message: str) -> None:
    policy = copy.deepcopy(_json(POLICY_PATH))
    mutation(policy)

    with pytest.raises(DesignManualGovernanceError, match=message):
        verify_governance_policy(policy)


def test_policy_rejects_unknown_fields_and_unsafe_paths() -> None:
    policy = copy.deepcopy(_json(POLICY_PATH))
    policy["alternate_authority"] = "legacy-tex"
    with pytest.raises(DesignManualGovernanceError, match="fields differ"):
        verify_governance_policy(policy)

    policy = copy.deepcopy(_json(POLICY_PATH))
    policy["canonical_source"]["path"] = "../outside"
    with pytest.raises(DesignManualGovernanceError, match="normalized relative path"):
        verify_governance_policy(policy)


def test_empty_registry_cannot_claim_approval() -> None:
    registry = copy.deepcopy(_json(REGISTRY_PATH))
    registry["release_status"] = "approved"
    registry["inventory_commit"] = "1" * 40
    registry["blockers"] = []

    with pytest.raises(DesignManualGovernanceError, match="approved registry requires"):
        verify_calculation_registry(registry)


def test_agent_context_exposes_update_and_artifact_rules() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    claude = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    spec = (REPO_ROOT / "SPEC.md").read_text(encoding="utf-8")
    handoff = (REPO_ROOT / "AGENT_HANDOFF.md").read_text(encoding="utf-8")

    for text in (agents, claude):
        normalized = " ".join(text.lower().split())
        assert "manuals/upstreamdrift" in text
        assert "scripts.check_design_manual_governance" in text
        assert "generated latex, pdf, docx, and html" in normalized
    assert "UP-D0 (#9066)" in spec
    assert "UP-D0 (#9066)" in handoff
    assert "UP-D1 (#9067)" in handoff


def test_manual_tree_contains_no_editable_release_artifacts() -> None:
    manual_root = REPO_ROOT / "manuals" / "upstreamdrift"
    forbidden_suffixes = {".docx", ".html", ".pdf", ".tex"}
    found = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in manual_root.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    )

    assert found == []
