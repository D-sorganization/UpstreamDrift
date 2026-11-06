"""OSL-14 (#13057) scoping contracts: lift-body spec and bundle provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.osl_scoping.osl14 import (
    bundle_spec_matches_committed_lift_body_document,
    bundle_structurally_matches_lift_body,
    committed_lift_body_spec,
    issue_todo_block,
    lift_body_matches_osl14_acceptance,
    validate_osl14_fixture,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE = _ROOT / "tests" / "fixtures" / "lifting" / "osl_scoping" / "osl14_scoping_v1.json"
_BACK_SQUAT_BUNDLE = _ROOT / "docs/development/lifting/evidence/back_squat/back_squat_bundle.npz"


def test_osl14_fixture_schema() -> None:
    record = json.loads(_FIXTURE.read_bytes())
    assert validate_osl14_fixture(record)["issue"]["id"] == 13057


def test_committed_lift_body_matches_osl14_structural_acceptance() -> None:
    spec = committed_lift_body_spec()
    fixture = json.loads(_FIXTURE.read_bytes())
    lift_body_matches_osl14_acceptance(spec, fixture=fixture)


def test_back_squat_bundle_matches_lift_body_invariants() -> None:
    bundle_structurally_matches_lift_body(_BACK_SQUAT_BUNDLE)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OSL-14 (#13057): regenerate back_squat_bundle.npz so spec bytes hash "
        "to the committed lift_body_spec.json (until then prov sha != file sha)."
    ),
)
def test_back_squat_bundle_embedded_spec_matches_committed_document_bytes() -> None:
    bundle_spec_matches_committed_lift_body_document(_BACK_SQUAT_BUNDLE)


def test_issue_todo_block_lists_asset_gates() -> None:
    block = issue_todo_block()
    assert "TODO:" in block
    assert "OpenSim" in block
    assert "13057" in block
