"""LIFT-6 (#11746) OpenSim lift API contracts (skip when OpenSim absent)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.lift_scoping.lift6 import (
    assert_evaluate_kinetics_contract,
    assert_opensim_lift_composition_counts,
    host_gated_followups_block,
    validate_lift6_fixture,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "lifting"
    / "lift_scoping"
    / "lift6_epic_v1.json"
)


def test_lift6_fixture_schema() -> None:
    record = json.loads(_FIXTURE.read_bytes())
    assert validate_lift6_fixture(record)["issue"]["id"] == 11746


def test_host_gated_followups_block_lists_host_gates() -> None:
    text = host_gated_followups_block()
    assert "11746" in text and "Tracked follow-up" in text


def _require_lifting_opensim_import_stack() -> None:
    """``lifting`` pulls workspace/matplotlib; CI has them, minimal venvs may not."""
    pytest.importorskip("defusedxml")
    pytest.importorskip("matplotlib")


@pytest.mark.requires_opensim
def test_squat_composition_muscle_count_matches_rajgagopal_fixture() -> None:
    _require_lifting_opensim_import_stack()
    fixture = json.loads(_FIXTURE.read_bytes())
    assert_opensim_lift_composition_counts(fixture=fixture)


@pytest.mark.requires_opensim
def test_evaluate_kinetics_requires_trajectory() -> None:
    _require_lifting_opensim_import_stack()
    fixture = json.loads(_FIXTURE.read_bytes())
    assert_evaluate_kinetics_contract(fixture=fixture)
