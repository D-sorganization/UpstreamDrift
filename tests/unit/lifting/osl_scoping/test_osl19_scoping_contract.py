"""OSL-19 (#13158) scoping: grip closure vs OpenSim reach receipt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.osl_scoping.osl19 import (
    assert_bundle_carries_opensim_reach_receipt,
    assert_mujoco_grip_closure_passes,
    issue_todo_block,
    validate_osl19_fixture,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE = _ROOT / "tests/fixtures/lifting/osl_scoping/osl19_scoping_v1.json"
_SQUAT_BB = _ROOT / "docs/development/lifting/evidence/back_squat/back_squat_barbell_bundle.npz"


def test_osl19_fixture_schema() -> None:
    record = json.loads(_FIXTURE.read_bytes())
    assert validate_osl19_fixture(record)["issue"]["id"] == 13158


def test_back_squat_barbell_bundle_mujoco_grip_under_1mm() -> None:
    fixture = json.loads(_FIXTURE.read_bytes())
    assert_mujoco_grip_closure_passes(_SQUAT_BB, fixture=fixture)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OSL-19 (#13158): OpenSim bar-reach receipt absent from bundle provenance "
        "until the OpenSim solve matches MuJoCo grips (add opensim_bar_reach_receipt_v1)."
    ),
)
def test_back_squat_barbell_bundle_has_opensim_reach_receipt() -> None:
    assert_bundle_carries_opensim_reach_receipt(_SQUAT_BB)


def test_issue_todo_calls_out_os_reach() -> None:
    body = issue_todo_block()
    assert "OpenSim" in body and "13158" in body
