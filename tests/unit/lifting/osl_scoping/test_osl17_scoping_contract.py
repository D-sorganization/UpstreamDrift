"""OSL-17 (#13131) scoping: rack provenance and window lead-in."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.osl_scoping.osl17 import (
    assert_high_bar_rack_in_ik,
    assert_lead_in_duration,
    assert_lead_in_record_present,
    asset_gated_followups_block,
    validate_osl17_fixture,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE = _ROOT / "tests/fixtures/lifting/osl_scoping/osl17_scoping_v1.json"
_SQUAT_BB = (
    _ROOT / "docs/development/lifting/evidence/back_squat/back_squat_barbell_bundle.npz"
)


def test_osl17_fixture_schema() -> None:
    record = json.loads(_FIXTURE.read_bytes())
    assert validate_osl17_fixture(record)["issue"]["id"] == 13131


def test_back_squat_barbell_records_high_bar_rack() -> None:
    fixture = json.loads(_FIXTURE.read_bytes())
    assert_high_bar_rack_in_ik(_SQUAT_BB, fixture=fixture)


def test_back_squat_barbell_has_lead_in_provenance() -> None:
    assert_lead_in_record_present(_SQUAT_BB)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OSL-17 (#13131): extend lead_in_s beyond 0.1 s (fixture min 0.25 s) until "
        "OpenSim Hicks gate is clean at window start."
    ),
)
def test_back_squat_barbell_lead_in_meets_extended_duration() -> None:
    fixture = json.loads(_FIXTURE.read_bytes())
    assert_lead_in_duration(_SQUAT_BB, fixture=fixture)


def test_asset_gated_followups_lists_receipts() -> None:
    body = asset_gated_followups_block()
    assert "13131" in body and "Tracked follow-up" in body
