"""OSL-15 (#13093) scoping: bar-load balance in lift IK provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.osl_scoping.osl15 import (
    assert_ik_records_load_balance,
    asset_gated_followups_block,
    deadlift_pull_bundle_paths,
    validate_osl15_fixture,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE = _ROOT / "tests/fixtures/lifting/osl_scoping/osl15_scoping_v1.json"
_BACK_SQUAT = (
    _ROOT / "docs/development/lifting/evidence/back_squat/back_squat_barbell_bundle.npz"
)
_BACK_SQUAT_SOURCE = (
    _ROOT / "docs/development/lifting/evidence/back_squat/back_squat_bundle.npz"
)


def test_osl15_fixture_schema() -> None:
    record = json.loads(_FIXTURE.read_bytes())
    assert validate_osl15_fixture(record)["issue"]["id"] == 13093


def test_back_squat_barbell_bundle_records_load_balance() -> None:
    fixture = json.loads(_FIXTURE.read_bytes())
    assert_ik_records_load_balance(_BACK_SQUAT, fixture=fixture)


def test_back_squat_source_bundle_records_load_balance() -> None:
    fixture = json.loads(_FIXTURE.read_bytes())
    assert_ik_records_load_balance(_BACK_SQUAT_SOURCE, fixture=fixture)


def test_asset_gated_followups_block_lists_regeneration() -> None:
    block = asset_gated_followups_block()
    assert "Tracked follow-up" in block and "13093" in block


@pytest.mark.parametrize(
    "path",
    deadlift_pull_bundle_paths(_ROOT),
    ids=["deadlift_barbell", "clean_and_jerk_barbell"],
)
@pytest.mark.xfail(
    strict=True,
    reason="OSL-15 (#13093): regenerate deadlift and clean barbell bundles with ik.balance",
)
def test_pull_barbell_bundles_record_load_balance(path: Path) -> None:
    fixture = json.loads(_FIXTURE.read_bytes())
    assert_ik_records_load_balance(path, fixture=fixture)
