"""LIFT-5 (#11745) epic evidence contracts (paths and parity JSON only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.lift_scoping.lift5 import (
    assert_back_squat_parity_record,
    assert_committed_lift_bundles_exist,
    load_back_squat_bar_path_parity_record,
    validate_lift5_fixture,
)

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "lifting"
    / "lift_scoping"
    / "lift5_epic_v1.json"
)


def test_lift5_fixture_schema() -> None:
    record = json.loads(_FIXTURE.read_bytes())
    assert validate_lift5_fixture(record)["issue"]["id"] == 11745


def test_all_five_committed_bundles_exist() -> None:
    assert_committed_lift_bundles_exist()


def test_back_squat_bar_path_parity_covers_all_four_engines() -> None:
    record = load_back_squat_bar_path_parity_record()
    fixture = json.loads(_FIXTURE.read_bytes())
    assert_back_squat_parity_record(record, fixture=fixture)


def test_back_squat_bar_path_parity_default_engines_include_opensim() -> None:
    record = load_back_squat_bar_path_parity_record()
    assert_back_squat_parity_record(record)
