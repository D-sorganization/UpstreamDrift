"""Invariant tests for validate_swing_report (#12324, epic #11419, policy #10382)."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from src.shared.python.reporting.swing_report.contract import SwingReport
from src.shared.python.reporting.swing_report.validator import (
    SwingReportContractError,
    validate_swing_report,
)

from .conftest import EXAMPLES_PATH, load_json

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_CASES = load_json(EXAMPLES_PATH)["cases"]
_VALID = [c for c in _CASES if c["valid"]]
_INVALID = [c for c in _CASES if not c["valid"]]


def _codes(doc: Any) -> set[str]:
    with pytest.raises(SwingReportContractError) as info:
        validate_swing_report(doc)
    return {v.code for v in info.value.violations}


@pytest.mark.parametrize("case", _INVALID, ids=lambda c: c["name"])
def test_invalid_examples_fail_with_named_code(case: dict[str, Any]) -> None:
    assert case["expected_code"] in _codes(case["data"])


@pytest.mark.parametrize("case", _VALID, ids=lambda c: c["name"])
def test_valid_examples_pass(case: dict[str, Any]) -> None:
    assert isinstance(validate_swing_report(case["data"]), SwingReport)


def test_every_invariant_has_a_violating_example() -> None:
    names = {c["name"] for c in _INVALID}
    for n in range(1, 13):
        assert any(name.startswith(f"i{n}_") for name in names), f"I{n}"


def test_valid_fixture_files_pass(valid_full: dict, valid_custom: dict) -> None:
    assert validate_swing_report(valid_full).scope == "full"
    assert validate_swing_report(valid_custom).scope == "custom"


def test_nan_value_is_rejected(valid_full: dict) -> None:
    valid_full["quantities"][0]["series"][0]["values"][1] = float("nan")
    assert "I2_NON_FINITE" in _codes(valid_full)


def test_missing_sample_stays_null(valid_full: dict) -> None:
    report = validate_swing_report(valid_full)
    assert report.quantities[0].series[0].values[2] is None


def test_all_violations_are_listed_not_just_first(valid_full: dict) -> None:
    doc = copy.deepcopy(valid_full)
    doc["quantities"][0]["status"] = "validated"
    doc["provenance"]["capture"] = {
        "capture_id": "c",
        "visibility": "private",
        "capture_sha256": "a" * 64,
    }
    assert {"STATUS_VALIDATED", "PRIVATE_CAPTURE_HASH"} <= _codes(doc)


def test_error_message_names_each_code(valid_full: dict) -> None:
    valid_full["quantities"][0]["status"] = "validated"
    with pytest.raises(SwingReportContractError, match="STATUS_VALIDATED"):
        validate_swing_report(valid_full)


@pytest.mark.parametrize("bad", [None, [], "x", 3])
def test_non_mapping_input_is_a_contract_error(bad: Any) -> None:
    assert "SCHEMA" in _codes(bad)


def test_withheld_leak_in_nested_extension_rejected(valid_full: dict) -> None:
    valid_full["extensions"] = {"x-v": {"a": {"face_angle": {"v": [1.0]}}}}
    assert "I9_WITHHELD_LEAK" in _codes(valid_full)


def test_non_numeric_extension_named_like_withheld_is_allowed(valid_full: dict) -> None:
    valid_full["extensions"] = {"x-v": {"face_angle": "see DV-10382"}}
    validate_swing_report(valid_full)


def test_result_round_trips_to_source_document(valid_full: dict) -> None:
    report = validate_swing_report(valid_full)
    assert json.loads(json.dumps(report.to_dict())) == valid_full
