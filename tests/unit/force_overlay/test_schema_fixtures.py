"""Test JSON schema and example fixtures for force-torque-frame-v1."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.shared.python.force_overlay.contracts import ForceTorqueFrame

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "schemas" / "force-torque-frame-v1.json"
EXAMPLES_PATH = ROOT / "schemas" / "force-torque-frame-examples.json"


def test_schema_and_fixtures_exist():
    assert SCHEMA_PATH.is_file(), f"Missing schema at {SCHEMA_PATH}"
    assert EXAMPLES_PATH.is_file(), f"Missing examples at {EXAMPLES_PATH}"


def test_examples_against_schema_and_contracts():
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))

    validator = jsonschema.Draft202012Validator(schema)

    # Must contain at least the 7 named test cases
    expected_cases = {
        "valid_torque_only",
        "valid_force_only",
        "valid_both_halves",
        "valid_with_axial_loads",
        "invalid_both_halves_null",
        "invalid_duplicate_labels",
        "invalid_nan",
    }
    case_names = {case["name"] for case in examples["cases"]}
    missing = expected_cases - case_names
    assert not missing, f"Missing required fixture cases: {missing}"

    for case in examples["cases"]:
        name = case["name"]
        payload = case["data"]
        is_valid = case["is_valid"]

        schema_errors = list(validator.iter_errors(payload))
        if is_valid:
            assert not schema_errors, (
                f"Valid case {name} failed schema: {schema_errors}"
            )
            # Also test contract from_dict
            frame = ForceTorqueFrame.from_dict(payload)
            assert frame.engine == payload["engine"]
            roundtrip = frame.to_dict()
            assert roundtrip["schema_version"] == "force-torque-frame-v1"
        else:
            # Either schema rejects or contract from_dict rejects
            contract_rejected = False
            try:
                ForceTorqueFrame.from_dict(payload)
            except (ValueError, TypeError):
                contract_rejected = True

            assert bool(schema_errors) or contract_rejected, (
                f"Invalid case {name} was accepted by both schema and contract"
            )
