"""Unit tests validating JSON Schema and example fixtures (ADR-0052, #11286)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.shared.python.force_overlay.contracts import ForceTorqueFrame

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

jsonschema = pytest.importorskip("jsonschema")


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def schema(repo_root: Path) -> dict:
    schema_path = repo_root / "schemas" / "force-torque-frame-v1.json"
    assert schema_path.exists(), f"Schema file not found: {schema_path}"
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fixtures(repo_root: Path) -> dict:
    fixtures_path = repo_root / "schemas" / "force-torque-frame-examples.json"
    assert fixtures_path.exists(), f"Examples file not found: {fixtures_path}"
    with open(fixtures_path, encoding="utf-8") as f:
        return json.load(f)


def test_schema_valid_draft_2020_12(schema: dict) -> None:
    """Schema itself must be valid according to Draft 2020-12."""
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)


def test_fixtures_contains_required_named_cases(fixtures: dict) -> None:
    """Fixtures must include at least 7 named cases covering required scenarios."""
    cases = fixtures.get("cases", [])
    assert len(cases) >= 7, f"Expected >= 7 cases, found {len(cases)}"
    names = {c["name"] for c in cases}
    required_names = {
        "torque_only",
        "force_only",
        "both",
        "with_axial_loads",
        "both_null_invalid",
        "duplicate_labels_invalid",
        "nan_invalid",
    }
    missing = required_names - names
    assert not missing, f"Missing required fixture cases: {missing}"


def test_fixtures_conformance(schema: dict, fixtures: dict) -> None:
    """Every case in force-torque-frame-examples.json matches schema and contract expectations."""
    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema)

    for case in fixtures["cases"]:
        name = case["name"]
        data = case["data"]
        is_valid = case["valid"]

        if is_valid:
            # Must pass jsonschema validation
            validator.validate(data)

            # Must parse cleanly into ForceTorqueFrame
            frame = ForceTorqueFrame.from_dict(data)
            assert frame.engine == data["engine"]
            assert frame.time_s == data["time_s"]

            # Must round-trip to dict
            rt_dict = frame.to_dict()
            assert rt_dict["schema_version"] == "force-torque-frame-v1"
            validator.validate(rt_dict)
        else:
            # Either jsonschema validation fails OR ForceTorqueFrame.from_dict fails
            schema_failed = False
            try:
                validator.validate(data)
            except jsonschema.ValidationError:
                schema_failed = True

            contract_failed = False
            try:
                ForceTorqueFrame.from_dict(data)
            except (ValueError, TypeError, KeyError):
                contract_failed = True

            assert schema_failed or contract_failed, (
                f"Invalid case '{name}' was unexpectedly accepted by both schema and contract!"
            )
