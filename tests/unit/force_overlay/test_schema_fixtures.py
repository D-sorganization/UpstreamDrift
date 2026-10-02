"""Validate force-torque-frame-examples.json fixtures against ForceTorqueFrame contract and JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.shared.python.force_overlay.contracts import ForceTorqueFrame

pytestmark = pytest.mark.unit


def test_schema_fixtures_contract_behavior():
    fixtures_path = (
        Path(__file__).resolve().parents[3]
        / "schemas"
        / "force-torque-frame-examples.json"
    )
    with open(fixtures_path, encoding="utf-8") as f:
        catalog = json.load(f)

    cases = catalog["cases"]
    assert len(cases) >= 7

    for case in cases:
        name = case["name"]
        data = case["data"]
        is_valid = case["valid"]
        if is_valid:
            frame = ForceTorqueFrame.from_dict(data)
            assert frame.engine == data["engine"], f"Case {name} engine mismatch"
            assert frame.time_s == data["time_s"], f"Case {name} time mismatch"
        else:
            with pytest.raises(Exception, match=case["expected_error"]):
                ForceTorqueFrame.from_dict(data)
