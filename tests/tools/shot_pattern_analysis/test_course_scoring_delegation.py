"""Characterize the public course report before its per-row helper refactor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.shot_pattern_analysis.scenario_scoring import score_course_endpoints

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).parent / "data" / "legacy_course_reports.json"


def test_delegation_preserves_all_saved_report_fields_and_numeric_values() -> None:
    fixture = json.loads(_FIXTURE.read_text())
    assert len(fixture["cases"]) == 6
    for case in fixture["cases"]:
        endpoints = {
            name: [tuple(row) for row in shots]
            for name, shots in case["endpoints"].items()
        }
        assert score_course_endpoints(endpoints, **case["options"]) == case["report"]


def test_delegation_preserves_invalid_and_unsupported_course_errors() -> None:
    fixture = json.loads(_FIXTURE.read_text())
    assert len(fixture["errors"]) == 3
    for case in fixture["errors"]:
        endpoints = {
            name: [tuple(row) for row in shots]
            for name, shots in case["endpoints"].items()
        }
        with pytest.raises(ValueError) as exc_info:
            score_course_endpoints(endpoints, **case["options"])
        assert str(exc_info.value) == case["error"]
