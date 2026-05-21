"""Comprehensive tests for programmatic_pid.validation module.

Tests collect_issues, validate_spec, validate_spec_json.
"""

from __future__ import annotations

import pytest
from programmatic_pid.types import SpecValidationError
from programmatic_pid.validation import (
    collect_issues,
    validate_spec,
    validate_spec_json,
)

VALID_SPEC: dict = {
    "project": {"id": "P001", "title": "Test PID"},
    "equipment": [
        {"id": "V101", "w": 10.0, "h": 10.0},
        {"id": "P101", "w": 5.0, "h": 5.0},
    ],
    "instruments": [{"id": "FT101"}],
    "streams": [
        {"id": "S1", "from": {"equipment": "V101"}, "to": {"equipment": "P101"}},
    ],
    "control_loops": [
        {"id": "FC101", "measurement": "FT101", "final_element": "V101"},
    ],
}


class TestCollectIssues:
    def test_valid_spec_has_no_issues(self):
        issues = collect_issues(VALID_SPEC)
        errors = [i for i in issues if i.severity == "error"]
        assert errors == []

    def test_non_dict_spec_returns_issue(self):
        issues = collect_issues("not a dict")
        assert len(issues) == 1
        assert "mapping" in issues[0].message.lower()

    def test_missing_project_id(self):
        spec = dict(VALID_SPEC)
        spec["project"] = {"title": "Test"}
        spec["equipment"] = VALID_SPEC["equipment"]
        issues = collect_issues(spec)
        paths = [i.path for i in issues]
        assert "project.id" in paths

    def test_missing_project_title(self):
        spec = {
            "project": {"id": "P001"},
            "equipment": VALID_SPEC["equipment"],
        }
        issues = collect_issues(spec)
        paths = [i.path for i in issues]
        assert "project.title" in paths

    def test_empty_equipment_list(self):
        spec = {
            "project": {"id": "P001", "title": "T"},
            "equipment": [],
        }
        issues = collect_issues(spec)
        paths = [i.path for i in issues]
        assert "equipment" in paths

    def test_equipment_missing_id(self):
        spec = {
            "project": {"id": "P001", "title": "T"},
            "equipment": [{"w": 5.0, "h": 5.0}],  # no id
        }
        issues = collect_issues(spec)
        paths = [i.path for i in issues]
        assert any("equipment[0].id" in p for p in paths)

    def test_duplicate_equipment_id(self):
        spec = {
            "project": {"id": "P001", "title": "T"},
            "equipment": [
                {"id": "V101", "w": 5.0, "h": 5.0},
                {"id": "V101", "w": 5.0, "h": 5.0},  # duplicate
            ],
        }
        issues = collect_issues(spec)
        messages = [i.message for i in issues]
        assert any("duplicate" in m for m in messages)

    def test_non_positive_width_height(self):
        spec = {
            "project": {"id": "P001", "title": "T"},
            "equipment": [{"id": "V101", "w": 0.0, "h": 5.0}],
        }
        issues = collect_issues(spec)
        messages = [i.message for i in issues]
        assert any("non-positive" in m for m in messages)

    def test_instrument_missing_id(self):
        spec = {
            "project": {"id": "P001", "title": "T"},
            "equipment": [{"id": "V101", "w": 5.0, "h": 5.0}],
            "instruments": [{}],  # no id
        }
        issues = collect_issues(spec)
        paths = [i.path for i in issues]
        assert any("instruments[0].id" in p for p in paths)

    def test_duplicate_instrument_id(self):
        spec = {
            "project": {"id": "P001", "title": "T"},
            "equipment": [{"id": "V101", "w": 5.0, "h": 5.0}],
            "instruments": [{"id": "FT101"}, {"id": "FT101"}],
        }
        issues = collect_issues(spec)
        messages = [i.message for i in issues]
        assert any("duplicate" in m for m in messages)

    def test_stream_unknown_from_equipment(self):
        spec = {
            "project": {"id": "P001", "title": "T"},
            "equipment": [{"id": "V101", "w": 5.0, "h": 5.0}],
            "streams": [{"id": "S1", "from": {"equipment": "UNKNOWN"}}],
        }
        issues = collect_issues(spec)
        messages = [i.message for i in issues]
        assert any("unknown from equipment" in m for m in messages)

    def test_stream_unknown_to_equipment(self):
        spec = {
            "project": {"id": "P001", "title": "T"},
            "equipment": [{"id": "V101", "w": 5.0, "h": 5.0}],
            "streams": [{"id": "S1", "to": {"equipment": "UNKNOWN"}}],
        }
        issues = collect_issues(spec)
        messages = [i.message for i in issues]
        assert any("unknown to equipment" in m for m in messages)

    def test_control_loop_missing_measurement(self):
        spec = {
            "project": {"id": "P001", "title": "T"},
            "equipment": [{"id": "V101", "w": 5.0, "h": 5.0}],
            "control_loops": [{"id": "FC101", "final_element": "V101"}],
        }
        issues = collect_issues(spec)
        messages = [i.message for i in issues]
        assert any("missing measurement" in m for m in messages)

    def test_control_loop_unknown_measurement(self):
        spec = {
            "project": {"id": "P001", "title": "T"},
            "equipment": [{"id": "V101", "w": 5.0, "h": 5.0}],
            "control_loops": [
                {"id": "FC101", "measurement": "UNKNOWN", "final_element": "V101"}
            ],
        }
        issues = collect_issues(spec)
        messages = [i.message for i in issues]
        assert any("unknown measurement reference" in m for m in messages)

    def test_control_loop_unknown_final_element(self):
        spec = {
            "project": {"id": "P001", "title": "T"},
            "equipment": [{"id": "V101", "w": 5.0, "h": 5.0}],
            "instruments": [{"id": "FT101"}],
            "control_loops": [
                {"id": "FC101", "measurement": "FT101", "final_element": "UNKNOWN"}
            ],
        }
        issues = collect_issues(spec)
        messages = [i.message for i in issues]
        assert any("unknown final element" in m for m in messages)

    def test_stream_no_from_or_to_key(self):
        """Streams without from/to keys should not cause issues."""
        spec = {
            "project": {"id": "P001", "title": "T"},
            "equipment": [{"id": "V101", "w": 5.0, "h": 5.0}],
            "streams": [{"id": "S1"}],  # no from/to
        }
        issues = collect_issues(spec)
        errors = [i for i in issues if i.severity == "error"]
        assert errors == []


class TestValidateSpec:
    def test_valid_spec_passes(self):
        validate_spec(VALID_SPEC)  # Should not raise

    def test_invalid_spec_raises(self):
        with pytest.raises(SpecValidationError, match="Invalid spec"):
            validate_spec({"project": {}, "equipment": []})

    def test_non_dict_raises(self):
        with pytest.raises(SpecValidationError):
            validate_spec("not a dict")


class TestValidateSpecJson:
    def test_valid_spec_returns_empty_list(self):
        result = validate_spec_json(VALID_SPEC)
        errors = [r for r in result if r["severity"] == "error"]
        assert errors == []

    def test_invalid_spec_returns_structured_issues(self):
        result = validate_spec_json({"project": {}, "equipment": []})
        assert isinstance(result, list)
        assert len(result) > 0
        assert all("path" in r and "message" in r and "severity" in r for r in result)

    def test_never_raises(self):
        # Should not raise even for completely invalid input
        result = validate_spec_json(None)
        assert isinstance(result, list)
