"""Parity tests: schema enums vs. code sources of truth (#12324)."""

from __future__ import annotations

from typing import Any

import jsonschema
import pytest

from src.shared.python.engine_core.engine_registry import EngineType
from src.shared.python.estimation.dime_report_integration import (
    ReportForceProvenance,
    ReportScope,
    RunClassification,
)
from src.shared.python.reporting.swing_report.policy import (
    ADVERTISED,
    POLICY,
    WITHHELD,
    QuantityClass,
    classify,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _defs(schema: dict[str, Any]) -> dict[str, Any]:
    return schema["$defs"]


def test_schema_is_valid_draft_2020_12(schema: dict[str, Any]) -> None:
    jsonschema.Draft202012Validator.check_schema(schema)


def test_engine_enum_matches_engine_type(schema: dict[str, Any]) -> None:
    assert set(_defs(schema)["Engine"]["enum"]) == {e.value for e in EngineType}


def test_run_class_scope_force_provenance_match_dime(schema: dict[str, Any]) -> None:
    run = _defs(schema)["EngineRun"]["properties"]
    assert set(run["run_class"]["enum"]) == {e.value for e in RunClassification}
    assert set(run["force_provenance"]["enum"]) == {
        e.value for e in ReportForceProvenance
    }
    assert set(schema["properties"]["scope"]["enum"]) == {e.value for e in ReportScope}


def test_schema_withheld_enum_matches_policy(schema: dict[str, Any]) -> None:
    held = _defs(schema)["WithheldQuantity"]["properties"]["quantity_id"]["enum"]
    assert set(held) == set(WITHHELD)


def test_schema_advertised_enum_matches_policy(schema: dict[str, Any]) -> None:
    adv = _defs(schema)["AdvertisedQuantity"]["properties"]["quantity_id"]["enum"]
    assert set(adv) == set(ADVERTISED)


def test_schema_quantity_catalog_is_policy_keys(schema: dict[str, Any]) -> None:
    assert set(_defs(schema)["QuantityId"]["enum"]) == set(POLICY)


def test_plot_withheld_rule_matches_policy(schema: dict[str, Any]) -> None:
    rule = _defs(schema)["PlotDescriptor"]["allOf"][1]["if"]["properties"][
        "quantity_id"
    ]
    assert set(rule["enum"]) == set(WITHHELD)


def test_policy_sets_are_disjoint_and_complete() -> None:
    assert ADVERTISED.isdisjoint(WITHHELD)
    assert set(POLICY) == ADVERTISED | WITHHELD


def test_v1_policy_decisions() -> None:
    assert {"clubhead_position", "clubhead_speed", "kinematic_sequence"} == ADVERTISED
    assert {"joint_angle", "segment_angular_velocity", "face_angle"} <= WITHHELD


def test_classify_unknown_fails_closed() -> None:
    assert classify("spin_axis") is QuantityClass.WITHHELD
    assert classify("clubhead_speed") is QuantityClass.ADVERTISED


def test_policy_is_immutable() -> None:
    with pytest.raises(TypeError):
        POLICY["face_angle"] = QuantityClass.ADVERTISED  # type: ignore[index]
