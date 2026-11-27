"""Strict JSON reports for the heuristic iron divot model (#15523)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace

import pytest

from src.shared.python.physics.divot_model import DivotResult, SwingArc, compute_divot
from src.shared.python.physics.terrain.materials import MATERIALS
from src.tools.iron_divot_report import DIVOT_REPORT_FORMAT, export_divot_report

pytestmark = pytest.mark.unit


def _result(offset: float = -0.05, height: float = -0.01) -> DivotResult:
    return compute_divot(SwingArc(offset, height), 40.0, MATERIALS["fairway"])


@pytest.mark.parametrize("offset,height", [(-0.05, -0.01), (0.30, -0.01), (0, 0.01)])
def test_model_results_round_trip_through_strict_json(
    offset: float, height: float
) -> None:
    result = _result(offset, height)
    report = export_divot_report(result)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["format"] == DIVOT_REPORT_FORMAT
    assert report["model_status"] == "uncalibrated_heuristic"
    assert report["has_divot"] is result.has_divot
    assert report["is_fat_shot"] is result.is_fat_shot
    for field in ("volume_m3", "energy_loss_j", "speed_retention", "spin_retention"):
        assert report[field] == getattr(result, field)


def test_no_contact_uses_null_offsets_without_mutating_the_result() -> None:
    result = _result(height=0.01)
    report = export_divot_report(result)
    assert report["entry_offset_m"] is None
    assert report["exit_offset_m"] is None
    assert math.isnan(result.entry_offset_m)
    assert math.isnan(result.exit_offset_m)
    assert report["speed_retention"] == report["spin_retention"] == 1.0


def test_contact_preserves_all_model_fields_and_returns_a_fresh_report() -> None:
    result = _result()
    report = export_divot_report(result)
    assert {key: report[key] for key in asdict(result)} == asdict(result)
    report["volume_m3"] = 0.0
    assert export_divot_report(result)["volume_m3"] == result.volume_m3 > 0


@pytest.mark.parametrize("field", ["has_divot", "is_fat_shot"])
def test_flags_must_be_booleans(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        export_divot_report(replace(_result(), **{field: 1}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("entry_offset_m", math.nan),
        ("exit_offset_m", math.inf),
        ("volume_m3", math.nan),
        ("length_m", -1.0),
        ("energy_loss_j", -1.0),
        ("speed_retention", 1.1),
        ("spin_retention", -0.1),
        ("volume_m3", True),
    ],
)
def test_invalid_numbers_are_refused_instead_of_sanitized(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        export_divot_report(replace(_result(), **{field: value}))


def test_missing_geometry_is_not_used_to_hide_invalid_no_contact_data() -> None:
    with pytest.raises(ValueError, match="no-contact"):
        export_divot_report(replace(_result(height=0.01), volume_m3=0.1))


def test_inverted_contact_and_retention_order_are_refused() -> None:
    with pytest.raises(ValueError, match="offset"):
        export_divot_report(replace(_result(), exit_offset_m=-1.0))
    with pytest.raises(ValueError, match="retention"):
        export_divot_report(replace(_result(), speed_retention=0.2, spin_retention=0.8))


def test_wrong_result_type_is_refused() -> None:
    with pytest.raises(TypeError, match="DivotResult"):
        export_divot_report({})  # type: ignore[arg-type]
