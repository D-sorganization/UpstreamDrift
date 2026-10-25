"""HRG-E4 (#13175): display view of the anatomical event table.

The desktop widget and the web table both render the strict-JSON export
(``event_table.to_json_dict``). One golden fixture, shared with the web
vitest suite, pins the document and the exact cell text, so both surfaces show
identical values for the same bundle.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from src.shared.python.anatomical_angles import event_table as et
from src.shared.python.anatomical_angles import event_table_view as etv
from src.shared.python.anatomical_angles.wrist import (
    ForearmAngles,
    WristAngles,
    WristForearm,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = (
    ROOT
    / "ui"
    / "src"
    / "components"
    / "analysis"
    / "anatomicalEventTable"
    / "fixtures"
    / "eventTableView.json"
)
N = 4
TIME = np.array([0.0, 0.1234, 0.25, 0.4])
EVENTS = {"top": 1, "impact": 2}
SOURCE = "swing_comparison.events.detect_events"
RECEIPT = "receipts/fit/receipt.json"
MISSING_REASON = "marker occluded"


def _reasons(*names: str) -> dict[str, np.ndarray]:
    return {n: np.full(N, "", dtype=object) for n in names}


def _arm(side: str, offset: float) -> WristForearm:
    base = np.arange(N, dtype=float) * 1.37
    rad = [np.deg2rad(offset - 15.5 * k + base) for k in range(3)]
    wrist_reasons = _reasons("flexion", "ulnar_deviation", "axial_rotation")
    if side == "R":
        rad[1][2] = np.nan
        wrist_reasons["ulnar_deviation"][2] = MISSING_REASON
    wrist = WristAngles(
        rad[0],
        rad[1],
        rad[2],
        wrist_reasons,
        side,
        RECEIPT,
        (f"cal/{side}/radius", f"cal/{side}/hand"),
    )
    forearm = ForearmAngles(
        np.zeros(N),
        np.zeros(N),
        np.deg2rad(offset + 30.25 + base),
        _reasons("flexion", "carrying_angle", "pronation"),
        side,
        RECEIPT,
        (f"cal/{side}/humerus", f"cal/{side}/forearm"),
    )
    return WristForearm(wrist, forearm)


def build_document() -> dict:
    """The export document the golden fixture was generated from."""
    table = et.anatomical_event_table(
        {"L": _arm("L", 12.0), "R": _arm("R", -20.0)},
        EVENTS,
        TIME,
        handedness="right",
        event_source=SOURCE,
    )
    return et.to_json_dict(table)


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_document_is_the_shared_export_of_the_bundle(fixture) -> None:
    # json round-trip: the fixture stores what the API serialises.
    assert fixture["document"] == json.loads(json.dumps(build_document()))
    jsonschema.validate(fixture["document"], et.EVENT_TABLE_JSON_SCHEMA)


def test_view_matches_the_golden_cells_shared_with_the_web(fixture) -> None:
    assert etv.event_table_view(fixture["document"]).as_dict() == fixture["view"]


def test_one_row_per_export_row_in_export_order(fixture) -> None:
    doc = fixture["document"]
    view = etv.event_table_view(doc)
    assert view.columns == etv.COLUMNS
    assert len(view.rows) == len(doc["rows"])
    assert [(r[0], r[3]) for r in view.rows] == [
        (r["event"], r["term_id"]) for r in doc["rows"]
    ]


def test_missing_value_is_shown_as_missing_with_its_reason(fixture) -> None:
    view = etv.event_table_view(fixture["document"])
    (row,) = [
        r
        for r in view.rows
        if r[0] == "impact" and r[3] == "trail_wrist_ulnar_deviation"
    ]
    value = row[etv.COLUMNS.index("Value (deg)")]
    note = row[etv.COLUMNS.index("Note")]
    assert value == etv.MISSING_TEXT
    assert note == MISSING_REASON


def test_side_cell_names_role_and_body_side(fixture) -> None:
    view = etv.event_table_view(fixture["document"])
    sides = {r[etv.COLUMNS.index("Side")] for r in view.rows}
    assert sides == {"lead (L)", "trail (R)"}


def test_caption_carries_source_handedness_units_and_reference(fixture) -> None:
    doc = fixture["document"]
    caption = etv.event_table_view(doc).caption
    for part in (doc["event_source"], "right-handed", "deg", doc["reference"]):
        assert part in caption


def test_fixed_formatting_matches_the_web_rule(fixture) -> None:
    for value, digits, text in fixture["format_cases"]:
        assert etv.format_fixed(value, digits) == text, (value, digits)


def test_format_cases_cover_ties_signs_and_negative_zero(fixture) -> None:
    values = [c[0] for c in fixture["format_cases"]]
    assert 0.125 in values and -0.125 in values
    assert any(v == 0 and math.copysign(1, v) < 0 for v in values)


def test_non_finite_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        etv.format_fixed(math.inf, 2)


def test_document_with_wrong_schema_is_rejected(fixture) -> None:
    doc = dict(fixture["document"], schema="something.else")
    with pytest.raises(ValueError, match="schema"):
        etv.event_table_view(doc)


def test_unknown_term_in_a_row_is_rejected(fixture) -> None:
    doc = json.loads(json.dumps(fixture["document"]))
    doc["rows"][0]["term_id"] = "not_a_term"
    with pytest.raises(ValueError, match="not_a_term"):
        etv.event_table_view(doc)
