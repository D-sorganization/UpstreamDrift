"""HRG-E4 (#13175): event table and CSV/JSON export of anatomical angles."""

from __future__ import annotations

import csv
import io
import json
import math

import numpy as np
import pytest

from src.shared.python.anatomical_angles import OBSERVABILITY
from src.shared.python.anatomical_angles import event_table as et
from src.shared.python.anatomical_angles.terms import wrist_term_specs
from src.shared.python.anatomical_angles.wrist import (
    ForearmAngles,
    WristAngles,
    WristForearm,
)

pytestmark = pytest.mark.unit

RECEIPT = "receipts/fit/receipt.json"
N = 6
TIME = np.linspace(0.0, 0.5, N)
EVENTS = {"address": 0, "top": 2, "impact": 4, "finish": 5}
SOURCE = "swing_comparison.events.detect_events"


def _empty_reasons(*names: str) -> dict[str, np.ndarray]:
    return {n: np.full(N, "", dtype=object) for n in names}


def _arm(side: str, offset: float) -> WristForearm:
    """Known per-sample values: component k at frame i = offset + 10 k + i deg."""
    base = np.arange(N, dtype=float)
    rad = [np.deg2rad(offset + 10.0 * k + base) for k in range(3)]
    wrist_reasons = _empty_reasons("flexion", "ulnar_deviation", "axial_rotation")
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
        np.deg2rad(offset + 30.0 + base),
        _empty_reasons("flexion", "carrying_angle", "pronation"),
        side,
        RECEIPT,
        (f"cal/{side}/humerus", f"cal/{side}/forearm"),
    )
    return WristForearm(wrist, forearm)


def _table(**kw: object) -> et.AnatomicalEventTable:
    args: dict[str, object] = {
        "arms": {"L": _arm("L", 100.0), "R": _arm("R", 200.0)},
        "events": EVENTS,
        "time_s": TIME,
        "handedness": "right",
        "event_source": SOURCE,
    }
    args.update(kw)
    return et.anatomical_event_table(**args)  # type: ignore[arg-type]


def _row(table: et.AnatomicalEventTable, event: str, term: str) -> et.EventRow:
    (row,) = [r for r in table.rows if r.event == event and r.term_id == term]
    return row


def test_rows_cover_every_event_and_registered_term_in_order() -> None:
    table = _table()
    term_ids = [s.term_id for s in wrist_term_specs()]
    expected = [(e, t) for e in EVENTS for t in term_ids]
    assert [(r.event, r.term_id) for r in table.rows] == expected


def test_values_are_sampled_at_event_frames_for_lead_and_trail() -> None:
    table = _table()
    # Right-handed: lead = L (offset 100), trail = R (offset 200).
    top = EVENTS["top"]
    assert _row(table, "top", "lead_wrist_flexion_extension").value_deg == (
        pytest.approx(100.0 + top)
    )
    assert _row(table, "top", "lead_wrist_ulnar_deviation").value_deg == (
        pytest.approx(110.0 + top)
    )
    assert _row(table, "impact", "trail_wrist_axial_rotation").value_deg == (
        pytest.approx(220.0 + EVENTS["impact"])
    )
    assert _row(table, "finish", "trail_forearm_pronation").value_deg == (
        pytest.approx(230.0 + EVENTS["finish"])
    )
    row = _row(table, "impact", "lead_forearm_pronation")
    assert (row.side, row.role, row.frame) == ("L", "lead", EVENTS["impact"])
    assert row.time_s == pytest.approx(TIME[EVENTS["impact"]])


def test_left_handed_golfer_swaps_lead_side() -> None:
    table = _table(handedness="left")
    row = _row(table, "address", "lead_wrist_flexion_extension")
    assert row.side == "R"
    assert row.value_deg == pytest.approx(200.0)


def test_every_row_carries_provenance_and_observability() -> None:
    for row in _table().rows:
        assert row.observability == OBSERVABILITY
        assert row.receipt_path == RECEIPT
        assert len(row.calibration) == 2
        assert row.calibration[0].startswith(f"cal/{row.side}/")


def test_missing_arm_omits_its_role_not_invented() -> None:
    table = _table(arms={"L": _arm("L", 100.0)})
    assert {r.role for r in table.rows} == {"lead"}


def test_singular_sample_exports_reason_not_a_number() -> None:
    arm = _arm("L", 100.0)
    flex = arm.wrist.flexion_rad.copy()
    flex[EVENTS["top"]] = np.nan
    reasons = dict(arm.wrist.reasons)
    why = np.full(N, "", dtype=object)
    why[EVENTS["top"]] = "gimbal lock: flexion indeterminate"
    reasons["flexion"] = why
    wrist = WristAngles(
        flex,
        arm.wrist.ulnar_deviation_rad,
        arm.wrist.axial_rotation_rad,
        reasons,
        "L",
        RECEIPT,
        arm.wrist.calibration,
    )
    table = _table(arms={"L": WristForearm(wrist, arm.forearm)})
    row = _row(table, "top", "lead_wrist_flexion_extension")
    assert math.isnan(row.value_deg)
    assert row.invalid_reason == "gimbal lock: flexion indeterminate"
    doc = et.to_json_dict(table)
    (jrow,) = [
        r
        for r in doc["rows"]
        if r["event"] == "top" and r["term_id"] == "lead_wrist_flexion_extension"
    ]
    assert jrow["value_deg"] is None
    assert jrow["invalid_reason"] == "gimbal lock: flexion indeterminate"
    json.dumps(doc, allow_nan=False)  # strict JSON: no NaN literal


def test_nan_without_reason_still_gets_a_reason() -> None:
    arm = _arm("L", 100.0)
    flex = arm.wrist.flexion_rad.copy()
    flex[0] = np.nan
    wrist = WristAngles(
        flex,
        arm.wrist.ulnar_deviation_rad,
        arm.wrist.axial_rotation_rad,
        arm.wrist.reasons,
        "L",
        RECEIPT,
        arm.wrist.calibration,
    )
    table = _table(arms={"L": WristForearm(wrist, arm.forearm)})
    assert _row(table, "address", "lead_wrist_flexion_extension").invalid_reason


@pytest.mark.parametrize(
    ("events", "match"),
    [
        ({"address": 0, "kickoff": 1}, "unknown event"),
        ({"pelvis_reversal": 1}, "detected"),
        ({"address": 3, "top": 2}, "order"),
        ({"address": N}, "range"),
        ({"address": -1}, "range"),
        ({"address": 1.5}, "integer"),
        ({}, "at least one"),
    ],
)
def test_invalid_events_are_rejected(events: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _table(events=events)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="event_source"):
        _table(event_source=" ")
    with pytest.raises(ValueError, match="handedness"):
        _table(handedness="both")
    with pytest.raises(ValueError, match="side"):
        _table(arms={"X": _arm("L", 0.0)})
    with pytest.raises(ValueError, match="does not match"):
        _table(arms={"R": _arm("L", 0.0)})
    with pytest.raises(ValueError, match="length"):
        _table(time_s=TIME[:-1])
    with pytest.raises(ValueError, match="arm"):
        _table(arms={})


def test_json_export_validates_against_schema_and_carries_conventions() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    doc = et.to_json_dict(_table())
    jsonschema.validate(doc, et.EVENT_TABLE_JSON_SCHEMA)
    assert doc["units"] == {"angle": "deg", "time": "s"}
    assert doc["handedness"] == "right"
    assert doc["event_source"] == SOURCE
    assert doc["events"]["impact"] == {
        "frame": EVENTS["impact"],
        "time_s": pytest.approx(TIME[EVENTS["impact"]]),
    }
    specs = {s.term_id: s for s in wrist_term_specs()}
    assert set(doc["terms"]) == set(specs)
    for term_id, meta in doc["terms"].items():
        assert meta["sign_convention"] == specs[term_id].sign_convention
        assert meta["frame"] == specs[term_id].frame
        assert meta["unit"] == "deg"
    # Round trip through text.
    assert json.loads(et.to_json(_table())) == json.loads(json.dumps(doc))


def test_schema_rejects_a_row_without_observability() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    doc = et.to_json_dict(_table())
    del doc["rows"][0]["observability"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, et.EVENT_TABLE_JSON_SCHEMA)


def test_csv_header_states_units_and_sign_conventions() -> None:
    text = et.to_csv(_table())
    comments = [ln for ln in text.splitlines() if ln.startswith("#")]
    joined = "\n".join(comments)
    assert "# units: angle=deg time=s" in comments
    assert f"# event_source: {SOURCE}" in comments
    assert "# handedness: right" in comments
    for spec in wrist_term_specs():
        assert f"# sign_convention {spec.term_id}: {spec.sign_convention}" in joined
    body = io.StringIO(
        "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
    )
    rows = list(csv.DictReader(body))
    assert list(rows[0]) == list(et.CSV_COLUMNS)
    assert len(rows) == len(_table().rows)
    first = rows[0]
    assert first["event"] == "address"
    assert float(first["value_deg"]) == pytest.approx(100.0)
    assert first["observability"] == OBSERVABILITY
    assert first["receipt_path"] == RECEIPT


def test_csv_leaves_invalid_values_blank() -> None:
    arm = _arm("L", 100.0)
    flex = arm.wrist.flexion_rad.copy()
    flex[0] = np.nan
    wrist = WristAngles(
        flex,
        arm.wrist.ulnar_deviation_rad,
        arm.wrist.axial_rotation_rad,
        arm.wrist.reasons,
        "L",
        RECEIPT,
        arm.wrist.calibration,
    )
    text = et.to_csv(_table(arms={"L": WristForearm(wrist, arm.forearm)}))
    body = io.StringIO(
        "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
    )
    first = next(csv.DictReader(body))
    assert first["value_deg"] == ""
    assert first["invalid_reason"]
