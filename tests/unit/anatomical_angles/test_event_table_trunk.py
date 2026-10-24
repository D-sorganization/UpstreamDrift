"""HRG-E4 (#13175): pelvis-reversal event and trunk/pelvis terms in the table."""

from __future__ import annotations

import csv
import io
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.anatomical_angles import event_table as et
from src.shared.python.anatomical_angles import trunk_events as te
from src.shared.python.anatomical_angles.trunk_pelvis import (
    golfer_lab_frame,
    term_specs,
)

pytestmark = pytest.mark.unit

UP = np.array([0.0, 0.0, 1.0])
TARGET = np.array([1.0, 0.0, 0.0])
TIME = np.linspace(0.0, 0.7, 8)
EVENTS = {"address": 0, "top": 4, "impact": 6, "finish": 7}
SOURCE = "swing_comparison.events.detect_events"
RECEIPT = "receipts/fit/receipt.json"
CAL = ("cal/pelvis", "cal/thorax")
OBS = "fitted_segment_rotation"
# Trail-positive turns (deg). Pelvis peaks at frame 3; the X-factor
# (thorax - pelvis) = [0, 10, 20, 40, 60, 50, 30, -10] peaks at frame 4.
PELVIS_TURN = np.array([0.0, 20.0, 40.0, 45.0, 35.0, 10.0, -30.0, -50.0])
THORAX_TURN = np.array([0.0, 30.0, 60.0, 85.0, 95.0, 60.0, 0.0, -60.0])
TERM_IDS = [s.term_id for s in term_specs()]


def _segment(turn_deg: np.ndarray, hand: str, bend_deg: float = 0.0) -> np.ndarray:
    """World rotations whose lab-frame turn (trail positive) is ``turn_deg``.

    Raw ISB AR is left-positive, so trail rotation is ``-AR`` right-handed and
    ``+AR`` left-handed; ``bend_deg`` is a constant forward bend (``-FE``).
    """
    sign = -1.0 if hand == "right" else 1.0
    angles = np.zeros((turn_deg.size, 3))
    angles[:, 0] = np.deg2rad(-bend_deg)
    angles[:, 2] = np.deg2rad(sign * turn_deg)
    lab = golfer_lab_frame(UP, TARGET, hand)
    return lab @ Rotation.from_euler("ZXY", angles).as_matrix()


def _trunk(hand: str = "right", pelvis=PELVIS_TURN, thorax=THORAX_TURN):
    return te.TrunkPelvisSeries(
        pelvis_rotation_world=_segment(pelvis, hand),
        thorax_rotation_world=_segment(thorax, hand, bend_deg=30.0),
        up=UP,
        target=TARGET,
        observability=OBS,
        receipt_path=RECEIPT,
        calibration=CAL,
    )


def _table(**kw: object) -> et.AnatomicalEventTable:
    hand = kw.get("handedness", "right")
    args: dict[str, object] = {
        "arms": {},
        "events": EVENTS,
        "time_s": TIME,
        "handedness": hand,
        "event_source": SOURCE,
        "trunk": _trunk(str(hand)),
    }
    args.update(kw)
    return et.anatomical_event_table(**args)  # type: ignore[arg-type]


def _row(table: et.AnatomicalEventTable, event: str, term: str) -> et.EventRow:
    (row,) = [r for r in table.rows if r.event == event and r.term_id == term]
    return row


@pytest.mark.parametrize("hand", ["right", "left"])
def test_pelvis_reversal_is_peak_pelvis_turn_not_max_x_factor(hand: str) -> None:
    table = _table(handedness=hand)
    frame, time = table.events["pelvis_reversal"]
    assert frame == 3
    assert time == pytest.approx(TIME[3])
    x_factor = THORAX_TURN - PELVIS_TURN
    assert int(np.argmax(x_factor)) == 4 != frame
    assert table.undetected_events == {}
    assert "pelvis_turn_isb" in table.event_detection["pelvis_reversal"]


def test_rows_list_events_in_table_order_then_trunk_terms() -> None:
    table = _table()
    order = [e for e in et.TABLE_EVENTS if e in table.events]
    assert order == ["address", "pelvis_reversal", "top", "impact", "finish"]
    assert [(r.event, r.term_id) for r in table.rows] == [
        (e, t) for e in order for t in TERM_IDS
    ]


def test_trunk_values_match_the_registered_terms_at_event_frames() -> None:
    table = _table()
    assert _row(table, "pelvis_reversal", "pelvis_turn_isb").value_deg == (
        pytest.approx(45.0, abs=1e-9)
    )
    assert _row(table, "top", "thorax_turn_isb").value_deg == pytest.approx(
        95.0, abs=1e-9
    )
    assert _row(table, "impact", "thorax_forward_bend_isb").value_deg == (
        pytest.approx(30.0, abs=1e-9)
    )
    assert _row(table, "address", "pelvis_lead_hip_up_isb").value_deg == (
        pytest.approx(0.0, abs=1e-9)
    )


def test_trunk_rows_carry_segment_role_and_provenance() -> None:
    for row in _table().rows:
        expected = "pelvis" if row.term_id.startswith("pelvis") else "thorax"
        assert (row.role, row.side) == (expected, et.MIDLINE)
        assert row.observability == OBS
        assert row.receipt_path == RECEIPT
        assert row.calibration == CAL


def test_trunk_alone_is_enough_but_something_is_required() -> None:
    table = _table(arms={})
    assert {r.role for r in table.rows} == {"pelvis", "thorax"}
    with pytest.raises(ValueError, match="arm"):
        _table(trunk=None)


def test_no_reversal_inside_the_window_is_reported_not_invented() -> None:
    rising = np.linspace(0.0, 70.0, TIME.size)
    table = _table(trunk=_trunk(pelvis=rising, thorax=rising + 10.0))
    assert "pelvis_reversal" not in table.events
    assert "pelvis_reversal" in table.undetected_events
    assert not [r for r in table.rows if r.event == "pelvis_reversal"]
    doc = et.to_json_dict(table)
    assert doc["undetected_events"]["pelvis_reversal"]
    assert "pelvis_reversal" in et.to_csv(table)


def test_reversal_search_is_limited_to_address_through_impact() -> None:
    # Peak at frame 1 lies before address (frame 2): no reversal in window.
    pelvis = np.array([0.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0, -10.0])
    events = {"address": 2, "top": 4, "impact": 6, "finish": 7}
    table = _table(events=events, trunk=_trunk(pelvis=pelvis))
    assert "pelvis_reversal" in table.undetected_events


def test_detect_pelvis_reversal_ignores_nan_and_needs_an_interior_peak() -> None:
    turn = np.array([0.0, 10.0, np.nan, 30.0, 20.0])
    assert te.detect_pelvis_reversal(turn, 0, 4) == (3, "")
    frame, reason = te.detect_pelvis_reversal(np.full(5, np.nan), 0, 4)
    assert frame is None and reason
    frame, reason = te.detect_pelvis_reversal(turn, 3, 4)
    assert frame is None and reason


def test_caller_cannot_supply_pelvis_reversal() -> None:
    with pytest.raises(ValueError, match="detected"):
        _table(events={**EVENTS, "pelvis_reversal": 3})


def test_trunk_series_length_and_provenance_are_validated() -> None:
    with pytest.raises(ValueError, match="length"):
        _table(time_s=TIME[:-1])
    bad = te.TrunkPelvisSeries(
        _segment(PELVIS_TURN, "right"),
        _segment(THORAX_TURN, "right"),
        UP,
        TARGET,
        observability=" ",
        receipt_path=RECEIPT,
        calibration=CAL,
    )
    with pytest.raises(ValueError, match="observability"):
        _table(trunk=bad)


def test_gimbal_locked_trunk_sample_exports_reason_not_a_number() -> None:
    top = EVENTS["top"]
    lab = golfer_lab_frame(UP, TARGET, "right")
    thorax = _segment(THORAX_TURN, "right")
    thorax[top] = lab @ Rotation.from_euler("ZXY", [0.0, np.pi / 2, 0.0]).as_matrix()
    trunk = te.TrunkPelvisSeries(
        _segment(PELVIS_TURN, "right"), thorax, UP, TARGET, OBS, RECEIPT, CAL
    )
    table = _table(trunk=trunk)
    row = _row(table, "top", "thorax_turn_isb")
    assert math.isnan(row.value_deg)
    assert row.invalid_reason == te.SINGULAR_REASON
    side_bend = _row(table, "top", "thorax_lead_side_bend_isb")
    assert not math.isnan(side_bend.value_deg)
    doc = et.to_json_dict(table)
    json.dumps(doc, allow_nan=False)


def test_json_with_trunk_validates_and_states_conventions() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    table = _table()
    doc = et.to_json_dict(table)
    jsonschema.validate(doc, et.EVENT_TABLE_JSON_SCHEMA)
    assert doc["schema_version"] == et.SCHEMA_VERSION == 2
    rev = doc["events"]["pelvis_reversal"]
    assert rev["frame"] == 3
    assert rev["detection"] == table.event_detection["pelvis_reversal"]
    specs = {s.term_id: s for s in term_specs()}
    for term_id, spec in specs.items():
        meta = doc["terms"][term_id]
        assert meta["sign_convention"] == spec.sign_convention
        assert meta["unit"] == "deg"
    assert doc["undetected_events"] == {}
    json.loads(et.to_json(table))


def test_schema_rejects_unknown_role() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    doc = et.to_json_dict(_table())
    doc["rows"][0]["role"] = "spine"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, et.EVENT_TABLE_JSON_SCHEMA)


def test_csv_header_has_trunk_conventions_and_reversal_definition() -> None:
    text = et.to_csv(_table())
    comments = [ln for ln in text.splitlines() if ln.startswith("#")]
    joined = "\n".join(comments)
    for spec in term_specs():
        assert f"# sign_convention {spec.term_id}: {spec.sign_convention}" in joined
    assert "# detection pelvis_reversal:" in joined
    body = io.StringIO(
        "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
    )
    rows = list(csv.DictReader(body))
    assert list(rows[0]) == list(et.CSV_COLUMNS)
    assert {r["side"] for r in rows} == {et.MIDLINE}
