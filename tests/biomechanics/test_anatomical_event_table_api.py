"""HRG-E4 (#13175): API surface of the anatomical-angle event table.

The endpoint computes from calibrated segment frames with the same shared
functions the desktop export uses, so both clients get identical values.
"""

from __future__ import annotations

import csv
import io

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scipy.spatial.transform import Rotation

from src.api.routes.biomechanics import router
from src.shared.python.anatomical_angles import event_table as et
from src.shared.python.anatomical_angles.wrist import wrist_forearm_angles
from src.shared.python.biomechanics.joint_conventions import AnatomicalFrame

pytestmark = pytest.mark.unit

URL = "/biomechanics/anatomical-event-table"
N = 6
TIME = np.linspace(0.0, 0.5, N).tolist()
EVENTS = {"address": 0, "top": 2, "impact": 4, "finish": 5}
SOURCE = "swing_comparison.events.detect_events"
RECEIPT = "receipts/fit/receipt.json"


def _frame(side: str, segment: str, angles_deg: np.ndarray) -> dict[str, object]:
    rot = Rotation.from_euler("ZXY", angles_deg, degrees=True).as_matrix()
    return {
        "rotation_world": rot.tolist(),
        "origin_world": np.zeros((N, 3)).tolist(),
        "calibration_id": f"cal/{side}/{segment}",
    }


def _arm(side: str, offset: float) -> dict[str, object]:
    ramp = np.arange(N, dtype=float)
    zero = np.zeros((N, 3))
    hand = np.column_stack([offset + ramp, 5.0 + ramp, 10.0 + ramp])
    forearm = np.column_stack([np.zeros(N), np.zeros(N), offset / 2 + ramp])
    return {
        "humerus": _frame(side, "humerus", zero),
        "forearm": _frame(side, "forearm", forearm),
        "hand": _frame(side, "hand", hand),
    }


def _payload(**kw: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "time_s": TIME,
        "events": EVENTS,
        "handedness": "right",
        "event_source": SOURCE,
        "receipt_path": RECEIPT,
        "arms": {"L": _arm("L", 20.0), "R": _arm("R", 30.0)},
    }
    payload.update(kw)
    return payload


def _shared_export(payload: dict[str, object]) -> dict[str, object]:
    """The desktop path: shared functions called directly on the same bundle."""
    arms = {}
    for side, segs in payload["arms"].items():  # type: ignore[attr-defined]
        frames = {name: AnatomicalFrame(**seg) for name, seg in segs.items()}
        arms[side] = wrist_forearm_angles(
            frames["humerus"],
            frames["forearm"],
            frames["hand"],
            side,
            receipt_path=RECEIPT,
        )
    table = et.anatomical_event_table(
        arms, EVENTS, TIME, handedness="right", event_source=SOURCE
    )
    return et.to_json_dict(table)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_json_matches_the_shared_export_for_the_same_bundle(client) -> None:
    payload = _payload()
    response = client.post(URL, json=payload)
    assert response.status_code == 200, response.text
    assert response.json() == _shared_export(payload)


def test_json_response_validates_against_the_export_schema(client) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    response = client.post(URL, json=_payload())
    assert response.status_code == 200, response.text
    jsonschema.validate(response.json(), et.EVENT_TABLE_JSON_SCHEMA)


def test_csv_format_returns_the_shared_csv_text(client) -> None:
    payload = _payload(format="csv")
    response = client.post(URL, json=payload)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    body = response.text
    assert body.startswith(f"# schema: {et.SCHEMA_ID} v{et.SCHEMA_VERSION}")
    data = [line for line in body.splitlines() if not line.startswith("#")]
    rows = list(csv.DictReader(io.StringIO("\n".join(data))))
    expected = _shared_export(payload)["rows"]
    assert len(rows) == len(expected)  # type: ignore[arg-type]
    assert tuple(rows[0]) == et.CSV_COLUMNS


def test_single_side_yields_only_that_role(client) -> None:
    response = client.post(URL, json=_payload(arms={"L": _arm("L", 20.0)}))
    assert response.status_code == 200, response.text
    assert {r["role"] for r in response.json()["rows"]} == {"lead"}


@pytest.mark.parametrize(
    "override",
    [
        {"events": {"top": 4, "impact": 2}},
        {"events": {"backswing": 1}},
        {"event_source": " "},
        {"receipt_path": ""},
        {"handedness": "ambidextrous"},
        {"arms": {}},
        {"time_s": TIME[:-1]},
    ],
)
def test_invalid_bundle_is_a_client_error(client, override) -> None:
    response = client.post(URL, json=_payload(**override))
    assert response.status_code == 422, response.text


def test_unknown_format_is_rejected(client) -> None:
    response = client.post(URL, json=_payload(format="xlsx"))
    assert response.status_code == 422
