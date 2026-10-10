"""Tests for the ``POST /v2/import/preview`` and ``POST /v2/import`` routes.

Mirrors the PyQt Sessions tab's import flow
(``src/tools/launch_monitor_analytics/widgets.py``'s ``ImportMappingDialog``,
``src/tools/launch_monitor_analytics/gui.py``'s ``import_file``) so the web
and desktop import paths share one implementation: ``import_session``
(issue #11987).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.launch_monitor_analytics_sessions import router

pytestmark = pytest.mark.unit

_URL_PREVIEW = "/tools/launch-monitor-analytics/v2/import/preview"
_URL_IMPORT = "/tools/launch-monitor-analytics/v2/import"
_FIXTURES = Path(__file__).parents[2] / "fixtures" / "launch_monitor"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _trackman_csv_bytes() -> bytes:
    return (_FIXTURES / "trackman.csv").read_bytes()


# ---- /v2/import/preview ------------------------------------------------------


def test_preview_detects_trackman_profile_and_auto_mappings(client: TestClient) -> None:
    response = client.post(
        _URL_PREVIEW,
        files={"file": ("trackman.csv", _trackman_csv_bytes(), "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["filename"] == "trackman.csv"
    assert "Club Speed (mph)" in payload["headers"]
    assert payload["detected_profile_id"] == "trackman"
    assert payload["detection_confidence"] > 0.5
    assert {"profile_id": "trackman", "vendor": "TrackMan"} in payload["profiles"]
    assert payload["auto_mappings"]["trackman"]["Club Speed (mph)"] == "club_speed"
    assert "club_speed" in payload["targets"]
    assert payload["targets"][0] == "(retain only)"
    assert payload["measurement_statuses"] == [
        "reported",
        "measured",
        "estimated",
        "derived",
        "unknown",
    ]


def test_preview_rejects_unsupported_suffix(client: TestClient) -> None:
    response = client.post(
        _URL_PREVIEW,
        files={"file": ("shots.pdf", b"not a real pdf", "application/pdf")},
    )
    assert response.status_code == 415


# ---- /v2/import ---------------------------------------------------------------


def test_import_v2_matches_direct_import_session_call(
    client: TestClient, tmp_path: Path
) -> None:
    """The web import and the desktop's direct call must agree byte-for-byte."""
    from src.tools.launch_monitor_model import import_session

    content = _trackman_csv_bytes()
    source_path = tmp_path / "trackman.csv"
    source_path.write_bytes(content)
    direct = import_session(source_path)

    response = client.post(
        _URL_IMPORT,
        files={"file": ("trackman.csv", content, "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["session_id"] == direct.session_id
    assert payload["name"] == direct.name
    assert payload["row_count"] == len(direct.shots)
    assert set(payload["columns"]) == {str(c) for c in direct.shots.columns}

    direct_rows = direct.shots.reset_index(drop=True)
    for index, record in enumerate(payload["records"]):
        assert record["club_speed"] == pytest.approx(
            direct_rows.loc[index, "club_speed"]
        )
        assert record["carry_distance"] == pytest.approx(
            direct_rows.loc[index, "carry_distance"]
        )


def test_import_v2_manifest_has_no_temp_path(client: TestClient) -> None:
    content = _trackman_csv_bytes()
    response = client.post(
        _URL_IMPORT,
        files={"file": ("trackman.csv", content, "text/csv")},
    )
    assert response.status_code == 200
    manifest = response.json()["manifest"]
    assert manifest["source_path"] == "trackman.csv"
    assert "lm_import_" not in json.dumps(manifest)


def test_import_v2_unit_and_multiplier_override_flips_sign(
    client: TestClient,
) -> None:
    content = _trackman_csv_bytes()
    baseline = client.post(
        _URL_IMPORT,
        files={"file": ("trackman.csv", content, "text/csv")},
    ).json()
    baseline_speed = baseline["records"][0]["club_speed"]

    options = {
        "profile_id": "trackman",
        "mappings": [
            {
                "source_column": "Club Speed (mph)",
                "target_column": "club_speed",
                "source_unit": "mph",
                "multiplier": -1.0,
                "measurement_status": "measured",
            }
        ],
    }
    response = client.post(
        _URL_IMPORT,
        files={"file": ("trackman.csv", content, "text/csv")},
        data={"options": json.dumps(options)},
    )
    assert response.status_code == 200
    overridden_speed = response.json()["records"][0]["club_speed"]
    assert overridden_speed == pytest.approx(-baseline_speed)


def test_import_v2_nan_metric_is_null_not_zero(client: TestClient) -> None:
    rows = [
        "Shot,Date,Time,Club,Club Speed (mph),Ball Speed (mph),Attack Angle (deg),"
        "Club Path (deg),Face Angle (deg),Face to Path (deg),Dynamic Loft (deg),"
        "Launch Angle (deg),Launch Direction (deg),Spin Rate (rpm),Spin Axis (deg),"
        "Carry (yd),Total (yd),Carry Side (yd),Height (ft)",
        "1,2026-01-02,10:00:00,7 Iron,,121,-3.5,2.0,1.0,-1.0,20.0,16.0,0.8,6100,"
        "-2.5,170,176,3,92",
    ]
    content = ("\n".join(rows) + "\n").encode("utf-8")
    response = client.post(
        _URL_IMPORT,
        files={"file": ("trackman.csv", content, "text/csv")},
    )
    assert response.status_code == 200
    record = response.json()["records"][0]
    assert record["club_speed"] is None


def test_import_v2_unsupported_suffix_is_415(client: TestClient) -> None:
    response = client.post(
        _URL_IMPORT,
        files={"file": ("shots.pdf", b"not a real pdf", "application/pdf")},
    )
    assert response.status_code == 415


def test_import_v2_no_rows_error_has_no_temp_path(client: TestClient) -> None:
    header_only = (
        b"Shot,Date,Time,Club,Club Speed (mph),Ball Speed (mph),Attack Angle (deg),"
        b"Club Path (deg),Face Angle (deg),Face to Path (deg),Dynamic Loft (deg),"
        b"Launch Angle (deg),Launch Direction (deg),Spin Rate (rpm),Spin Axis (deg),"
        b"Carry (yd),Total (yd),Carry Side (yd),Height (ft)\n"
    )
    response = client.post(
        _URL_IMPORT,
        files={"file": ("trackman.csv", header_only, "text/csv")},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail == "Launch-monitor source contains no rows: trackman.csv"


def test_import_v2_invalid_options_json_is_422(client: TestClient) -> None:
    response = client.post(
        _URL_IMPORT,
        files={"file": ("trackman.csv", _trackman_csv_bytes(), "text/csv")},
        data={"options": "not valid json"},
    )
    assert response.status_code == 422


def test_import_v2_unknown_target_column_is_422(client: TestClient) -> None:
    options = {
        "mappings": [
            {
                "source_column": "Club Speed (mph)",
                "target_column": "not_a_real_target",
            }
        ],
    }
    response = client.post(
        _URL_IMPORT,
        files={"file": ("trackman.csv", _trackman_csv_bytes(), "text/csv")},
        data={"options": json.dumps(options)},
    )
    assert response.status_code == 422
    assert "Unknown target column" in response.json()["detail"]


def test_import_v2_oversized_row_count_is_413(client: TestClient) -> None:
    rows = ["Note"] + [str(index) for index in range(20_001)]
    content = ("\n".join(rows) + "\n").encode("utf-8")
    response = client.post(
        _URL_IMPORT,
        files={"file": ("bulk.csv", content, "text/csv")},
    )
    assert response.status_code == 413


def test_import_v2_tags_round_trip_via_options(client: TestClient) -> None:
    options = {"tags": ["range-session", "7-iron"]}
    response = client.post(
        _URL_IMPORT,
        files={"file": ("trackman.csv", _trackman_csv_bytes(), "text/csv")},
        data={"options": json.dumps(options)},
    )
    assert response.status_code == 200
    record = response.json()["records"][0]
    assert record["tags"] == "range-session, 7-iron"


def test_import_v2_default_session_name_is_filename_stem(client: TestClient) -> None:
    response = client.post(
        _URL_IMPORT,
        files={"file": ("trackman.csv", _trackman_csv_bytes(), "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "trackman"


def test_import_v2_explicit_session_name_overrides_default(
    client: TestClient,
) -> None:
    options = {"session_name": "Driving Range Visit"}
    response = client.post(
        _URL_IMPORT,
        files={"file": ("trackman.csv", _trackman_csv_bytes(), "text/csv")},
        data={"options": json.dumps(options)},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Driving Range Visit"


def test_import_v2_records_have_no_nan_strings(client: TestClient) -> None:
    """Every float value is finite or ``None``; never the string ``"nan"``."""
    response = client.post(
        _URL_IMPORT,
        files={"file": ("trackman.csv", _trackman_csv_bytes(), "text/csv")},
    )
    assert response.status_code == 200
    for record in response.json()["records"]:
        for value in record.values():
            if isinstance(value, float):
                assert math.isfinite(value)
