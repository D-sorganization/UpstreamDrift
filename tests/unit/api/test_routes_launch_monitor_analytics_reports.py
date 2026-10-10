"""Tests for the ``POST /v2/report`` and ``POST /v2/export`` routes.

Mirrors the PyQt Reports tab (``src/tools/launch_monitor_analytics/gui.py``
``_refresh_report``/``export_data``/``export_manifest``) so the API and
desktop paths share one implementation:
``src/tools/launch_monitor_analytics/reporting.py`` (issue #11987 slice 7a).
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.launch_monitor_analytics_reports import router

pytestmark = pytest.mark.unit

_URL_REPORT = "/tools/launch-monitor-analytics/v2/report"
_URL_EXPORT = "/tools/launch-monitor-analytics/v2/export"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _report_records() -> list[dict[str, object]]:
    """Four shots across two sessions, plus one shot with no session id."""
    return [
        {"shot_id": 1, "session_id": "s1", "club_speed": 90.0, "club": "Driver"},
        {"shot_id": 2, "session_id": "s1", "club_speed": 95.0, "club": "Iron"},
        {"shot_id": 3, "session_id": "s2", "club_speed": 100.0, "club": "Driver"},
        {"shot_id": 4, "session_id": None, "club_speed": 80.0, "club": "Wedge"},
    ]


def _direct_report_text(
    records: list[dict[str, object]],
    *,
    project_name: str,
    session_count: int,
    treatment_action_count: int,
) -> str:
    import pandas as pd

    from src.tools.launch_monitor_analytics.reporting import build_project_report

    return build_project_report(
        project_name=project_name,
        session_count=session_count,
        frame=pd.DataFrame.from_records(records),
        import_warning_count=0,
        treatment_action_count=treatment_action_count,
    )


# ---- /v2/report -------------------------------------------------------------


def test_report_v2_matches_direct_build_project_report_call(
    client: TestClient,
) -> None:
    records = _report_records()
    audit_log = [{"action": "drop_missing", "row_index": 1}]
    response = client.post(
        _URL_REPORT,
        json={
            "records": records,
            "project_name": "My Project",
            "treatment_audit_log": audit_log,
        },
    )
    assert response.status_code == 200
    payload = response.json()

    expected_text = _direct_report_text(
        records,
        project_name="My Project",
        session_count=2,
        treatment_action_count=1,
    )
    assert payload["report_text"] == expected_text
    assert payload["project_name"] == "My Project"
    assert payload["session_count"] == 2
    assert payload["shot_count"] == len(records)
    assert payload["canonical_metrics"] == ["club_speed"]
    assert payload["treatment_action_count"] == 1


def test_report_v2_session_count_ignores_null_and_dedupes(
    client: TestClient,
) -> None:
    response = client.post(_URL_REPORT, json={"records": _report_records()})
    assert response.status_code == 200
    assert response.json()["session_count"] == 2


def test_report_v2_session_count_is_zero_without_session_id_column(
    client: TestClient,
) -> None:
    records = [{"club_speed": 90.0}, {"club_speed": 95.0}]
    response = client.post(_URL_REPORT, json={"records": records})
    assert response.status_code == 200
    assert response.json()["session_count"] == 0


def test_report_v2_default_project_name_matches_desktop_clear_project(
    client: TestClient,
) -> None:
    response = client.post(_URL_REPORT, json={"records": _report_records()})
    assert response.status_code == 200
    assert response.json()["project_name"] == "Untitled Launch Monitor Study"


def test_report_v2_import_warning_count_is_always_zero(client: TestClient) -> None:
    """The web app has no import step, so no import warnings can occur."""
    response = client.post(_URL_REPORT, json={"records": _report_records()})
    assert response.status_code == 200
    assert "Import Warnings: 0" in response.json()["report_text"]


def test_report_v2_empty_records_is_422(client: TestClient) -> None:
    response = client.post(_URL_REPORT, json={"records": []})
    assert response.status_code == 422


def test_report_v2_whitespace_only_project_name_is_400(client: TestClient) -> None:
    """Schema allows non-empty text; the domain rule rejects blank-after-strip."""
    response = client.post(
        _URL_REPORT,
        json={"records": _report_records(), "project_name": "   "},
    )
    assert response.status_code == 400


def test_report_v2_empty_project_name_is_422(client: TestClient) -> None:
    """An actually-empty string fails the schema's ``min_length=1``."""
    response = client.post(
        _URL_REPORT,
        json={"records": _report_records(), "project_name": ""},
    )
    assert response.status_code == 422


# ---- /v2/export -------------------------------------------------------------


def test_export_v2_csv_starts_with_export_id_comment_and_roundtrips(
    client: TestClient,
) -> None:
    import pandas as pd

    # No null cells here (unlike `_report_records()`): a round-tripped CSV
    # represents a missing cell as NaN, not None, so comparing against a
    # frame containing a literal `None` would just be testing that quirk.
    records = [
        {"shot_id": 1, "session_id": "s1", "club_speed": 90.0, "club": "Driver"},
        {"shot_id": 2, "session_id": "s2", "club_speed": 95.0, "club": "Iron"},
    ]
    response = client.post(_URL_EXPORT, json={"records": records})
    assert response.status_code == 200
    payload = response.json()

    csv_text = payload["csv"]
    first_line = csv_text.splitlines()[0]
    assert first_line.startswith("# export_id=") and "exported_at=" in first_line

    parsed = pd.read_csv(io.StringIO(csv_text), comment="#")
    pd.testing.assert_frame_equal(parsed, pd.DataFrame.from_records(records))


def test_export_v2_data_sha256_matches_hashlib_of_csv_bytes(
    client: TestClient,
) -> None:
    response = client.post(_URL_EXPORT, json={"records": _report_records()})
    assert response.status_code == 200
    payload = response.json()

    csv_bytes = payload["csv"].encode("utf-8")
    assert (
        payload["data_export"]["data_sha256"] == hashlib.sha256(csv_bytes).hexdigest()
    )
    assert payload["data_export"]["data_file"] == "launch_monitor_data.csv"
    assert payload["data_export"]["export_id"]
    assert payload["data_export"]["exported_at"]


def test_export_v2_manifest_keys_equal_desktop_keys(client: TestClient) -> None:
    response = client.post(
        _URL_EXPORT,
        json={"records": _report_records(), "project_name": "Manifest Project"},
    )
    assert response.status_code == 200
    manifest = response.json()["manifest"]

    assert list(manifest.keys()) == [
        "project",
        "sessions",
        "treatment_audit_log",
        "analysis_rows",
        "canonical_metrics",
        "scientific_boundary",
        "data_export",
    ]
    assert manifest["project"] == "Manifest Project"
    assert manifest["sessions"] == []
    assert manifest["analysis_rows"] == len(_report_records())
    assert manifest["canonical_metrics"] == ["club_speed"]
    assert "Scientific Boundary" in manifest["scientific_boundary"]


def test_export_v2_manifest_data_export_matches_response_data_export(
    client: TestClient,
) -> None:
    response = client.post(_URL_EXPORT, json={"records": _report_records()})
    assert response.status_code == 200
    payload = response.json()
    assert payload["manifest"]["data_export"] == payload["data_export"]


def test_export_v2_audit_log_roundtrips_in_manifest(client: TestClient) -> None:
    audit_log: list[dict[str, Any]] = [
        {"action": "flag_outlier", "row_index": 2, "metric": "club_speed"}
    ]
    response = client.post(
        _URL_EXPORT,
        json={"records": _report_records(), "treatment_audit_log": audit_log},
    )
    assert response.status_code == 200
    assert response.json()["manifest"]["treatment_audit_log"] == audit_log


def test_export_v2_empty_records_is_422(client: TestClient) -> None:
    response = client.post(_URL_EXPORT, json={"records": []})
    assert response.status_code == 422
