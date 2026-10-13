"""Swing report web routes (issue #12330): parity with the PyQt dialog (#12329).

The React dialog consumes the same validated ``swing-report-v1`` documents as
the PyQt ``SwingReportDialog``: the rows it shows are the Python
``ReportViewModel`` rows (text only, so a withheld quantity cannot leak a
number), and the export is the S4 package.
"""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path
import zipfile

import pytest

pytest.importorskip("jsonschema")
pytest.importorskip("matplotlib")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.routes import swing_report as route  # noqa: E402
from src.shared.python.reporting.swing_report import (  # noqa: E402
    select_plots,
    validate_swing_report,
)
from src.shared.python.reporting.swing_report.exporter import (  # noqa: E402
    load_package,
)
from src.shared.python.reporting.swing_report.view_model import (  # noqa: E402
    ReportViewModel,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "swing_report" / "valid_full.json"
WEB_ROWS = (
    ROOT
    / "ui"
    / "src"
    / "components"
    / "analysis"
    / "swingReport"
    / "fixtures"
    / "validFullView.json"
)
WITHHELD = {"face_angle_ts", "joint_load_ts"}
ROW_FIELDS = {
    "plot_id",
    "title",
    "kind",
    "enabled",
    "status_label",
    "detail",
    "unavailable_reason",
}


@pytest.fixture
def doc() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(route.router)
    return TestClient(app)


def _view(client: TestClient, doc: dict) -> dict:
    resp = client.post("/swing-report/view", json={"report": doc})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_view_rows_match_pyqt_view_model(client, doc):
    body = _view(client, doc)
    expected = [
        dataclasses.asdict(r) for r in ReportViewModel(validate_swing_report(doc)).rows
    ]
    assert body["rows"] == expected
    assert body["report_id"] == doc["report_id"]
    assert body["package_filename"] == route.package_filename(doc["report_id"])


def test_withheld_set_unchanged_and_disabled_with_reason(client, doc):
    rows = _view(client, doc)["rows"]
    disabled = {r["plot_id"] for r in rows if not r["enabled"]}
    assert disabled == WITHHELD
    for row in rows:
        if row["plot_id"] in WITHHELD:
            assert row["status_label"] == "Not Validated"
            assert row["unavailable_reason"]


def test_rows_expose_no_numeric_data(client, doc):
    for row in _view(client, doc)["rows"]:
        assert set(row) == ROW_FIELDS
        assert not any(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in row.values()
        )


def test_default_custom_ids_are_enabled_default_selected(client, doc):
    body = _view(client, doc)
    assert body["default_custom_ids"] == ["clubhead_speed_ts"]


def test_web_fixture_matches_python_view(client, doc):
    """The Vitest fixture is the Python view, so both UIs test the same rows."""
    assert json.loads(WEB_ROWS.read_text(encoding="utf-8")) == _view(client, doc)


def test_view_rejects_invalid_report(client, doc):
    doc["quantities"] = "nope"
    resp = client.post("/swing-report/view", json={"report": doc})
    assert resp.status_code == 422
    assert "violates contract" in resp.json()["detail"]


def _unzip(resp, tmp_path: Path) -> Path:
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/zip"
    assert "attachment" in resp.headers["content-disposition"]
    out = tmp_path / "pkg"
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        archive.extractall(out)
    return out


def test_full_export_round_trips(client, doc, tmp_path):
    resp = client.post("/swing-report/export", json={"report": doc, "mode": "full"})
    loaded = load_package(_unzip(resp, tmp_path))
    assert loaded == select_plots(validate_swing_report(doc)).report


def test_custom_export_round_trips_selection(client, doc, tmp_path):
    resp = client.post(
        "/swing-report/export",
        json={"report": doc, "mode": "custom", "plot_ids": ["clubhead_path"]},
    )
    loaded = load_package(_unzip(resp, tmp_path))
    assert loaded.scope == "custom"
    assert [p.plot_id for p in loaded.plots] == ["clubhead_path"]


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "custom", "plot_ids": ["face_angle_ts"]},
        {"mode": "custom", "plot_ids": ["no_such_plot"]},
        {"mode": "custom", "plot_ids": []},
    ],
)
def test_export_rejects_unselectable_custom_selection(client, doc, payload):
    resp = client.post("/swing-report/export", json={"report": doc, **payload})
    assert resp.status_code == 422


def test_export_rejects_unknown_mode(client, doc):
    resp = client.post("/swing-report/export", json={"report": doc, "mode": "all"})
    assert resp.status_code == 422


def test_export_filename_is_sanitised():
    assert route.package_filename("a/b c..") == "swing-report-a_b_c...zip"
