"""Tests for the Reports tab's pure report/export builders (issue #11987 slice 7a).

Extracted from ``src/tools/launch_monitor_analytics/gui.py`` so the PyQt
desktop app and the FastAPI web routes build byte-identical report text, CSV
exports, and reproducibility manifests from one implementation. These tests
pin the exact text/CSV/manifest shape the desktop's ``_refresh_report``,
``export_data``, and ``export_manifest`` already produce.
"""

from __future__ import annotations

import hashlib
import io

import pandas as pd
import pytest

from src.tools.launch_monitor_analytics.reporting import (
    SCIENTIFIC_BOUNDARY_TEXT,
    build_data_export_record,
    build_project_report,
    build_reproducibility_manifest,
    format_canonical_csv_export,
)

pytestmark = pytest.mark.unit


def _sample_frame() -> pd.DataFrame:
    """Two shots with one canonical metric, one non-metric, one source field."""
    return pd.DataFrame(
        {
            "club_speed": [90.0, 95.0],
            "club": ["Driver", "Iron"],
            "source::vendor_field": ["a", "b"],
        }
    )


# ---- build_project_report -------------------------------------------------


def test_build_project_report_golden_text() -> None:
    """Exact text match for a frame with one metric and one source:: column."""
    report = build_project_report(
        project_name="Test Project",
        session_count=3,
        frame=_sample_frame(),
        import_warning_count=2,
        treatment_action_count=5,
    )
    assert report == (
        "Launch Monitor Analytics Project\n"
        "================================\n"
        "Project: Test Project\n"
        "Sessions: 3\n"
        "Shots: 2\n"
        "Canonical Numeric Metrics: 1\n"
        "Retained Source Fields: 1\n"
        "Import Warnings: 2\n\n"
        "Recorded Treatment Actions: 5\n\n"
        "Scientific Interpretation & Traceability\n"
        "-----------------------------------------\n"
        "Relationships describe association, not causation. Identity-derived "
        "metrics are marked by the metric registry. Matched shots are required "
        "for monitor bias and agreement claims; unmatched comparisons remain "
        "descriptive. Original source columns and per-file SHA-256 provenance "
        "are retained in the project.\n\n"
        "Methodology & Formula Traceability:\n"
        "- Longitudinal trends: Theil-Sen robust linear regression with Mann-Kendall test\n"
        "- Dispersion: 95% bivariate normal confidence ellipse (Hotelling T^2)\n"
        "- Multicollinearity: Variance Inflation Factor (VIF = 1 / (1 - R_i^2))\n"
        "- Strokes Gained: SG = verified E(start state) - 1 - verified E(finish state) "
        "(Broadie 2011/2014, DOI: 10.1287/inte.1110.0594)\n"
    )


def test_build_project_report_rejects_empty_project_name() -> None:
    with pytest.raises(ValueError, match="project_name"):
        build_project_report(
            project_name="   ",
            session_count=1,
            frame=_sample_frame(),
            import_warning_count=0,
            treatment_action_count=0,
        )


@pytest.mark.parametrize(
    "field", ["session_count", "import_warning_count", "treatment_action_count"]
)
def test_build_project_report_rejects_negative_counts(field: str) -> None:
    kwargs: dict[str, object] = {
        "project_name": "P",
        "session_count": 1,
        "frame": _sample_frame(),
        "import_warning_count": 0,
        "treatment_action_count": 0,
    }
    kwargs[field] = -1
    with pytest.raises(ValueError, match=field):
        build_project_report(**kwargs)


# ---- format_canonical_csv_export ------------------------------------------


def test_format_canonical_csv_export_header_and_roundtrip() -> None:
    frame = pd.DataFrame({"club_speed": [90.0, 95.5], "club": ["Driver", "Iron"]})
    csv_text = format_canonical_csv_export(
        frame, export_id="abc123", exported_at="2026-01-01T00:00:00+00:00"
    )
    first_line = csv_text.splitlines()[0]
    assert first_line == "# export_id=abc123 exported_at=2026-01-01T00:00:00+00:00"

    parsed = pd.read_csv(io.StringIO(csv_text), comment="#")
    pd.testing.assert_frame_equal(parsed, frame)


@pytest.mark.parametrize(
    ("export_id", "exported_at"), [("", "2026-01-01"), ("abc", "")]
)
def test_format_canonical_csv_export_rejects_blank_fields(
    export_id: str, exported_at: str
) -> None:
    frame = pd.DataFrame({"club_speed": [1.0]})
    with pytest.raises(ValueError):
        format_canonical_csv_export(frame, export_id=export_id, exported_at=exported_at)


# ---- build_data_export_record ---------------------------------------------


def test_build_data_export_record_sha256_matches_hashlib() -> None:
    data_bytes = b"# export_id=eid exported_at=t\nclub_speed\n90.0\n"
    record = build_data_export_record(
        export_id="eid",
        exported_at="2026-01-01T00:00:00+00:00",
        data_file="launch_monitor_data.csv",
        data_bytes=data_bytes,
    )
    assert record == {
        "export_id": "eid",
        "exported_at": "2026-01-01T00:00:00+00:00",
        "data_file": "launch_monitor_data.csv",
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "export_id": "",
            "exported_at": "t",
            "data_file": "f.csv",
            "data_bytes": b"x",
        },
        {
            "export_id": "e",
            "exported_at": "",
            "data_file": "f.csv",
            "data_bytes": b"x",
        },
        {
            "export_id": "e",
            "exported_at": "t",
            "data_file": "",
            "data_bytes": b"x",
        },
        {
            "export_id": "e",
            "exported_at": "t",
            "data_file": "f.csv",
            "data_bytes": b"",
        },
    ],
)
def test_build_data_export_record_rejects_blank_fields(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        build_data_export_record(**kwargs)


# ---- build_reproducibility_manifest ----------------------------------------


def test_build_reproducibility_manifest_keys_and_values() -> None:
    frame = _sample_frame()
    sessions = [{"vendor": "FlightScope"}]
    audit_log = [{"action": "drop_missing", "row_index": 3}]
    data_export = {
        "export_id": "eid",
        "exported_at": "2026-01-01T00:00:00+00:00",
        "data_file": "launch_monitor_data.csv",
        "data_sha256": "deadbeef",
    }
    manifest = build_reproducibility_manifest(
        project_name="My Project",
        sessions=sessions,
        treatment_audit_log=audit_log,
        frame=frame,
        scientific_boundary=SCIENTIFIC_BOUNDARY_TEXT,
        data_export=data_export,
    )
    assert list(manifest.keys()) == [
        "project",
        "sessions",
        "treatment_audit_log",
        "analysis_rows",
        "canonical_metrics",
        "scientific_boundary",
        "data_export",
    ]
    assert manifest["project"] == "My Project"
    assert manifest["sessions"] is sessions
    assert manifest["treatment_audit_log"] is audit_log
    assert manifest["analysis_rows"] == 2
    assert manifest["canonical_metrics"] == ["club_speed"]
    assert manifest["scientific_boundary"] == SCIENTIFIC_BOUNDARY_TEXT
    assert manifest["data_export"] is data_export


def test_build_reproducibility_manifest_data_export_none_passthrough() -> None:
    manifest = build_reproducibility_manifest(
        project_name="P",
        sessions=[],
        treatment_audit_log=[],
        frame=pd.DataFrame({"club_speed": [1.0]}),
        scientific_boundary="boundary text",
        data_export=None,
    )
    assert manifest["data_export"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"project_name": "", "scientific_boundary": "b"},
        {"project_name": "   ", "scientific_boundary": "b"},
        {"project_name": "p", "scientific_boundary": ""},
        {"project_name": "p", "scientific_boundary": "   "},
    ],
)
def test_build_reproducibility_manifest_rejects_blank_text(
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        build_reproducibility_manifest(
            sessions=[],
            treatment_audit_log=[],
            frame=pd.DataFrame({"club_speed": [1.0]}),
            data_export=None,
            **kwargs,
        )
