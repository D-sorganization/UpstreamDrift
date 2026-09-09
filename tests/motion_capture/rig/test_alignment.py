"""Strobe alignment applied at ingest, as evidence beside the per-view clock."""

from __future__ import annotations

import pytest

from src.motion_capture.rig.alignment import (
    TIMING_REPORT_SCHEMA_VERSION,
    TimingReport,
    annotate_rows,
    build_timing_report,
    reference_time_fields,
    view_timing,
)

pytestmark = pytest.mark.unit

TIMING = {
    "method": "flash_event",
    "clock_domain": "host_monotonic_ns",
    "reference_view": "ref",
    "status": "unavailable",
    "views": [
        {
            "view": "ref",
            "status": "available",
            "offset_ns": 0,
            "uncertainty_ns": 23_570_226,
            "rate_deviation_ppm": 0.0,
        },
        {
            "view": "side",
            "status": "available",
            "offset_ns": 7_000_000,
            "uncertainty_ns": 26_034_000,
            "rate_deviation_ppm": 200_000.0,
        },
        {
            "view": "dark",
            "status": "unavailable",
            "reason": "no strobe detected",
            "rate_deviation_ppm": None,
        },
    ],
}


def test_view_timing_lookup() -> None:
    assert view_timing(TIMING, "side") is not None
    assert view_timing(TIMING, "nope") is None
    assert view_timing({}, "ref") is None


def test_reference_time_fields_subtract_offset_and_carry_uncertainty() -> None:
    fields = reference_time_fields(1.0, 7_000_000, 26_034_000, "flash_event")
    assert fields == {
        "time_ref_s": pytest.approx(0.993),
        "time_ref_uncertainty_s": pytest.approx(0.026034),
        "time_ref_source": "flash_event",
    }
    with pytest.raises(Exception, match="uncertainty_ns"):
        reference_time_fields(1.0, 0, -1, "flash_event")


def test_annotate_rows_keeps_original_time_and_adds_reference_fields() -> None:
    rows = [
        {"camera_id": "2", "time_s": 0.0, "keypoints_px": []},
        {"camera_id": "2", "time_s": 0.5},
    ]
    out = annotate_rows(rows, 7_000_000, 26_034_000, "flash_event")
    assert [r["time_s"] for r in out] == [0.0, 0.5]  # untouched
    assert out[1]["time_ref_s"] == pytest.approx(0.493)
    assert out[0]["time_ref_source"] == "flash_event"
    assert rows[0] == {
        "camera_id": "2",
        "time_s": 0.0,
        "keypoints_px": [],
    }  # inputs not mutated


def test_build_timing_report_computes_expected_skew_and_keeps_unavailable() -> None:
    report = build_timing_report(TIMING, {"ref": 10.0, "side": 10.3, "dark": 10.0})
    assert isinstance(report, TimingReport)
    assert report.schema_version == TIMING_REPORT_SCHEMA_VERSION
    assert (report.reference_view, report.method, report.status) == (
        "ref",
        "flash_event",
        "unavailable",
    )
    by_view = {v.view: v for v in report.views}
    assert by_view["ref"].expected_skew_ms == pytest.approx(0.0)
    # 200 000 ppm = 20 % slow over 10.3 s -> 2.06 s of accumulated skew
    assert by_view["side"].expected_skew_ms == pytest.approx(2060.0)
    assert by_view["side"].offset_ns == 7_000_000
    assert (
        by_view["dark"].status == "unavailable"
        and by_view["dark"].expected_skew_ms is None
    )
    assert by_view["dark"].reason == "no strobe detected"


def test_build_timing_report_requires_the_timing_record_shape() -> None:
    with pytest.raises(Exception, match="timing block lacks"):
        build_timing_report({"views": []}, {})
