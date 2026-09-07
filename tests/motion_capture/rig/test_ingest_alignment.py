"""Ingest applies the session's strobe alignment as evidence beside ``time_s``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.motion_capture.rig.alignment import TIMING_REPORT_FILE
from src.motion_capture.rig.bundle import MANIFEST_FILE
from src.motion_capture.rig.ingest import ingest_bundle
from src.motion_capture.rig.session import SessionManifest
from src.shared.python.engine_core.engine_availability import skip_if_unavailable

from .test_ingest import FakeEstimator, _bundle

pytestmark = [pytest.mark.unit, skip_if_unavailable("cv2")]

UNCERTAINTY_NS = 47_140_452  # two 30 fps intervals in quadrature


def _timing(status_b: str = "available") -> dict:
    views = [
        {
            "view": "a",
            "status": "available",
            "offset_ns": 0,
            "uncertainty_ns": UNCERTAINTY_NS,
            "rate_deviation_ppm": 0.0,
        },
        {
            "view": "b",
            "status": status_b,
            "offset_ns": 100_000_000 if status_b == "available" else None,
            "uncertainty_ns": UNCERTAINTY_NS if status_b == "available" else None,
            "rate_deviation_ppm": 0.0,
            "reason": None if status_b == "available" else "no strobe detected",
        },
    ]
    return {
        "method": "flash_event",
        "clock_domain": "host_monotonic_ns",
        "reference_view": "a",
        "status": "available" if status_b == "available" else "unavailable",
        "views": views,
    }


def _with_timing(bundle: Path, timing: dict) -> None:
    manifest = SessionManifest.model_validate_json(
        (bundle / MANIFEST_FILE).read_text(encoding="utf-8")
    )
    manifest.model_copy(update={"timing": timing}).save(bundle / MANIFEST_FILE)


def test_reference_clock_fields_and_report_when_timing_is_available(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    _with_timing(bundle, _timing())
    out = tmp_path / "obs"
    result = ingest_bundle(bundle, out, FakeEstimator)
    assert result.timing_report == TIMING_REPORT_FILE
    report = json.loads((out / TIMING_REPORT_FILE).read_text(encoding="utf-8"))
    assert report["reference_view"] == "a" and len(report["views"]) == 2
    view_b = json.loads((out / "b.json").read_text(encoding="utf-8"))
    rows = view_b["frames"]
    assert rows[1]["time_s"] == pytest.approx(1 / 30)  # per-view clock untouched
    assert rows[1]["time_ref_s"] == pytest.approx(1 / 30 - 0.1)
    assert rows[1]["time_ref_uncertainty_s"] == pytest.approx(UNCERTAINTY_NS / 1e9)
    assert rows[1]["time_ref_source"] == "flash_event"
    assert view_b["provenance"]["timing_applied"] == "flash_event"


def test_views_without_a_strobe_keep_only_their_own_clock(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    _with_timing(bundle, _timing(status_b="unavailable"))
    out = tmp_path / "obs"
    ingest_bundle(bundle, out, FakeEstimator)
    view_b = json.loads((out / "b.json").read_text(encoding="utf-8"))
    assert "time_ref_s" not in view_b["frames"][0]
    assert "timing_applied" not in view_b["provenance"]
    report = json.loads((out / TIMING_REPORT_FILE).read_text(encoding="utf-8"))
    b = next(v for v in report["views"] if v["view"] == "b")
    assert b["status"] == "unavailable" and b["reason"] == "no strobe detected"


def test_no_timing_block_means_no_report(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    out = tmp_path / "obs"
    result = ingest_bundle(bundle, out, FakeEstimator)
    assert result.timing_report is None
    assert not (out / TIMING_REPORT_FILE).exists()
