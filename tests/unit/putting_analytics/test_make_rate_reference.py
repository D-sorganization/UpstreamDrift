"""Cited make-rate reference tables: loading, provenance and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.putting_analytics import make_rate_reference as ref

pytestmark = pytest.mark.unit


def test_every_table_is_cited_and_in_range() -> None:
    tables = ref.load_reference_tables()
    assert len({t.source_id for t in tables}) >= 8
    for table in tables:
        assert table.url.startswith("https://") and table.publisher
        assert all(0.0 <= p.pct <= 100.0 for p in table.points)


def test_calibration_target_and_check_are_present() -> None:
    target = ref.reference_table("golfmagic_2022", "pga_tour")
    assert target.role == "calibration_target" and len(target.points) == 13
    assert (target.points[1].low_ft, target.points[1].pct) == (3.0, 96.0)
    check = ref.reference_table("shotscope_mygolfspy", "hcp_25")
    assert check.role == "check" and check.points[1].pct == 48.0


def test_bands_open_bands_and_approximate_figures() -> None:
    scratch = ref.reference_table("shotscope_mygolfspy", "hcp_0")
    assert scratch.points[0].is_band and scratch.points[0].high_ft == 3.0
    assert scratch.points[-1].high_ft == ref.OPEN_BAND_CAP_FT
    tour = ref.reference_table("golfdigest_2024", "tour")
    assert tour.points[0].approx is False and tour.points[-1].approx is True


def test_unknown_table_is_a_key_error() -> None:
    with pytest.raises(KeyError, match="nope"):
        ref.reference_table("nope", "tour")


def _write(tmp_path: Path, groups: dict[str, list[object]]) -> Path:
    source = {
        "id": "s",
        "role": "reference",
        "metric": "make_pct",
        "publisher": "p",
        "url": "https://example.org",
        "published": None,
        "underlying_data": "d",
        "groups": groups,
    }
    path = tmp_path / "rates.json"
    path.write_text(json.dumps({"sources": [source, source]}), encoding="utf-8")
    return path


def test_duplicate_tables_and_bad_flags_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        ref.load_reference_tables(_write(tmp_path, {"g": [[3, 50]]}))
    with pytest.raises(ValueError, match="approx"):
        ref.load_reference_tables(_write(tmp_path, {"g": [[3, 50, "guess"]]}))


def test_points_validate_their_band_and_percentage() -> None:
    with pytest.raises(ValueError, match="ordered"):
        ref.MakeRatePoint(6.0, 3.0, 50.0)
    with pytest.raises(ValueError, match="pct"):
        ref.MakeRatePoint(3.0, 3.0, 101.0)
