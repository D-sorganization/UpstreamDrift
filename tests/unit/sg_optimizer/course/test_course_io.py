"""Unit tests for sg_optimizer.course.course_io (Phase 2).

Tests GeoJSON schema v1.1 round-trip, validation, and error paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.course.course_io import (
    HoleGeometry,
    load_hole_geojson,
    save_hole_geojson,
)
from src.shared.python.sg_optimizer.course.geometry import LatLonPoint


# ---------------------------------------------------------------------------
# HoleGeometry contract checks
# ---------------------------------------------------------------------------


def test_hole_geometry_invalid_par():
    with pytest.raises((ContractViolationError, ValueError)):
        HoleGeometry(
            hole_number=1,
            par=2,  # < 3
            yardage=150,
            name="test",
            tee=LatLonPoint(lat=30.0, lon=-81.0),
            green_center=LatLonPoint(lat=30.001, lon=-81.001),
        )


def test_hole_geometry_invalid_yardage():
    with pytest.raises((ContractViolationError, ValueError)):
        HoleGeometry(
            hole_number=1,
            par=3,
            yardage=0,
            name="test",
            tee=LatLonPoint(lat=30.0, lon=-81.0),
            green_center=LatLonPoint(lat=30.001, lon=-81.001),
        )


def test_hole_geometry_empty_name():
    with pytest.raises((ContractViolationError, ValueError)):
        HoleGeometry(
            hole_number=1,
            par=3,
            yardage=150,
            name="",
            tee=LatLonPoint(lat=30.0, lon=-81.0),
            green_center=LatLonPoint(lat=30.001, lon=-81.001),
        )


# ---------------------------------------------------------------------------
# Round-trip: save then reload
# ---------------------------------------------------------------------------


def _make_test_hole() -> HoleGeometry:
    return HoleGeometry(
        hole_number=17,
        par=3,
        yardage=137,
        name="Test Island",
        tee=LatLonPoint(lat=30.19780, lon=-81.39720),
        green_center=LatLonPoint(lat=30.19710, lon=-81.39580),
        water=[
            [
                [-81.39730, 30.19800],
                [-81.39530, 30.19800],
                [-81.39530, 30.19660],
                [-81.39730, 30.19660],
                [-81.39730, 30.19800],
            ]
        ],
        green=[
            [
                [-81.39610, 30.19730],
                [-81.39580, 30.19735],
                [-81.39550, 30.19725],
                [-81.39610, 30.19730],
            ]
        ],
    )


def test_round_trip_basic(tmp_path):
    hole = _make_test_hole()
    out = tmp_path / "sawgrass_17.geojson"
    save_hole_geojson(hole, out)

    loaded = load_hole_geojson(out)
    assert loaded.hole_number == hole.hole_number
    assert loaded.par == hole.par
    assert loaded.yardage == hole.yardage
    assert loaded.name == hole.name
    assert loaded.tee.lat == pytest.approx(hole.tee.lat, abs=1e-7)
    assert loaded.tee.lon == pytest.approx(hole.tee.lon, abs=1e-7)
    assert len(loaded.water) == 1
    assert len(loaded.green) == 1


def test_round_trip_creates_parent_dirs(tmp_path):
    hole = _make_test_hole()
    nested = tmp_path / "a" / "b" / "c" / "hole.geojson"
    save_hole_geojson(hole, nested)
    assert nested.exists()


def test_round_trip_schema_version(tmp_path):
    hole = _make_test_hole()
    out = tmp_path / "h.geojson"
    save_hole_geojson(hole, out)
    raw = json.loads(out.read_text())
    assert raw["properties"]["schema_version"] == "1.1"


# ---------------------------------------------------------------------------
# load_hole_geojson — error paths
# ---------------------------------------------------------------------------


def test_load_missing_file_raises():
    with pytest.raises((FileNotFoundError, ContractViolationError, ValueError)):
        load_hole_geojson("/nonexistent/path/hole.geojson")


def test_load_wrong_type_raises(tmp_path):
    bad = tmp_path / "bad.geojson"
    bad.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    with pytest.raises(ValueError, match="Feature"):
        load_hole_geojson(bad)


def test_load_missing_tee_raises(tmp_path):
    bad = tmp_path / "bad.geojson"
    bad.write_text(
        json.dumps(
            {
                "type": "Feature",
                "properties": {"hole_number": 1, "par": 3, "yardage": 150, "name": "x"},
                "geometry": {
                    "type": "GeometryCollection",
                    "geometries": [
                        {
                            "type": "Point",
                            "role": "green_center",
                            "coordinates": [-81.0, 30.0],
                        }
                    ],
                },
            }
        )
    )
    with pytest.raises(ValueError, match="tee"):
        load_hole_geojson(bad)


def test_load_missing_green_center_raises(tmp_path):
    bad = tmp_path / "bad.geojson"
    bad.write_text(
        json.dumps(
            {
                "type": "Feature",
                "properties": {"hole_number": 1, "par": 3, "yardage": 150, "name": "x"},
                "geometry": {
                    "type": "GeometryCollection",
                    "geometries": [
                        {
                            "type": "Point",
                            "role": "tee",
                            "coordinates": [-81.0, 30.0],
                        }
                    ],
                },
            }
        )
    )
    with pytest.raises(ValueError, match="green_center"):
        load_hole_geojson(bad)


# ---------------------------------------------------------------------------
# Classic data files parseable
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[4]
CLASSICS_DIR = REPO_ROOT / "data" / "sg_optimizer" / "courses" / "classics"

_CLASSIC_SLUGS = [
    "sawgrass_17",
    "augusta_13",
    "pebble_7",
    "road_hole_17",
    "cypress_16",
]


@pytest.mark.parametrize("slug", _CLASSIC_SLUGS)
def test_classic_geojson_loads(slug):
    path = CLASSICS_DIR / f"{slug}.geojson"
    assert path.exists(), f"Classic GeoJSON missing: {path}"
    hole = load_hole_geojson(path)
    assert hole.par >= 3
    assert hole.yardage > 0
    assert hole.name
