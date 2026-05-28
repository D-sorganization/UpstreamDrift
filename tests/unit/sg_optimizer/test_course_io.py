"""Unit tests for sg_optimizer.course.course_io — GeoJSON load/save round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.sg_optimizer.course.course_io import (
    HoleGeometry,
    load_hole_geojson,
    save_hole_geojson,
)
from src.shared.python.sg_optimizer.course.geometry import LatLonPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_geojson(tmp_path: Path) -> Path:
    """Write a minimal valid GeoJSON Feature to ``tmp_path``."""
    data = {
        "type": "Feature",
        "properties": {
            "hole_number": 17,
            "par": 3,
            "yardage": 137,
            "name": "Test Hole 17",
            "schema_version": "1.1",
        },
        "geometry": {
            "type": "GeometryCollection",
            "geometries": [
                {
                    "type": "Point",
                    "role": "tee",
                    "coordinates": [-81.39720, 30.19780],
                },
                {
                    "type": "Point",
                    "role": "green_center",
                    "coordinates": [-81.39580, 30.19710],
                },
                {
                    "type": "Polygon",
                    "role": "green",
                    "coordinates": [
                        [
                            [-81.3961, 30.1973],
                            [-81.3958, 30.19735],
                            [-81.39555, 30.1969],
                            [-81.3961, 30.1973],
                        ]
                    ],
                },
                {
                    "type": "Polygon",
                    "role": "water",
                    "coordinates": [
                        [
                            [-81.397, 30.198],
                            [-81.395, 30.198],
                            [-81.395, 30.196],
                            [-81.397, 30.196],
                            [-81.397, 30.198],
                        ]
                    ],
                },
            ],
        },
    }
    p = tmp_path / "test_hole.geojson"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# HoleGeometry dataclass
# ---------------------------------------------------------------------------


def test_hole_geometry_construction():
    h = HoleGeometry(
        hole_number=17,
        par=3,
        yardage=137,
        name="Sawgrass 17",
        tee=LatLonPoint(lat=30.19780, lon=-81.39720),
        green_center=LatLonPoint(lat=30.19710, lon=-81.39580),
    )
    assert h.par == 3
    assert h.yardage == 137
    assert h.fairway == []
    assert h.water == []


def test_hole_geometry_invalid_hole_number():
    from src.shared.python.contracts import ContractViolationError

    with pytest.raises(ContractViolationError):
        HoleGeometry(
            hole_number=0,
            par=3,
            yardage=137,
            name="x",
            tee=LatLonPoint(lat=30.0, lon=-81.0),
            green_center=LatLonPoint(lat=30.0, lon=-81.0),
        )


def test_hole_geometry_invalid_par():
    from src.shared.python.contracts import ContractViolationError

    with pytest.raises(ContractViolationError):
        HoleGeometry(
            hole_number=1,
            par=2,
            yardage=100,
            name="x",
            tee=LatLonPoint(lat=30.0, lon=-81.0),
            green_center=LatLonPoint(lat=30.0, lon=-81.0),
        )


# ---------------------------------------------------------------------------
# load_hole_geojson
# ---------------------------------------------------------------------------


def test_load_minimal_geojson(tmp_path):
    p = _minimal_geojson(tmp_path)
    hole = load_hole_geojson(p)
    assert hole.hole_number == 17
    assert hole.par == 3
    assert hole.yardage == 137
    assert hole.name == "Test Hole 17"
    assert hole.tee.lat == pytest.approx(30.19780)
    assert hole.tee.lon == pytest.approx(-81.39720)
    assert hole.green_center.lat == pytest.approx(30.19710)
    assert len(hole.green) == 1
    assert len(hole.water) == 1
    assert hole.fairway == []
    assert hole.bunker == []


def test_load_missing_file_raises():
    from src.shared.python.contracts import ContractViolationError

    with pytest.raises(ContractViolationError):
        load_hole_geojson("/nonexistent/path/hole.geojson")


def test_load_wrong_geojson_type_raises(tmp_path):
    p = tmp_path / "bad.geojson"
    p.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="expected GeoJSON Feature"):
        load_hole_geojson(p)


def test_load_missing_tee_raises(tmp_path):
    data = {
        "type": "Feature",
        "properties": {"hole_number": 1, "par": 3, "yardage": 100, "name": "x"},
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
    p = tmp_path / "notee.geojson"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required.*tee"):
        load_hole_geojson(p)


# ---------------------------------------------------------------------------
# save_hole_geojson
# ---------------------------------------------------------------------------


def test_save_produces_valid_json(tmp_path):
    hole = HoleGeometry(
        hole_number=13,
        par=5,
        yardage=510,
        name="Augusta 13",
        tee=LatLonPoint(lat=33.50310, lon=-82.02290),
        green_center=LatLonPoint(lat=33.50120, lon=-82.01850),
        fairway=[
            [
                [
                    [-82.022, 33.502],
                    [-82.021, 33.502],
                    [-82.021, 33.501],
                    [-82.022, 33.501],
                    [-82.022, 33.502],
                ]
            ]
        ],
    )
    out = tmp_path / "augusta_13.geojson"
    save_hole_geojson(hole, out)
    assert out.exists()
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["type"] == "Feature"
    assert raw["properties"]["schema_version"] == "1.1"
    assert raw["geometry"]["type"] == "GeometryCollection"


def test_save_creates_parent_dirs(tmp_path):
    hole = HoleGeometry(
        hole_number=7,
        par=3,
        yardage=106,
        name="Pebble 7",
        tee=LatLonPoint(lat=36.56890, lon=-121.94880),
        green_center=LatLonPoint(lat=36.56820, lon=-121.94870),
    )
    out = tmp_path / "subdir" / "deeper" / "pebble_7.geojson"
    save_hole_geojson(hole, out)
    assert out.exists()


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_all_fields(tmp_path):
    original = HoleGeometry(
        hole_number=17,
        par=3,
        yardage=137,
        name="Sawgrass 17",
        tee=LatLonPoint(lat=30.19780, lon=-81.39720),
        green_center=LatLonPoint(lat=30.19710, lon=-81.39580),
        green=[
            [
                [
                    [-81.3961, 30.1973],
                    [-81.3958, 30.19735],
                    [-81.39555, 30.1969],
                    [-81.3961, 30.1973],
                ]
            ]
        ],
        water=[
            [
                [
                    [-81.397, 30.198],
                    [-81.395, 30.198],
                    [-81.395, 30.196],
                    [-81.397, 30.196],
                    [-81.397, 30.198],
                ]
            ]
        ],
    )
    out = tmp_path / "sawgrass_17.geojson"
    save_hole_geojson(original, out)
    loaded = load_hole_geojson(out)

    assert loaded.hole_number == original.hole_number
    assert loaded.par == original.par
    assert loaded.yardage == original.yardage
    assert loaded.name == original.name
    assert loaded.tee.lat == pytest.approx(original.tee.lat)
    assert loaded.tee.lon == pytest.approx(original.tee.lon)
    assert loaded.green_center.lat == pytest.approx(original.green_center.lat)
    assert len(loaded.green) == len(original.green)
    assert len(loaded.water) == len(original.water)
