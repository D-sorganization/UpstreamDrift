"""GeoJSON schema v1.1 read/write for sg_optimizer holes.

Schema (one Feature per hole)::

    {
      "type": "Feature",
      "properties": {
        "hole_number": <int>,
        "par": <int>,
        "yardage": <int>,
        "name": <str>,
        "schema_version": "1.1"
      },
      "geometry": {
        "type": "GeometryCollection",
        "geometries": [
          {"type": "Point",   "role": "tee",          "coordinates": [lon, lat]},
          {"type": "Point",   "role": "green_center",  "coordinates": [lon, lat]},
          {"type": "Polygon", "role": "fairway",       "coordinates": [[[lon, lat], ...]]},
          {"type": "Polygon", "role": "rough",         "coordinates": [[[lon, lat], ...]]},
          {"type": "Polygon", "role": "water",         "coordinates": [[[lon, lat], ...]]},
          {"type": "Polygon", "role": "bunker",        "coordinates": [[[lon, lat], ...]]},
          {"type": "Polygon", "role": "ob",            "coordinates": [[[lon, lat], ...]]}
        ]
      }
    }

Phase 2 (#6271).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.shared.python.contracts import require
from src.shared.python.sg_optimizer.course.geometry import LatLonPoint

_SCHEMA_VERSION = "1.1"

# Recognised polygon roles for hazard/feature classification.
_POLYGON_ROLES = frozenset({"fairway", "rough", "water", "bunker", "ob", "green"})
_POINT_ROLES = frozenset({"tee", "green_center"})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoleGeometry:
    """Georeferenced geometry for a single golf hole.

    All coordinate lists follow GeoJSON convention: [longitude, latitude].
    Polygon coordinates are lists of rings; the outer ring is index 0.
    """

    hole_number: int
    par: int
    yardage: int
    name: str
    tee: LatLonPoint
    green_center: LatLonPoint
    # Polygon layers — each entry is a list of coordinate rings where each
    # ring is a list of [lon, lat] pairs.
    fairway: list[list[list[float]]] = field(default_factory=list)
    rough: list[list[list[float]]] = field(default_factory=list)
    water: list[list[list[float]]] = field(default_factory=list)
    bunker: list[list[list[float]]] = field(default_factory=list)
    ob: list[list[list[float]]] = field(default_factory=list)
    green: list[list[list[float]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        require(self.hole_number >= 1, "hole_number must be >= 1", self.hole_number)
        require(self.par >= 3, "par must be >= 3", self.par)
        require(self.yardage > 0, "yardage must be > 0", self.yardage)
        require(len(self.name) > 0, "name must be non-empty")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def load_hole_geojson(path: str | Path) -> HoleGeometry:
    """Load a HoleGeometry from a GeoJSON file.

    Validates schema version and required roles.
    Raises ``ValueError`` on schema violations.
    """
    p = Path(path)
    require(p.exists(), f"GeoJSON file not found: {p}")
    raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return _parse_feature(raw, path=str(p))


def _parse_feature(raw: dict[str, Any], *, path: str = "<data>") -> HoleGeometry:
    if raw.get("type") != "Feature":
        raise ValueError(f"{path}: expected GeoJSON Feature, got {raw.get('type')!r}")

    props: dict[str, Any] = raw.get("properties") or {}
    geom: dict[str, Any] = raw.get("geometry") or {}

    if geom.get("type") != "GeometryCollection":
        raise ValueError(
            f"{path}: geometry.type must be 'GeometryCollection', "
            f"got {geom.get('type')!r}"
        )

    hole_number = int(props.get("hole_number", 1))
    par = int(props.get("par", 4))
    yardage = int(props.get("yardage", 400))
    name = str(props.get("name", ""))

    tee: LatLonPoint | None = None
    green_center: LatLonPoint | None = None
    fairway: list[list[list[float]]] = []
    rough: list[list[list[float]]] = []
    water: list[list[list[float]]] = []
    bunker: list[list[list[float]]] = []
    ob: list[list[list[float]]] = []
    green_poly: list[list[list[float]]] = []

    for g in geom.get("geometries") or []:
        role = str(g.get("role", ""))
        gtype = str(g.get("type", ""))
        coords = g.get("coordinates")

        if gtype == "Point" and role == "tee" and coords is not None:
            tee = LatLonPoint(lat=float(coords[1]), lon=float(coords[0]))
        elif gtype == "Point" and role == "green_center" and coords is not None:
            green_center = LatLonPoint(lat=float(coords[1]), lon=float(coords[0]))
        elif gtype == "Polygon" and coords is not None:
            rings: list[list[float]] = coords  # type: ignore[assignment]
            if role == "fairway":
                fairway.append(rings)
            elif role == "rough":
                rough.append(rings)
            elif role == "water":
                water.append(rings)
            elif role == "bunker":
                bunker.append(rings)
            elif role == "ob":
                ob.append(rings)
            elif role == "green":
                green_poly.append(rings)

    if tee is None:
        raise ValueError(f"{path}: missing required Point geometry with role='tee'")
    if green_center is None:
        raise ValueError(
            f"{path}: missing required Point geometry with role='green_center'"
        )

    return HoleGeometry(
        hole_number=hole_number,
        par=par,
        yardage=yardage,
        name=name,
        tee=tee,
        green_center=green_center,
        fairway=fairway,
        rough=rough,
        water=water,
        bunker=bunker,
        ob=ob,
        green=green_poly,
    )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def save_hole_geojson(hole: HoleGeometry, path: str | Path) -> None:
    """Serialize a HoleGeometry to a GeoJSON file (schema v1.1).

    Creates parent directories if needed.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    geometries: list[dict[str, Any]] = [
        {
            "type": "Point",
            "role": "tee",
            "coordinates": [hole.tee.lon, hole.tee.lat],
        },
        {
            "type": "Point",
            "role": "green_center",
            "coordinates": [hole.green_center.lon, hole.green_center.lat],
        },
    ]

    def _add_polygons(rings_list: list[list[list[float]]], role: str) -> None:
        for rings in rings_list:
            geometries.append({"type": "Polygon", "role": role, "coordinates": rings})

    _add_polygons(hole.fairway, "fairway")
    _add_polygons(hole.rough, "rough")
    _add_polygons(hole.green, "green")
    _add_polygons(hole.water, "water")
    _add_polygons(hole.bunker, "bunker")
    _add_polygons(hole.ob, "ob")

    feature: dict[str, Any] = {
        "type": "Feature",
        "properties": {
            "hole_number": hole.hole_number,
            "par": hole.par,
            "yardage": hole.yardage,
            "name": hole.name,
            "schema_version": _SCHEMA_VERSION,
        },
        "geometry": {
            "type": "GeometryCollection",
            "geometries": geometries,
        },
    }

    out.write_text(json.dumps(feature, indent=2), encoding="utf-8")
