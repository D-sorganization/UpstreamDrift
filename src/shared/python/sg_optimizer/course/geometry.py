"""UTM projection utilities for georeferenced course geometry.

Provides lat/lon ↔ UTM round-trip accurate to <1 cm using pyproj.
Phase 2 (#6271).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pyproj import CRS, Transformer

from src.shared.python.contracts import require


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatLonPoint:
    """A geographic point in WGS-84 decimal degrees."""

    lat: float
    lon: float

    def __post_init__(self) -> None:
        require(-90.0 <= self.lat <= 90.0, "lat must be in [-90, 90]", self.lat)
        require(-180.0 <= self.lon <= 180.0, "lon must be in [-180, 180]", self.lon)


@dataclass(frozen=True)
class UTMPoint:
    """A UTM point with easting (x), northing (y) in metres, and zone string.

    ``zone`` should be a string like ``"17N"`` or ``"30S"``.
    """

    x: float  # easting (metres)
    y: float  # northing (metres)
    zone: str

    def __post_init__(self) -> None:
        require(len(self.zone) >= 2, "zone must be a non-empty zone string", self.zone)
        require(math.isfinite(self.x), "x must be finite", self.x)
        require(math.isfinite(self.y), "y must be finite", self.y)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utm_zone_number(lon: float) -> int:
    """Return UTM zone number for a given longitude."""
    return int((lon + 180.0) / 6.0) % 60 + 1


def _utm_hemisphere(lat: float) -> str:
    return "N" if lat >= 0.0 else "S"


def _utm_epsg(lon: float, lat: float) -> int:
    """Return EPSG code for the UTM zone covering (lon, lat)."""
    zone = _utm_zone_number(lon)
    if lat >= 0.0:
        return 32600 + zone  # WGS 84 / UTM zone N
    return 32700 + zone  # WGS 84 / UTM zone S


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def project_to_utm(points: list[LatLonPoint]) -> list[UTMPoint]:
    """Project a list of lat/lon points to UTM.

    All points are projected to the UTM zone of the *first* point so that
    distances computed within the returned list are Euclidean-valid.

    Round-trip accuracy is <1 cm for any point within ±500 km of the zone
    centre (standard UTM spec).

    Postcondition: ``len(result) == len(points)``.
    """
    require(len(points) > 0, "points list must be non-empty")

    ref = points[0]
    epsg = _utm_epsg(ref.lon, ref.lat)
    zone_num = _utm_zone_number(ref.lon)
    hemi = _utm_hemisphere(ref.lat)
    zone_str = f"{zone_num}{hemi}"

    wgs84 = CRS.from_epsg(4326)
    utm_crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(wgs84, utm_crs, always_xy=True)

    result: list[UTMPoint] = []
    for p in points:
        x, y = transformer.transform(p.lon, p.lat)
        result.append(UTMPoint(x=float(x), y=float(y), zone=zone_str))

    assert len(result) == len(points), "project_to_utm postcondition violated"
    return result


def utm_to_latlon(points: list[UTMPoint]) -> list[LatLonPoint]:
    """Inverse-project UTM points back to WGS-84 lat/lon.

    All points must share the same zone string (as returned by
    ``project_to_utm``).

    Postcondition: ``len(result) == len(points)``.
    """
    require(len(points) > 0, "points list must be non-empty")

    zone_str = points[0].zone
    require(
        all(p.zone == zone_str for p in points),
        "all UTM points must share the same zone",
    )

    # Parse zone string to get EPSG.
    zone_num = int(zone_str[:-1])
    hemi = zone_str[-1].upper()
    require(hemi in ("N", "S"), f"invalid hemisphere in zone {zone_str!r}", zone_str)
    epsg = 32600 + zone_num if hemi == "N" else 32700 + zone_num

    wgs84 = CRS.from_epsg(4326)
    utm_crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    result: list[LatLonPoint] = []
    for p in points:
        lon, lat = transformer.transform(p.x, p.y)
        result.append(LatLonPoint(lat=float(lat), lon=float(lon)))

    assert len(result) == len(points), "utm_to_latlon postcondition violated"
    return result


def haversine_m(p1: LatLonPoint, p2: LatLonPoint) -> float:
    """Return the haversine (great-circle) distance between two points in metres.

    Accurate to ~0.5 % for distances up to a few hundred kilometres; suitable
    for golf-course-scale distances where ~1 m accuracy suffices.
    """
    R = 6_371_000.0  # metres, mean Earth radius

    phi1 = math.radians(p1.lat)
    phi2 = math.radians(p2.lat)
    d_phi = math.radians(p2.lat - p1.lat)
    d_lam = math.radians(p2.lon - p1.lon)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c
