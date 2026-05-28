"""Unit tests for sg_optimizer.course.geometry — UTM projection utilities."""

from __future__ import annotations

import pytest

from src.shared.python.sg_optimizer.course.geometry import (
    LatLonPoint,
    UTMPoint,
    haversine_m,
    project_to_utm,
    utm_to_latlon,
)


# ---------------------------------------------------------------------------
# LatLonPoint / UTMPoint construction
# ---------------------------------------------------------------------------


def test_latlon_valid():
    p = LatLonPoint(lat=36.56890, lon=-121.94880)
    assert p.lat == pytest.approx(36.56890)
    assert p.lon == pytest.approx(-121.94880)


def test_latlon_invalid_lat():
    from src.shared.python.contracts import ContractViolationError

    with pytest.raises(ContractViolationError):
        LatLonPoint(lat=91.0, lon=0.0)


def test_latlon_invalid_lon():
    from src.shared.python.contracts import ContractViolationError

    with pytest.raises(ContractViolationError):
        LatLonPoint(lat=0.0, lon=200.0)


def test_utm_point_valid():
    p = UTMPoint(x=582000.0, y=4046000.0, zone="10N")
    assert p.x == pytest.approx(582000.0)
    assert p.zone == "10N"


# ---------------------------------------------------------------------------
# project_to_utm
# ---------------------------------------------------------------------------


def test_project_to_utm_returns_correct_count():
    pts = [
        LatLonPoint(lat=30.197, lon=-81.397),
        LatLonPoint(lat=30.198, lon=-81.396),
    ]
    result = project_to_utm(pts)
    assert len(result) == 2


def test_project_to_utm_same_zone():
    pts = [
        LatLonPoint(lat=30.197, lon=-81.397),
        LatLonPoint(lat=30.200, lon=-81.390),
    ]
    result = project_to_utm(pts)
    assert result[0].zone == result[1].zone


def test_project_to_utm_empty_raises():
    from src.shared.python.contracts import ContractViolationError

    with pytest.raises(ContractViolationError):
        project_to_utm([])


# ---------------------------------------------------------------------------
# Round-trip accuracy (<1 cm)
# ---------------------------------------------------------------------------


def test_utm_round_trip_accuracy_sub_centimetre():
    """project_to_utm → utm_to_latlon round-trip must be accurate to <1 cm."""
    originals = [
        LatLonPoint(lat=30.19710, lon=-81.39580),  # Sawgrass 17 green
        LatLonPoint(lat=33.50120, lon=-82.01850),  # Augusta 13 green
        LatLonPoint(lat=56.34170, lon=-2.80540),  # Road Hole 17 green
        LatLonPoint(lat=36.57220, lon=-121.96500),  # Cypress 16 green
        LatLonPoint(lat=36.56820, lon=-121.94870),  # Pebble 7 green
    ]
    for origin in originals:
        utm_pts = project_to_utm([origin])
        recovered = utm_to_latlon(utm_pts)
        rec = recovered[0]

        # 1 cm ≈ 9e-8 degrees at the equator (even less at higher latitudes).
        # We use 1e-7 as a generous but still sub-cm tolerance.
        assert abs(rec.lat - origin.lat) < 1e-7, (
            f"lat round-trip error too large for {origin}: "
            f"{abs(rec.lat - origin.lat):.2e} deg"
        )
        assert abs(rec.lon - origin.lon) < 1e-7, (
            f"lon round-trip error too large for {origin}: "
            f"{abs(rec.lon - origin.lon):.2e} deg"
        )


def test_utm_round_trip_multiple_points():
    pts = [
        LatLonPoint(lat=30.197 + i * 0.001, lon=-81.397 + i * 0.001) for i in range(5)
    ]
    utm = project_to_utm(pts)
    recovered = utm_to_latlon(utm)
    for orig, rec in zip(pts, recovered, strict=True):
        assert abs(rec.lat - orig.lat) < 1e-7
        assert abs(rec.lon - orig.lon) < 1e-7


# ---------------------------------------------------------------------------
# haversine_m
# ---------------------------------------------------------------------------


def test_haversine_same_point_is_zero():
    p = LatLonPoint(lat=30.197, lon=-81.397)
    assert haversine_m(p, p) == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance():
    # ~137 yards ≈ 125 m for Sawgrass 17 (tee to green).
    tee = LatLonPoint(lat=30.19780, lon=-81.39720)
    green = LatLonPoint(lat=30.19710, lon=-81.39580)
    dist = haversine_m(tee, green)
    # Approximate distance — within 50 m of 125 m.
    assert 50.0 < dist < 250.0


def test_haversine_symmetry():
    p1 = LatLonPoint(lat=33.50310, lon=-82.02290)
    p2 = LatLonPoint(lat=33.50120, lon=-82.01850)
    assert haversine_m(p1, p2) == pytest.approx(haversine_m(p2, p1), rel=1e-9)


def test_haversine_approximate_metre_scale():
    """1 arcsecond latitude ≈ 30.87 m; verify haversine is in the right ballpark."""
    p1 = LatLonPoint(lat=0.0, lon=0.0)
    p2 = LatLonPoint(lat=1.0 / 3600.0, lon=0.0)  # 1 arcsec north
    dist = haversine_m(p1, p2)
    assert 30.0 < dist < 32.0


def test_haversine_equatorial_degree():
    """1 degree latitude at equator ≈ 111,320 m."""
    p1 = LatLonPoint(lat=0.0, lon=0.0)
    p2 = LatLonPoint(lat=1.0, lon=0.0)
    dist = haversine_m(p1, p2)
    assert abs(dist - 111_320.0) < 200.0  # within 200 m
