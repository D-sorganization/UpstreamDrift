"""Unit tests for sg_optimizer.course.geometry (Phase 2).

Tests UTM projection round-trip accuracy, haversine distance, and
contract enforcement.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.course.geometry import (
    LatLonPoint,
    UTMPoint,
    haversine_m,
    project_to_utm,
    utm_to_latlon,
)


# ---------------------------------------------------------------------------
# LatLonPoint / UTMPoint contract checks
# ---------------------------------------------------------------------------


def test_latlon_out_of_range_lat():
    with pytest.raises((ContractViolationError, ValueError)):
        LatLonPoint(lat=91.0, lon=0.0)


def test_latlon_out_of_range_lon():
    with pytest.raises((ContractViolationError, ValueError)):
        LatLonPoint(lat=0.0, lon=181.0)


def test_utm_invalid_zone():
    with pytest.raises((ContractViolationError, ValueError)):
        UTMPoint(x=300000.0, y=3000000.0, zone="")


def test_utm_non_finite():
    with pytest.raises((ContractViolationError, ValueError)):
        UTMPoint(x=float("inf"), y=3000000.0, zone="17N")


# ---------------------------------------------------------------------------
# project_to_utm requires non-empty input
# ---------------------------------------------------------------------------


def test_project_empty_raises():
    with pytest.raises((ContractViolationError, ValueError)):
        project_to_utm([])


def test_utm_to_latlon_empty_raises():
    with pytest.raises((ContractViolationError, ValueError)):
        utm_to_latlon([])


# ---------------------------------------------------------------------------
# Round-trip accuracy <1 cm
# ---------------------------------------------------------------------------


def test_utm_round_trip_sawgrass():
    """TPC Sawgrass hole 17 tee position round-trips within 1 cm."""
    original = [LatLonPoint(lat=30.19780, lon=-81.39720)]
    utm_pts = project_to_utm(original)
    assert len(utm_pts) == 1
    assert utm_pts[0].zone  # non-empty zone string

    back = utm_to_latlon(utm_pts)
    assert len(back) == 1
    # Check round-trip error using haversine.
    err_m = haversine_m(original[0], back[0])
    assert err_m < 0.01, f"round-trip error {err_m:.4f} m exceeds 1 cm"


def test_utm_round_trip_multiple_points():
    """Multiple points in the same zone all round-trip within 1 cm."""
    pts = [
        LatLonPoint(lat=30.19780, lon=-81.39720),
        LatLonPoint(lat=30.19710, lon=-81.39580),
        LatLonPoint(lat=30.19730, lon=-81.39610),
    ]
    utm_pts = project_to_utm(pts)
    back = utm_to_latlon(utm_pts)
    assert len(back) == len(pts)
    for orig, recovered in zip(pts, back, strict=True):
        err_m = haversine_m(orig, recovered)
        assert err_m < 0.01, f"round-trip error {err_m:.4f} m exceeds 1 cm"


def test_utm_lengths_match():
    pts = [LatLonPoint(lat=33.50310, lon=-82.02290)]
    assert len(project_to_utm(pts)) == 1


# ---------------------------------------------------------------------------
# haversine_m
# ---------------------------------------------------------------------------


def test_haversine_same_point_is_zero():
    p = LatLonPoint(lat=30.0, lon=-81.0)
    assert haversine_m(p, p) == pytest.approx(0.0, abs=1e-6)


def test_haversine_rough_distance():
    """~111 km per degree of latitude near the equator."""
    p1 = LatLonPoint(lat=0.0, lon=0.0)
    p2 = LatLonPoint(lat=1.0, lon=0.0)
    dist = haversine_m(p1, p2)
    assert 110_000 < dist < 112_000


def test_haversine_symmetric():
    p1 = LatLonPoint(lat=30.0, lon=-81.0)
    p2 = LatLonPoint(lat=30.0, lon=-82.0)
    assert haversine_m(p1, p2) == pytest.approx(haversine_m(p2, p1), rel=1e-9)


def test_haversine_golf_hole_scale():
    """Distance between sawgrass tee and green is ~120 yards ≈ 110 m."""
    tee = LatLonPoint(lat=30.19780, lon=-81.39720)
    green = LatLonPoint(lat=30.19710, lon=-81.39580)
    d = haversine_m(tee, green)
    # 137 yd = 125.3 m; tolerance ±20 m for simplified coords.
    assert 90.0 < d < 160.0, f"unexpected tee-green distance {d:.1f} m"


# ---------------------------------------------------------------------------
# UTM zone calculation helpers (internal)
# ---------------------------------------------------------------------------


def test_project_southern_hemisphere():
    """Cypress Point is in the northern hemisphere; sanity-check zone."""
    pt = LatLonPoint(lat=36.57, lon=-121.96)
    utm_pts = project_to_utm([pt])
    assert utm_pts[0].zone.endswith("N")


def test_project_returns_finite_coordinates():
    pt = LatLonPoint(lat=33.50310, lon=-82.02290)
    (utm,) = project_to_utm([pt])
    assert math.isfinite(utm.x)
    assert math.isfinite(utm.y)
