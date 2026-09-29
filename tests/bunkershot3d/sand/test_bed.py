"""Bunker bed geometry tests (issue #8610)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from bunkershot3d.sand.bed import (
    USGA_FACE_DEPTH_RANGE_M,
    USGA_FLOOR_DEPTH_RANGE_M,
    BedZone,
    BunkerBedGeometry,
)
from bunkershot3d.sand.exceptions import BedGeometryError

pytestmark = pytest.mark.unit


class TestUsgaDepthBands:
    def test_published_ranges(self) -> None:
        assert USGA_FLOOR_DEPTH_RANGE_M == (0.100, 0.150)
        assert USGA_FACE_DEPTH_RANGE_M == (0.050, 0.075)

    @pytest.mark.parametrize(
        ("zone", "depth_m", "expected"),
        [
            (BedZone.FLOOR, 0.125, True),
            (BedZone.FLOOR, 0.080, False),
            (BedZone.FLOOR, 0.220, False),
            (BedZone.FACE, 0.0625, True),
            (BedZone.FACE, 0.125, False),
        ],
    )
    def test_depth_is_checked_against_the_zone(
        self, zone: BedZone, depth_m: float, expected: bool
    ) -> None:
        bed = BunkerBedGeometry(
            depth_m=depth_m, plan_length_m=0.4, plan_width_m=0.3, zone=zone
        )
        assert bed.is_within_usga_depth is expected

    def test_out_of_band_depth_is_advisory_not_fatal(self) -> None:
        """USGA depth is a recommendation; the trend is toward deeper beds."""
        bed = BunkerBedGeometry(depth_m=0.250, plan_length_m=0.4, plan_width_m=0.3)
        assert not bed.is_within_usga_depth
        advisory = bed.depth_advisory()
        assert "250" in advisory or "0.25" in advisory
        assert "USGA" in advisory


class TestGeometry:
    def test_plan_area_and_volume(self) -> None:
        bed = BunkerBedGeometry(depth_m=0.1, plan_length_m=0.4, plan_width_m=0.3)
        assert bed.plan_area_m2 == pytest.approx(0.12)
        assert bed.bulk_volume_m3 == pytest.approx(0.012)

    def test_flat_surface_profile_is_zero_everywhere(self) -> None:
        bed = BunkerBedGeometry(depth_m=0.1, plan_length_m=0.4, plan_width_m=0.3)
        assert bed.surface_height_m(0.1, 0.05) == pytest.approx(0.0)

    def test_sloped_surface_rises_along_the_azimuth(self) -> None:
        slope = math.radians(15.0)
        bed = BunkerBedGeometry(
            depth_m=0.06,
            plan_length_m=0.4,
            plan_width_m=0.3,
            zone=BedZone.FACE,
            surface_slope_rad=slope,
            surface_azimuth_rad=0.0,
        )
        assert bed.surface_height_m(1.0, 0.0) == pytest.approx(math.tan(slope))
        assert bed.surface_height_m(0.0, 1.0) == pytest.approx(0.0)

    def test_surface_profile_is_vectorised(self) -> None:
        bed = BunkerBedGeometry(
            depth_m=0.06,
            plan_length_m=0.4,
            plan_width_m=0.3,
            surface_slope_rad=math.radians(10.0),
        )
        x = np.array([0.0, 0.1, 0.2])
        heights = bed.surface_height_m(x, np.zeros_like(x))
        assert heights.shape == x.shape
        assert np.all(np.diff(heights) > 0.0)

    def test_stance_slope_is_independent_of_the_surface_slope(self) -> None:
        bed = BunkerBedGeometry(
            depth_m=0.06,
            plan_length_m=0.4,
            plan_width_m=0.3,
            surface_slope_rad=math.radians(12.0),
            stance_slope_rad=math.radians(-5.0),
        )
        assert bed.stance_slope_deg == pytest.approx(-5.0)
        assert bed.surface_slope_deg == pytest.approx(12.0)

    def test_state_is_frozen(self) -> None:
        bed = BunkerBedGeometry(depth_m=0.1, plan_length_m=0.4, plan_width_m=0.3)
        with pytest.raises((AttributeError, TypeError)):
            bed.depth_m = 0.2  # type: ignore[misc]


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"depth_m": 0.0}, "depth"),
            ({"depth_m": -0.1}, "depth"),
            ({"plan_length_m": 0.0}, "plan"),
            ({"plan_width_m": -1.0}, "plan"),
            ({"surface_slope_rad": math.radians(75.0)}, "slope"),
            ({"stance_slope_rad": math.radians(-75.0)}, "slope"),
        ],
    )
    def test_invalid_geometry_raises(self, kwargs: dict, match: str) -> None:
        base = {"depth_m": 0.1, "plan_length_m": 0.4, "plan_width_m": 0.3}
        base.update(kwargs)
        with pytest.raises(BedGeometryError, match=match):
            BunkerBedGeometry(**base)

    def test_non_finite_depth_raises(self) -> None:
        with pytest.raises(BedGeometryError, match="finite"):
            BunkerBedGeometry(depth_m=float("nan"), plan_length_m=0.4, plan_width_m=0.3)
