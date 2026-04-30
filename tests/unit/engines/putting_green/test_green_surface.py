"""Unit tests for GreenSurface module.

TDD Tests - These tests define the expected behavior of the GreenSurface
module for handling slopes, undulations, and surface contours.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.putting_green.python.green_surface import (
    ContourPoint,
    GreenSurface,
    SlopeRegion,
)
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)


class TestContourPoint:
    """Tests for ContourPoint dataclass."""

    def test_contour_point_creation(self) -> None:
        """ContourPoint should store position and elevation."""
        point = ContourPoint(x=1.0, y=2.0, elevation=0.05)
        assert point.x == 1.0
        assert point.y == 2.0
        assert point.elevation == 0.05

    def test_contour_point_as_array(self) -> None:
        """Should convert to numpy array easily."""
        point = ContourPoint(x=1.0, y=2.0, elevation=0.05)
        arr = point.as_array()
        assert np.allclose(arr, [1.0, 2.0, 0.05])


class TestSlopeRegion:
    """Tests for SlopeRegion definition."""

    def test_slope_region_creation(self) -> None:
        """SlopeRegion should define an area with slope properties."""
        region = SlopeRegion(
            center=np.array([5.0, 5.0]),
            radius=2.0,
            slope_direction=np.array([1.0, 0.0]),
            slope_magnitude=0.02,  # 2% slope
        )
        assert np.allclose(region.center, [5.0, 5.0])
        assert region.radius == 2.0
        assert region.slope_magnitude == 0.02

    def test_slope_direction_normalized(self) -> None:
        """Slope direction should be automatically normalized."""
        region = SlopeRegion(
            center=np.array([0.0, 0.0]),
            radius=1.0,
            slope_direction=np.array([3.0, 4.0]),
            slope_magnitude=0.01,
        )
        mag = np.linalg.norm(region.slope_direction)
        assert np.isclose(mag, 1.0, atol=1e-6)

    def test_point_in_region(self) -> None:
        """Should correctly identify points inside region."""
        region = SlopeRegion(
            center=np.array([5.0, 5.0]),
            radius=2.0,
            slope_direction=np.array([1.0, 0.0]),
            slope_magnitude=0.02,
        )
        assert region.contains(np.array([5.0, 5.0]))  # Center
        assert region.contains(np.array([6.0, 5.0]))  # Inside
        assert not region.contains(np.array([10.0, 10.0]))  # Outside


class TestGreenSurface:
    """Tests for GreenSurface class."""

    @pytest.fixture
    def flat_green(self) -> GreenSurface:
        """Create a flat putting green."""
        return GreenSurface(
            width=20.0,  # meters
            height=20.0,
            turf=TurfProperties.create_preset("tournament_fast"),
        )

    @pytest.fixture
    def sloped_green(self) -> GreenSurface:
        """Create a green with a single slope."""
        green = GreenSurface(
            width=20.0,
            height=20.0,
            turf=TurfProperties.create_preset("tournament_fast"),
        )
        green.add_slope_region(
            SlopeRegion(
                center=np.array([10.0, 10.0]),
                radius=5.0,
                slope_direction=np.array([1.0, 0.0]),
                slope_magnitude=0.03,  # 3% slope
            )
        )
        return green

    @pytest.fixture
    def contoured_green(self) -> GreenSurface:
        """Create a green with elevation contours."""
        green = GreenSurface(
            width=20.0,
            height=20.0,
            turf=TurfProperties(),
        )
        # Add some contour points for undulation
        contours = [
            ContourPoint(5.0, 5.0, 0.1),
            ContourPoint(15.0, 5.0, 0.05),
            ContourPoint(10.0, 10.0, 0.15),
            ContourPoint(5.0, 15.0, 0.08),
            ContourPoint(15.0, 15.0, 0.02),
        ]
        green.set_contour_points(contours)
        return green

    def test_green_dimensions(self, flat_green: GreenSurface) -> None:
        """Green should have correct dimensions."""
        assert flat_green.width == 20.0
        assert flat_green.height == 20.0

    def test_flat_green_no_slope(self, flat_green: GreenSurface) -> None:
        """Flat green should have zero slope everywhere."""
        positions = [
            np.array([5.0, 5.0]),
            np.array([10.0, 10.0]),
            np.array([15.0, 15.0]),
        ]
        for pos in positions:
            slope = flat_green.get_slope_at(pos)
            assert np.allclose(slope, [0.0, 0.0])

    def test_sloped_green_has_slope(self, sloped_green: GreenSurface) -> None:
        """Sloped region should return correct slope."""
        pos = np.array([10.0, 10.0])  # Center of slope region
        slope = sloped_green.get_slope_at(pos)

        # Should have slope in x-direction
        assert slope[0] != 0
        assert np.isclose(np.linalg.norm(slope), 0.03, atol=0.01)

    def test_slope_outside_region(self, sloped_green: GreenSurface) -> None:
        """Outside slope region should be flat."""
        pos = np.array([1.0, 1.0])  # Far from slope region
        slope = sloped_green.get_slope_at(pos)
        assert np.allclose(slope, [0.0, 0.0])

    def test_get_elevation_at(self, contoured_green: GreenSurface) -> None:
        """Should interpolate elevation from contour points."""
        # At a contour point, should return exact value
        elevation = contoured_green.get_elevation_at(np.array([10.0, 10.0]))
        assert np.isclose(elevation, 0.15, atol=0.01)

    def test_gradient_from_elevation(self, contoured_green: GreenSurface) -> None:
        """Gradient should be computed from elevation map."""
        pos = np.array([10.0, 10.0])
        gradient = contoured_green.get_gradient_at(pos)

        # Should return 2D gradient vector
        assert gradient.shape == (2,)
        assert np.all(np.isfinite(gradient))

    def test_gravitational_acceleration_on_slope(
        self, sloped_green: GreenSurface
    ) -> None:
        """Should compute gravitational acceleration from slope."""
        pos = np.array([10.0, 10.0])
        g_accel = sloped_green.get_gravitational_acceleration(pos)

        # On a slope, there should be gravitational component
        assert np.linalg.norm(g_accel) > 0
        # Direction should be downhill (opposite to slope direction)
        slope = sloped_green.get_slope_at(pos)
        if np.linalg.norm(slope) > 0:
            assert np.dot(g_accel, slope) < 0  # Downhill

    def test_hole_position(self, flat_green: GreenSurface) -> None:
        """Should have configurable hole position."""
        flat_green.set_hole_position(np.array([15.0, 15.0]))
        assert np.allclose(flat_green.hole_position, [15.0, 15.0])

    def test_hole_radius(self, flat_green: GreenSurface) -> None:
        """Hole should have standard radius (4.25 inches = 0.108m)."""
        assert np.isclose(flat_green.hole_radius, 0.054, atol=0.001)  # radius

    def test_is_in_hole(self, flat_green: GreenSurface) -> None:
        """Should detect when ball is in hole."""
        flat_green.set_hole_position(np.array([10.0, 10.0]))

        # Ball in hole
        assert flat_green.is_in_hole(np.array([10.0, 10.0]))
        assert flat_green.is_in_hole(np.array([10.02, 10.02]))  # Just inside

        # Ball outside hole
        assert not flat_green.is_in_hole(np.array([10.1, 10.1]))

    def test_is_on_green(self, flat_green: GreenSurface) -> None:
        """Should detect if position is on the green."""
        assert flat_green.is_on_green(np.array([10.0, 10.0]))
        assert flat_green.is_on_green(np.array([0.0, 0.0]))  # Edge

        # Off green
        assert not flat_green.is_on_green(np.array([-1.0, 10.0]))
        assert not flat_green.is_on_green(np.array([25.0, 10.0]))

    def test_multiple_slope_regions(self, flat_green: GreenSurface) -> None:
        """Should handle multiple overlapping slope regions."""
        flat_green.add_slope_region(
            SlopeRegion(
                center=np.array([5.0, 5.0]),
                radius=3.0,
                slope_direction=np.array([1.0, 0.0]),
                slope_magnitude=0.02,
            )
        )
        flat_green.add_slope_region(
            SlopeRegion(
                center=np.array([5.0, 5.0]),
                radius=3.0,
                slope_direction=np.array([0.0, 1.0]),
                slope_magnitude=0.01,
            )
        )

        slope = flat_green.get_slope_at(np.array([5.0, 5.0]))
        # Should combine slopes
        assert slope[0] != 0
        assert slope[1] != 0

    def test_add_ridge(self, flat_green: GreenSurface) -> None:
        """Should support adding ridge features."""
        flat_green.add_ridge(
            start=np.array([5.0, 5.0]),
            end=np.array([15.0, 15.0]),
            height=0.05,
            width=1.0,
        )

        # Point on ridge should have higher elevation
        elev_on_ridge = flat_green.get_elevation_at(np.array([10.0, 10.0]))
        elev_off_ridge = flat_green.get_elevation_at(np.array([2.0, 2.0]))
        assert elev_on_ridge > elev_off_ridge

    def test_add_depression(self, flat_green: GreenSurface) -> None:
        """Should support adding depression (hollow) features."""
        flat_green.add_depression(
            center=np.array([10.0, 10.0]),
            radius=2.0,
            depth=0.03,
        )

        elev_in_depression = flat_green.get_elevation_at(np.array([10.0, 10.0]))
        elev_outside = flat_green.get_elevation_at(np.array([1.0, 1.0]))
        assert elev_in_depression < elev_outside

    def test_break_calculation(self, sloped_green: GreenSurface) -> None:
        """Should calculate break for a putt line."""
        start = np.array([8.0, 10.0])
        end = np.array([12.0, 10.0])

        break_info = sloped_green.calculate_break(start, end)

        assert "total_break" in break_info
        assert "break_direction" in break_info
        assert "average_slope" in break_info

    def test_read_putt_line(self, sloped_green: GreenSurface) -> None:
        """Should provide putt reading along a line."""
        start = np.array([5.0, 10.0])
        end = np.array([15.0, 10.0])

        reading = sloped_green.read_putt_line(start, end, num_samples=10)

        assert len(reading["positions"]) == 10
        assert len(reading["elevations"]) == 10
        assert len(reading["slopes"]) == 10

    def test_create_from_heightmap(self) -> None:
        """Should create green from 2D heightmap array."""
        heightmap = np.random.rand(100, 100) * 0.1  # 0-10cm variation
        green = GreenSurface.from_heightmap(
            heightmap,
            width=20.0,
            height=20.0,
            turf=TurfProperties(),
        )

        assert green.width == 20.0
        assert green.height == 20.0
        # Check that elevation queries work
        elev = green.get_elevation_at(np.array([10.0, 10.0]))
        assert 0 <= elev <= 0.1

    def test_to_heightmap_export(self, contoured_green: GreenSurface) -> None:
        """Should export to heightmap array."""
        heightmap = contoured_green.to_heightmap(resolution=50)

        assert heightmap.shape == (50, 50)
        assert np.all(np.isfinite(heightmap))

    def test_surface_bounds_checking(self, flat_green: GreenSurface) -> None:
        """Should handle out-of-bounds positions gracefully."""
        pos = np.array([100.0, 100.0])  # Way outside

        # Should not crash, return edge values or zeros
        slope = flat_green.get_slope_at(pos)
        assert np.all(np.isfinite(slope))

    def test_turf_property_access(self, flat_green: GreenSurface) -> None:
        """Should provide access to turf properties."""
        assert flat_green.turf is not None
        assert flat_green.turf.stimp_rating > 0


class TestGreenSurfacePresets:
    """Tests for preset green configurations."""

    def test_flat_practice_green(self) -> None:
        """Flat practice green preset."""
        green = GreenSurface.create_preset("flat_practice")
        assert green.width >= 10
        # Should be flat
        slope = green.get_slope_at(np.array([5.0, 5.0]))
        assert np.allclose(slope, [0.0, 0.0])

    def test_undulating_championship(self) -> None:
        """Championship undulating green preset."""
        green = GreenSurface.create_preset("undulating_championship")

        # Should have some slope variation
        slopes = [
            green.get_slope_at(np.array([5.0, 5.0])),
            green.get_slope_at(np.array([10.0, 10.0])),
            green.get_slope_at(np.array([15.0, 15.0])),
        ]
        # At least some should be non-zero
        has_slopes = any(np.linalg.norm(s) > 0.001 for s in slopes)
        assert has_slopes

    def test_severe_slope_green(self) -> None:
        """Green with severe slopes (like Augusta)."""
        green = GreenSurface.create_preset("severe_slopes")

        # Should have significant slopes
        max_slope = 0.0
        for x in np.linspace(2, 18, 10):
            for y in np.linspace(2, 18, 10):
                slope = green.get_slope_at(np.array([x, y]))
                max_slope = max(max_slope, np.linalg.norm(slope))

        assert max_slope >= 0.03  # At least 3% slope somewhere
