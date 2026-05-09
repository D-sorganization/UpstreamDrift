"""Unit tests for terrain modeling system (TDD: Tests First).

Following the Pragmatic Programmer principles:
- DRY (Don't Repeat Yourself)
- Design by Contract
- Orthogonality
- Test-Driven Development

These tests define the expected behavior BEFORE implementation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

# Tests written first - imports will fail until implementation exists
from src.shared.python.physics.terrain import (
    ElevationMap,
    SurfaceMaterial,
    Terrain,
    TerrainConfig,
    TerrainPatch,
    TerrainRegion,
    TerrainType,
    create_flat_terrain,
    create_sloped_terrain,
    create_terrain_from_config,
)


class TestElevationMap:
    """Test elevation/height map functionality."""

    def test_flat_elevation_map(self) -> None:
        """Create a flat elevation map."""
        elev = ElevationMap.flat(width=100.0, length=200.0, resolution=1.0)

        assert elev.width == 100.0
        assert elev.length == 200.0
        assert elev.resolution == 1.0
        assert elev.data.shape == (200, 100)  # (rows, cols)
        assert np.allclose(elev.data, 0.0)

    def test_sloped_elevation_map(self) -> None:
        """Create a uniformly sloped terrain."""
        # 5 degree slope in X direction
        elev = ElevationMap.sloped(
            width=100.0,
            length=100.0,
            resolution=1.0,
            slope_angle_deg=5.0,
            slope_direction_deg=0.0,  # Slope in +X direction
        )

        # Height should increase in X direction
        assert elev.get_elevation(50, 50) > elev.get_elevation(0, 50)

        # Grid has 100 nodes at resolution 1m: valid range is [0, 99].
        # Height change over 99 nodes: 99m * tan(5°) ≈ 8.66m
        expected_rise = 99.0 * math.tan(math.radians(5.0))
        actual_rise = elev.get_elevation(99, 50) - elev.get_elevation(0, 50)
        assert abs(actual_rise - expected_rise) < 0.1

    def test_elevation_interpolation(self) -> None:
        """Elevation queries should interpolate between grid points."""
        elev = ElevationMap.sloped(
            width=10.0,
            length=10.0,
            resolution=1.0,
            slope_angle_deg=10.0,
            slope_direction_deg=0.0,
        )

        # Query at non-grid point
        h_interp = elev.get_elevation(5.5, 5.5)
        h_floor = elev.get_elevation(5.0, 5.0)
        h_ceil = elev.get_elevation(6.0, 6.0)

        # Interpolated value should be between neighbors
        assert h_floor <= h_interp <= h_ceil or h_ceil <= h_interp <= h_floor

    def test_elevation_gradient(self) -> None:
        """Get slope/gradient at a point."""
        elev = ElevationMap.sloped(
            width=100.0,
            length=100.0,
            resolution=1.0,
            slope_angle_deg=5.0,
            slope_direction_deg=0.0,
        )

        grad_x, grad_y = elev.get_gradient(50, 50)

        # Should have positive X gradient (uphill in X)
        assert grad_x > 0
        # Y gradient should be near zero (flat in Y)
        assert abs(grad_y) < 0.01

    def test_elevation_normal_vector(self) -> None:
        """Get surface normal at a point."""
        elev = ElevationMap.sloped(
            width=100.0,
            length=100.0,
            resolution=1.0,
            slope_angle_deg=30.0,
            slope_direction_deg=0.0,
        )

        normal = elev.get_normal(50, 50)

        # Normal should be unit vector
        assert abs(np.linalg.norm(normal) - 1.0) < 1e-6

        # Normal should point mostly up (positive Z)
        assert normal[2] > 0

        # Normal should tilt in -X direction (opposite to slope)
        assert normal[0] < 0

    def test_elevation_from_array(self) -> None:
        """Create elevation map from numpy array."""
        data = np.random.rand(50, 100) * 10  # Random elevations 0-10m
        elev = ElevationMap.from_array(data, resolution=0.5)

        assert elev.width == 50.0  # 100 cols * 0.5m
        assert elev.length == 25.0  # 50 rows * 0.5m
        assert elev.resolution == 0.5
        assert np.array_equal(elev.data, data)

    @pytest.mark.parametrize(
        "x,y",
        [(-1.0, 5.0), (15.0, 5.0)],
        ids=["negative_coordinate", "beyond_bounds"],
    )
    def test_elevation_bounds_checking(self, x, y) -> None:
        """Out-of-bounds queries should be handled gracefully."""
        elev = ElevationMap.flat(width=10.0, length=10.0, resolution=1.0)
        with pytest.raises(ValueError):
            elev.get_elevation(x, y)
