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


class TestTerrainPhysicsIntegration:
    """Test terrain integration with physics calculations."""

    def test_gravity_on_slope(self) -> None:
        """Calculate gravity component along slope."""
        from src.shared.python.physics.terrain import compute_gravity_on_slope

        # 30 degree slope
        slope_angle_deg = 30.0
        gravity = 9.81

        g_parallel, g_perpendicular = compute_gravity_on_slope(slope_angle_deg, gravity)

        # g_parallel = g * sin(θ)
        expected_parallel = gravity * math.sin(math.radians(slope_angle_deg))
        assert abs(g_parallel - expected_parallel) < 1e-6

        # g_perpendicular = g * cos(θ)
        expected_perpendicular = gravity * math.cos(math.radians(slope_angle_deg))
        assert abs(g_perpendicular - expected_perpendicular) < 1e-6

    def test_ball_roll_direction(self) -> None:
        """Calculate ball roll direction on sloped terrain."""
        from src.shared.python.physics.terrain import compute_roll_direction

        elevation = ElevationMap.sloped(
            width=100.0,
            length=100.0,
            resolution=1.0,
            slope_angle_deg=5.0,
            slope_direction_deg=0.0,  # Slope in +X
        )

        # Ball should roll in -X direction (downhill)
        roll_dir = compute_roll_direction(elevation, 50.0, 50.0)

        assert roll_dir[0] < 0  # Roll in -X
        assert abs(roll_dir[1]) < 0.01  # Minimal Y component

    def test_contact_normal_on_terrain(self) -> None:
        """Get contact normal for physics engine."""
        from src.shared.python.physics.terrain import get_contact_normal

        elevation = ElevationMap.sloped(
            width=100.0,
            length=100.0,
            resolution=1.0,
            slope_angle_deg=15.0,
            slope_direction_deg=45.0,  # Slope in XY diagonal
        )

        normal = get_contact_normal(elevation, 50.0, 50.0)

        # Should be unit vector
        assert abs(np.linalg.norm(normal) - 1.0) < 1e-6

        # Should point mostly up
        assert normal[2] > 0.9
