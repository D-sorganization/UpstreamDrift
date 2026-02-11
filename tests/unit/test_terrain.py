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


class TestTerrainType:
    """Test terrain type enumeration."""

    def test_terrain_types_exist(self) -> None:
        """Verify all expected terrain types are defined."""
        assert TerrainType.FAIRWAY is not None
        assert TerrainType.ROUGH is not None
        assert TerrainType.GREEN is not None
        assert TerrainType.BUNKER is not None
        assert TerrainType.TEE is not None
        assert TerrainType.FRINGE is not None
        assert TerrainType.WATER is not None
        assert TerrainType.CART_PATH is not None

    def test_terrain_type_values_unique(self) -> None:
        """Each terrain type should have a unique value."""
        values = [t.value for t in TerrainType]
        assert len(values) == len(set(values))


class TestSurfaceMaterial:
    """Test surface material properties."""

    def test_surface_material_creation(self) -> None:
        """Create a surface material with all properties."""
        material = SurfaceMaterial(
            name="fairway_grass",
            friction_coefficient=0.5,
            rolling_resistance=0.1,
            restitution=0.6,
            hardness=0.7,
            grass_height_m=0.015,
        )

        assert material.name == "fairway_grass"
        assert material.friction_coefficient == 0.5
        assert material.rolling_resistance == 0.1
        assert material.restitution == 0.6
        assert material.hardness == 0.7
        assert material.grass_height_m == 0.015

    def test_surface_material_default_values(self) -> None:
        """Surface material should have sensible defaults."""
        material = SurfaceMaterial(name="test")

        # Friction should be positive
        assert material.friction_coefficient > 0
        # Rolling resistance should be non-negative
        assert material.rolling_resistance >= 0
        # Restitution should be between 0 and 1
        assert 0 <= material.restitution <= 1
        # Hardness should be between 0 and 1
        assert 0 <= material.hardness <= 1
        # Grass height defaults to 0
        assert material.grass_height_m >= 0

    def test_surface_material_validation(self) -> None:
        """Invalid material properties should raise errors."""
        with pytest.raises(ValueError, match="friction"):
            SurfaceMaterial(name="invalid", friction_coefficient=-0.5)

        with pytest.raises(ValueError, match="restitution"):
            SurfaceMaterial(name="invalid", restitution=1.5)

        with pytest.raises(ValueError, match="hardness"):
            SurfaceMaterial(name="invalid", hardness=-0.1)

    def test_predefined_materials(self) -> None:
        """Predefined materials for common terrain types should exist."""
        from src.shared.python.physics.terrain import MATERIALS

        assert "fairway" in MATERIALS
        assert "rough" in MATERIALS
        assert "green" in MATERIALS
        assert "bunker" in MATERIALS
        assert "tee" in MATERIALS

        # Bunker should have higher friction (sand)
        assert (
            MATERIALS["bunker"].friction_coefficient
            > MATERIALS["fairway"].friction_coefficient
        )

        # Green should have lower grass
        assert MATERIALS["green"].grass_height_m < MATERIALS["fairway"].grass_height_m


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

        # Approximate height change: 100m * tan(5°) ≈ 8.75m
        expected_rise = 100.0 * math.tan(math.radians(5.0))
        actual_rise = elev.get_elevation(100, 50) - elev.get_elevation(0, 50)
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

    def test_elevation_bounds_checking(self) -> None:
        """Out-of-bounds queries should be handled gracefully."""
        elev = ElevationMap.flat(width=10.0, length=10.0, resolution=1.0)

        # Negative coordinates - should clamp or raise
        with pytest.raises(ValueError):
            elev.get_elevation(-1.0, 5.0)

        # Beyond bounds
        with pytest.raises(ValueError):
            elev.get_elevation(15.0, 5.0)


class TestTerrainPatch:
    """Test individual terrain patches (regions with uniform properties)."""

    def test_terrain_patch_creation(self) -> None:
        """Create a terrain patch with specified properties."""
        patch = TerrainPatch(
            terrain_type=TerrainType.FAIRWAY,
            x_min=0.0,
            x_max=100.0,
            y_min=0.0,
            y_max=50.0,
        )

        assert patch.terrain_type == TerrainType.FAIRWAY
        assert patch.x_min == 0.0
        assert patch.x_max == 100.0
        assert patch.y_min == 0.0
        assert patch.y_max == 50.0

    def test_terrain_patch_contains(self) -> None:
        """Check if a point is within the patch."""
        patch = TerrainPatch(
            terrain_type=TerrainType.GREEN,
            x_min=50.0,
            x_max=70.0,
            y_min=20.0,
            y_max=40.0,
        )

        assert patch.contains(60.0, 30.0)
        assert not patch.contains(0.0, 0.0)
        assert not patch.contains(60.0, 50.0)

    def test_terrain_patch_with_custom_material(self) -> None:
        """Patch can override default material for its terrain type."""
        custom_material = SurfaceMaterial(
            name="wet_fairway",
            friction_coefficient=0.4,  # Lower due to moisture
            rolling_resistance=0.15,
            restitution=0.5,
            hardness=0.6,
            grass_height_m=0.02,
        )

        patch = TerrainPatch(
            terrain_type=TerrainType.FAIRWAY,
            x_min=0.0,
            x_max=100.0,
            y_min=0.0,
            y_max=50.0,
            material=custom_material,
        )

        assert patch.material.name == "wet_fairway"
        assert patch.material.friction_coefficient == 0.4


class TestTerrainRegion:
    """Test terrain regions with complex shapes."""

    def test_circular_region(self) -> None:
        """Create a circular terrain region (e.g., green)."""
        region = TerrainRegion.circle(
            terrain_type=TerrainType.GREEN,
            center_x=100.0,
            center_y=100.0,
            radius=15.0,
        )

        assert region.contains(100.0, 100.0)  # Center
        assert region.contains(110.0, 100.0)  # On edge
        assert not region.contains(120.0, 100.0)  # Outside

    def test_polygon_region(self) -> None:
        """Create a polygon-shaped terrain region."""
        # Triangle
        vertices = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
        region = TerrainRegion.polygon(
            terrain_type=TerrainType.BUNKER,
            vertices=vertices,
        )

        assert region.contains(5.0, 3.0)  # Inside triangle
        assert not region.contains(0.0, 10.0)  # Outside


class TestTerrain:
    """Test complete terrain configuration."""

    def test_terrain_creation(self) -> None:
        """Create a complete terrain with elevation and patches."""
        elevation = ElevationMap.flat(width=200.0, length=400.0, resolution=1.0)

        patches = [
            TerrainPatch(TerrainType.TEE, 10.0, 30.0, 190.0, 210.0),
            TerrainPatch(TerrainType.FAIRWAY, 30.0, 180.0, 150.0, 250.0),
            TerrainPatch(TerrainType.GREEN, 180.0, 200.0, 190.0, 210.0),
        ]

        terrain = Terrain(
            name="Test Hole",
            elevation=elevation,
            patches=patches,
        )

        assert terrain.name == "Test Hole"
        assert terrain.elevation.width == 200.0
        assert len(terrain.patches) == 3

    def test_terrain_type_at_point(self) -> None:
        """Query terrain type at a given position."""
        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [
            TerrainPatch(TerrainType.FAIRWAY, 0.0, 100.0, 0.0, 100.0),
            TerrainPatch(
                TerrainType.GREEN, 80.0, 100.0, 40.0, 60.0
            ),  # Overlaps fairway
        ]

        terrain = Terrain(name="Test", elevation=elevation, patches=patches)

        # Green patch should take priority (defined last)
        assert terrain.get_terrain_type(90.0, 50.0) == TerrainType.GREEN

        # Fairway elsewhere
        assert terrain.get_terrain_type(50.0, 50.0) == TerrainType.FAIRWAY

    def test_terrain_properties_at_point(self) -> None:
        """Get all terrain properties at a point."""
        elevation = ElevationMap.sloped(
            width=100.0,
            length=100.0,
            resolution=1.0,
            slope_angle_deg=5.0,
            slope_direction_deg=0.0,
        )
        patches = [TerrainPatch(TerrainType.FAIRWAY, 0.0, 100.0, 0.0, 100.0)]

        terrain = Terrain(name="Test", elevation=elevation, patches=patches)

        props = terrain.get_properties_at(50.0, 50.0)

        assert "elevation" in props
        assert "gradient" in props
        assert "normal" in props
        assert "terrain_type" in props
        assert "material" in props
        assert props["terrain_type"] == TerrainType.FAIRWAY

    def test_terrain_contact_parameters(self) -> None:
        """Get physics-engine-ready contact parameters."""
        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [TerrainPatch(TerrainType.BUNKER, 0.0, 100.0, 0.0, 100.0)]

        terrain = Terrain(name="Test", elevation=elevation, patches=patches)

        contact = terrain.get_contact_params(50.0, 50.0)

        assert "friction" in contact
        assert "restitution" in contact
        assert "stiffness" in contact
        assert "damping" in contact

        # Bunker should have higher friction
        assert contact["friction"] > 0.5


class TestTerrainConfig:
    """Test terrain configuration loading/saving."""

    def test_terrain_config_to_dict(self) -> None:
        """Terrain config should be serializable to dict."""
        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [TerrainPatch(TerrainType.FAIRWAY, 0.0, 100.0, 0.0, 100.0)]
        terrain = Terrain(name="Test", elevation=elevation, patches=patches)

        config = TerrainConfig.from_terrain(terrain)
        data = config.to_dict()

        assert "name" in data
        assert "elevation" in data
        assert "patches" in data
        assert data["name"] == "Test"

    def test_terrain_config_to_json(self, tmp_path: Path) -> None:
        """Terrain config should save to JSON."""
        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [TerrainPatch(TerrainType.FAIRWAY, 0.0, 100.0, 0.0, 100.0)]
        terrain = Terrain(name="Test", elevation=elevation, patches=patches)

        config = TerrainConfig.from_terrain(terrain)
        json_path = tmp_path / "terrain.json"
        config.save(json_path)

        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert data["name"] == "Test"

    def test_terrain_config_from_json(self, tmp_path: Path) -> None:
        """Terrain config should load from JSON."""
        # Create JSON file
        json_data = {
            "name": "LoadedTerrain",
            "elevation": {
                "type": "flat",
                "width": 100.0,
                "length": 100.0,
                "resolution": 1.0,
            },
            "patches": [
                {
                    "terrain_type": "fairway",
                    "x_min": 0.0,
                    "x_max": 100.0,
                    "y_min": 0.0,
                    "y_max": 100.0,
                }
            ],
        }
        json_path = tmp_path / "terrain.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        config = TerrainConfig.load(json_path)
        terrain = config.to_terrain()

        assert terrain.name == "LoadedTerrain"
        assert len(terrain.patches) == 1


class TestTerrainFactories:
    """Test factory functions for common terrain configurations."""

    def test_create_flat_terrain(self) -> None:
        """Create a simple flat terrain."""
        terrain = create_flat_terrain(
            name="Flat Test",
            width=100.0,
            length=200.0,
            terrain_type=TerrainType.FAIRWAY,
        )

        assert terrain.name == "Flat Test"
        assert terrain.elevation.width == 100.0
        assert terrain.elevation.length == 200.0
        assert np.allclose(terrain.elevation.data, 0.0)

    def test_create_sloped_terrain(self) -> None:
        """Create a uniformly sloped terrain."""
        terrain = create_sloped_terrain(
            name="Sloped Test",
            width=100.0,
            length=200.0,
            slope_angle_deg=3.0,
            slope_direction_deg=90.0,  # Slope in Y direction
            terrain_type=TerrainType.FAIRWAY,
        )

        # Check slope exists in Y direction
        h1 = terrain.elevation.get_elevation(50.0, 0.0)
        h2 = terrain.elevation.get_elevation(50.0, 200.0)
        assert h2 > h1

    def test_create_terrain_from_config(self, tmp_path: Path) -> None:
        """Create terrain from config file."""
        config_data = {
            "name": "ConfigTest",
            "elevation": {
                "type": "sloped",
                "width": 50.0,
                "length": 100.0,
                "resolution": 0.5,
                "slope_angle_deg": 2.0,
                "slope_direction_deg": 0.0,
            },
            "patches": [
                {
                    "terrain_type": "tee",
                    "x_min": 0.0,
                    "x_max": 10.0,
                    "y_min": 20.0,
                    "y_max": 30.0,
                },
                {
                    "terrain_type": "fairway",
                    "x_min": 10.0,
                    "x_max": 50.0,
                    "y_min": 0.0,
                    "y_max": 100.0,
                },
            ],
        }
        config_path = tmp_path / "terrain_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        terrain = create_terrain_from_config(config_path)

        assert terrain.name == "ConfigTest"
        assert len(terrain.patches) == 2


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


class TestTerrainEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_size_terrain(self) -> None:
        """Zero-sized terrain should raise error."""
        with pytest.raises(ValueError):
            ElevationMap.flat(width=0.0, length=100.0, resolution=1.0)

    def test_negative_resolution(self) -> None:
        """Negative resolution should raise error."""
        with pytest.raises(ValueError):
            ElevationMap.flat(width=100.0, length=100.0, resolution=-1.0)

    def test_steep_slope(self) -> None:
        """Very steep slopes (>45°) should work but warn."""
        elev = ElevationMap.sloped(
            width=100.0,
            length=100.0,
            resolution=1.0,
            slope_angle_deg=60.0,
            slope_direction_deg=0.0,
        )

        # Should still work
        assert elev.get_elevation(50, 50) > 0

    def test_multiple_overlapping_patches(self) -> None:
        """Later patches should override earlier ones."""
        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [
            TerrainPatch(TerrainType.ROUGH, 0.0, 100.0, 0.0, 100.0),
            TerrainPatch(TerrainType.FAIRWAY, 20.0, 80.0, 20.0, 80.0),
            TerrainPatch(TerrainType.GREEN, 60.0, 80.0, 40.0, 60.0),
        ]

        terrain = Terrain(name="Test", elevation=elevation, patches=patches)

        # Check priority (last defined wins)
        assert terrain.get_terrain_type(70.0, 50.0) == TerrainType.GREEN
        assert terrain.get_terrain_type(50.0, 50.0) == TerrainType.FAIRWAY
        assert terrain.get_terrain_type(10.0, 10.0) == TerrainType.ROUGH

    def test_point_outside_all_patches(self) -> None:
        """Point outside all patches returns default terrain type."""
        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [TerrainPatch(TerrainType.GREEN, 40.0, 60.0, 40.0, 60.0)]

        terrain = Terrain(
            name="Test",
            elevation=elevation,
            patches=patches,
            default_type=TerrainType.ROUGH,
        )

        assert terrain.get_terrain_type(10.0, 10.0) == TerrainType.ROUGH
