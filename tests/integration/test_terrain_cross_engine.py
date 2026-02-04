"""Cross-engine terrain integration tests.

Tests that terrain features work consistently across all physics engines.
Follows Pragmatic Programmer principles for orthogonal, well-tested code.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.terrain import (
    ElevationMap,
    Terrain,
    TerrainPatch,
    TerrainType,
    create_flat_terrain,
    create_sloped_terrain,
)
from src.shared.python.terrain_engine import (
    CompressibleTurfModel,
    TerrainAwareEngine,
    TerrainContactModel,
    TerrainGeometryGenerator,
)


class TestTerrainConsistency:
    """Test terrain calculations are consistent."""

    @pytest.fixture
    def simple_terrain(self) -> Terrain:
        """Create a simple test terrain."""
        return create_flat_terrain("Test", 100.0, 100.0, TerrainType.FAIRWAY)

    @pytest.fixture
    def sloped_terrain(self) -> Terrain:
        """Create a sloped test terrain."""
        return create_sloped_terrain(
            "Sloped",
            100.0,
            100.0,
            slope_angle_deg=10.0,
            slope_direction_deg=0.0,
        )

    @pytest.fixture
    def mixed_terrain(self) -> Terrain:
        """Create a terrain with multiple surface types."""
        elevation = ElevationMap.sloped(
            width=100.0,
            length=100.0,
            resolution=1.0,
            slope_angle_deg=5.0,
            slope_direction_deg=45.0,
        )
        patches = [
            TerrainPatch(TerrainType.TEE, 0.0, 20.0, 40.0, 60.0),
            TerrainPatch(TerrainType.FAIRWAY, 20.0, 80.0, 20.0, 80.0),
            TerrainPatch(TerrainType.BUNKER, 60.0, 70.0, 45.0, 55.0),
            TerrainPatch(TerrainType.GREEN, 80.0, 100.0, 40.0, 60.0),
        ]
        return Terrain(
            name="GolfHole",
            elevation=elevation,
            patches=patches,
            default_type=TerrainType.ROUGH,
        )

    def test_elevation_consistency(self, sloped_terrain: Terrain) -> None:
        """Elevation queries should be consistent."""
        # Query same point multiple times
        heights = [
            sloped_terrain.elevation.get_elevation(50.0, 50.0) for _ in range(10)
        ]

        # All should be identical
        assert all(h == heights[0] for h in heights)

    def test_normal_is_unit_vector(self, sloped_terrain: Terrain) -> None:
        """Normal vectors should always be unit vectors."""
        test_points = [
            (10.0, 10.0),
            (50.0, 50.0),
            (90.0, 90.0),
            (25.0, 75.0),
        ]

        for x, y in test_points:
            normal = sloped_terrain.elevation.get_normal(x, y)
            magnitude = np.linalg.norm(normal)
            assert (
                abs(magnitude - 1.0) < 1e-6
            ), f"Normal at ({x}, {y}) not unit: {magnitude}"

    def test_gradient_consistency(self, sloped_terrain: Terrain) -> None:
        """Gradient should be consistent with elevation differences."""
        x, y = 50.0, 50.0
        dx = 0.01  # Small step

        # Get gradient
        grad_x, grad_y = sloped_terrain.elevation.get_gradient(x, y)

        # Compute numerical gradient
        h_center = sloped_terrain.elevation.get_elevation(x, y)
        h_x = sloped_terrain.elevation.get_elevation(x + dx, y)
        h_y = sloped_terrain.elevation.get_elevation(x, y + dx)

        numerical_grad_x = (h_x - h_center) / dx
        numerical_grad_y = (h_y - h_center) / dx

        # Should be close
        assert abs(grad_x - numerical_grad_x) < 0.1
        assert abs(grad_y - numerical_grad_y) < 0.1

    def test_terrain_type_zones(self, mixed_terrain: Terrain) -> None:
        """Terrain type queries should respect zone boundaries."""
        # Tee box
        assert mixed_terrain.get_terrain_type(10.0, 50.0) == TerrainType.TEE

        # Fairway
        assert mixed_terrain.get_terrain_type(50.0, 50.0) == TerrainType.FAIRWAY

        # Bunker
        assert mixed_terrain.get_terrain_type(65.0, 50.0) == TerrainType.BUNKER

        # Green
        assert mixed_terrain.get_terrain_type(90.0, 50.0) == TerrainType.GREEN

        # Rough (default)
        assert mixed_terrain.get_terrain_type(5.0, 5.0) == TerrainType.ROUGH

    def test_contact_parameters_vary_by_terrain(self, mixed_terrain: Terrain) -> None:
        """Contact parameters should vary by terrain type."""
        # Get contact params at different locations
        tee_params = mixed_terrain.get_contact_params(10.0, 50.0)
        fairway_params = mixed_terrain.get_contact_params(50.0, 50.0)
        bunker_params = mixed_terrain.get_contact_params(65.0, 50.0)
        green_params = mixed_terrain.get_contact_params(90.0, 50.0)

        # Bunker should have highest friction
        assert bunker_params["friction"] > fairway_params["friction"]

        # Green should have highest restitution
        assert green_params["restitution"] > bunker_params["restitution"]


class TestTerrainContactPhysics:
    """Test terrain contact physics calculations."""

    def test_contact_force_direction(self) -> None:
        """Contact force should point away from terrain."""
        terrain = create_sloped_terrain(
            "Sloped",
            100.0,
            100.0,
            slope_angle_deg=20.0,
            slope_direction_deg=0.0,
        )
        contact = TerrainContactModel(terrain)

        # Ball penetrating terrain
        force = contact.compute_contact_force(50.0, 50.0, z=0.0, radius=0.02)

        # Force should point up and away from slope
        assert force[2] > 0, "Force should have upward component"

        # On a slope in +X direction, force tilts in -X
        normal = terrain.elevation.get_normal(50.0, 50.0)
        # Force direction should align with normal
        force_dir = force / np.linalg.norm(force)
        alignment = np.dot(force_dir, normal)
        assert alignment > 0.99, "Force should align with normal"

    def test_friction_magnitude(self) -> None:
        """Friction force should be bounded by mu * N."""
        terrain = create_flat_terrain("Flat", 100.0, 100.0, TerrainType.FAIRWAY)
        contact = TerrainContactModel(terrain)

        # Ball in contact with velocity
        velocity = np.array([5.0, 0.0, 0.0])
        normal_force = contact.compute_contact_force(50.0, 50.0, z=-0.01, radius=0.02)
        friction = contact.compute_friction_force(
            50.0, 50.0, z=-0.01, radius=0.02, velocity=velocity
        )

        # Get friction coefficient
        mu = terrain.get_material(50.0, 50.0).friction_coefficient
        N = np.linalg.norm(normal_force)

        # Friction magnitude should be <= mu * N
        F_friction = np.linalg.norm(friction)
        assert F_friction <= mu * N * 1.01  # Small tolerance

    def test_energy_conservation_approximation(self) -> None:
        """Energy absorbed + remaining should equal initial (approximately)."""
        terrain = create_flat_terrain("Green", 100.0, 100.0, TerrainType.GREEN)
        turf = CompressibleTurfModel(terrain)

        impact_velocity = np.array([15.0, 0.0, -25.0])
        energy = turf.compute_energy_absorption(50.0, 50.0, impact_velocity)

        # Energy should be conserved (absorbed + remaining = initial)
        total = energy["absorbed_energy"] + energy["remaining_energy"]
        assert abs(total - energy["kinetic_energy"]) < 1e-6


class TestTerrainGeometryGeneration:
    """Test terrain geometry generation for physics engines."""

    def test_mesh_topology(self) -> None:
        """Generated mesh should have valid topology."""
        terrain = create_sloped_terrain(
            "Test",
            10.0,
            10.0,
            slope_angle_deg=5.0,
            slope_direction_deg=0.0,
            resolution=1.0,
        )
        generator = TerrainGeometryGenerator(terrain)
        vertices, triangles = generator.generate_mesh()

        # Check vertex count
        n_rows, n_cols = terrain.elevation.data.shape
        expected_vertices = n_rows * n_cols
        assert len(vertices) == expected_vertices

        # Check triangle count (2 triangles per cell)
        expected_triangles = 2 * (n_rows - 1) * (n_cols - 1)
        assert len(triangles) == expected_triangles

        # Check all triangles reference valid vertices
        for tri in triangles:
            for idx in tri:
                assert 0 <= idx < len(vertices)

    def test_mesh_normals_face_up(self) -> None:
        """Mesh triangles should face upward (positive Z normal)."""
        terrain = create_flat_terrain("Test", 10.0, 10.0, resolution=1.0)
        generator = TerrainGeometryGenerator(terrain)
        vertices, triangles = generator.generate_mesh()

        for tri in triangles[:10]:  # Check first 10 triangles
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]

            # Compute normal via cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)

            # Normal should point up (positive Z)
            assert normal[2] >= 0 or abs(normal[2]) < 1e-6

    def test_mujoco_hfield_normalized(self) -> None:
        """MuJoCo heightfield should be normalized to [0, 1]."""
        terrain = create_sloped_terrain(
            "Test",
            10.0,
            10.0,
            slope_angle_deg=15.0,
            slope_direction_deg=0.0,
            resolution=0.5,
        )
        generator = TerrainGeometryGenerator(terrain)
        hfield_data, hfield_size = generator.generate_mujoco_hfield()

        # Values should be in [0, 1]
        assert hfield_data.min() >= 0.0
        assert hfield_data.max() <= 1.0


class TestCompressibleTurfPhysics:
    """Test compressible turf physics model."""

    def test_compression_increases_stiffness(self) -> None:
        """Stiffness should increase with compression (progressive)."""
        terrain = create_flat_terrain("Test", 100.0, 100.0, TerrainType.FAIRWAY)
        turf = CompressibleTurfModel(terrain)

        # Get stiffness at different compressions
        state1 = turf.get_compression_state(50.0, 50.0, z=0.015, radius=0.02)
        state2 = turf.get_compression_state(50.0, 50.0, z=0.005, radius=0.02)
        state3 = turf.get_compression_state(50.0, 50.0, z=-0.005, radius=0.02)

        # More compression = higher stiffness
        if state1["compression_ratio"] < state2["compression_ratio"]:
            assert state2["effective_stiffness"] >= state1["effective_stiffness"]
        if state2["compression_ratio"] < state3["compression_ratio"]:
            assert state3["effective_stiffness"] >= state2["effective_stiffness"]

    def test_bunker_absorbs_more_energy(self) -> None:
        """Bunker (sand) should absorb more energy than green."""
        bunker_terrain = create_flat_terrain("Bunker", 100.0, 100.0, TerrainType.BUNKER)
        green_terrain = create_flat_terrain("Green", 100.0, 100.0, TerrainType.GREEN)

        bunker_turf = CompressibleTurfModel(bunker_terrain)
        green_turf = CompressibleTurfModel(green_terrain)

        impact_velocity = np.array([10.0, 0.0, -30.0])

        bunker_energy = bunker_turf.compute_energy_absorption(
            50.0, 50.0, impact_velocity
        )
        green_energy = green_turf.compute_energy_absorption(50.0, 50.0, impact_velocity)

        assert (
            bunker_energy["energy_absorption_ratio"]
            > green_energy["energy_absorption_ratio"]
        )

    def test_lie_quality_varies_by_terrain(self) -> None:
        """Lie quality should vary by terrain type."""
        fairway_terrain = create_flat_terrain(
            "Fairway", 100.0, 100.0, TerrainType.FAIRWAY
        )
        rough_terrain = create_flat_terrain("Rough", 100.0, 100.0, TerrainType.ROUGH)

        fairway_turf = CompressibleTurfModel(fairway_terrain)
        rough_turf = CompressibleTurfModel(rough_terrain)

        fairway_lie = fairway_turf.compute_lie_quality(50.0, 50.0)
        rough_lie = rough_turf.compute_lie_quality(50.0, 50.0)

        # Both should have valid playability factors
        assert 0 < fairway_lie["playability_factor"] <= 1.0
        assert 0 < rough_lie["playability_factor"] <= 1.0

        # Rough has taller grass
        assert rough_lie["grass_height"] > fairway_lie["grass_height"]


class TestTerrainAwareEngineWrapper:
    """Test the terrain-aware engine wrapper."""

    def test_terrain_engine_initialization(self) -> None:
        """TerrainAwareEngine should initialize correctly."""
        terrain = create_flat_terrain("Test", 100.0, 100.0)
        engine = TerrainAwareEngine(terrain)

        assert engine.terrain is not None
        assert engine.terrain.name == "Test"

    def test_terrain_engine_height_query(self) -> None:
        """Engine should provide terrain height queries."""
        terrain = create_sloped_terrain(
            "Sloped",
            100.0,
            100.0,
            slope_angle_deg=5.0,
            slope_direction_deg=0.0,
        )
        engine = TerrainAwareEngine(terrain)

        h1 = engine.get_ground_height(0.0, 50.0)
        h2 = engine.get_ground_height(100.0, 50.0)

        # Height should increase with X
        assert h2 > h1

    def test_terrain_engine_properties_query(self) -> None:
        """Engine should provide comprehensive terrain properties."""
        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [
            TerrainPatch(TerrainType.FAIRWAY, 0.0, 50.0, 0.0, 100.0),
            TerrainPatch(TerrainType.GREEN, 50.0, 100.0, 0.0, 100.0),
        ]
        terrain = Terrain(name="Mixed", elevation=elevation, patches=patches)
        engine = TerrainAwareEngine(terrain)

        props = engine.get_terrain_properties(75.0, 50.0)

        assert "elevation" in props
        assert "normal" in props
        assert "terrain_type" in props
        assert "friction" in props
        assert "restitution" in props
        assert props["terrain_type"] == TerrainType.GREEN


class TestTerrainEdgeCases:
    """Test terrain edge cases and boundary conditions."""

    def test_zero_slope_has_vertical_normal(self) -> None:
        """Flat terrain should have vertical normal."""
        terrain = create_flat_terrain("Flat", 100.0, 100.0)
        normal = terrain.elevation.get_normal(50.0, 50.0)

        assert abs(normal[0]) < 1e-6
        assert abs(normal[1]) < 1e-6
        assert abs(normal[2] - 1.0) < 1e-6

    def test_steep_slope_stability(self) -> None:
        """Steep slopes should still produce valid results."""
        terrain = create_sloped_terrain(
            "Steep",
            100.0,
            100.0,
            slope_angle_deg=45.0,
            slope_direction_deg=0.0,
        )

        normal = terrain.elevation.get_normal(50.0, 50.0)

        # Should still be unit vector
        assert abs(np.linalg.norm(normal) - 1.0) < 1e-6

        # Should point mostly up
        assert normal[2] > 0.5

    def test_terrain_boundary_queries(self) -> None:
        """Queries at terrain boundaries should work."""
        terrain = create_flat_terrain("Test", 100.0, 100.0)

        # Queries at edges
        terrain.elevation.get_elevation(0.0, 0.0)
        terrain.elevation.get_elevation(100.0, 0.0)
        terrain.elevation.get_elevation(0.0, 100.0)
        terrain.elevation.get_elevation(100.0, 100.0)

        # Should not raise exceptions

    def test_very_small_penetration(self) -> None:
        """Very small penetrations should produce proportional forces."""
        terrain = create_flat_terrain("Test", 100.0, 100.0)
        contact = TerrainContactModel(terrain)

        # Very small penetration
        force_small = contact.compute_contact_force(50.0, 50.0, z=-0.0001, radius=0.02)

        # Larger penetration
        force_large = contact.compute_contact_force(50.0, 50.0, z=-0.001, radius=0.02)

        # Larger penetration = larger force
        assert np.linalg.norm(force_large) > np.linalg.norm(force_small)
