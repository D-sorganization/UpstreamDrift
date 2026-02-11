"""Unit tests for terrain-aware physics engine interface (TDD: Tests First).

Tests define the expected behavior for terrain integration with physics engines.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.physics.terrain import (
    ElevationMap,
    Terrain,
    TerrainPatch,
    TerrainType,
    create_flat_terrain,
    create_sloped_terrain,
)


class TestTerrainAwareEngineInterface:
    """Test the terrain-aware physics engine interface."""

    def test_engine_accepts_terrain_config(self) -> None:
        """Engine should accept terrain configuration."""
        from src.shared.python.physics.terrain_engine import TerrainAwareEngine

        terrain = create_flat_terrain("Test", 100.0, 100.0, TerrainType.FAIRWAY)
        engine = TerrainAwareEngine()
        engine.set_terrain(terrain)

        assert engine.terrain is not None
        assert engine.terrain.name == "Test"

    def test_engine_terrain_ground_height(self) -> None:
        """Engine should query terrain for ground height."""
        from src.shared.python.physics.terrain_engine import TerrainAwareEngine

        terrain = create_sloped_terrain(
            "Sloped",
            100.0,
            100.0,
            slope_angle_deg=5.0,
            slope_direction_deg=0.0,
        )
        engine = TerrainAwareEngine()
        engine.set_terrain(terrain)

        # Ground height should vary based on position
        h1 = engine.get_ground_height(0.0, 50.0)
        h2 = engine.get_ground_height(50.0, 50.0)
        h3 = engine.get_ground_height(100.0, 50.0)

        # Height should increase with X (slope direction 0 = +X)
        assert h2 > h1
        assert h3 > h2

    def test_engine_terrain_contact_normal(self) -> None:
        """Engine should provide terrain contact normal."""
        from src.shared.python.physics.terrain_engine import TerrainAwareEngine

        terrain = create_sloped_terrain(
            "Sloped",
            100.0,
            100.0,
            slope_angle_deg=15.0,
            slope_direction_deg=0.0,
        )
        engine = TerrainAwareEngine()
        engine.set_terrain(terrain)

        normal = engine.get_contact_normal(50.0, 50.0)

        # Should be unit vector
        assert abs(np.linalg.norm(normal) - 1.0) < 1e-6

        # Should point mostly up
        assert normal[2] > 0.9

        # Should tilt opposite to slope direction
        assert normal[0] < 0

    def test_engine_terrain_friction(self) -> None:
        """Engine should provide terrain-specific friction."""
        from src.shared.python.physics.terrain_engine import TerrainAwareEngine

        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [
            TerrainPatch(TerrainType.FAIRWAY, 0.0, 50.0, 0.0, 100.0),
            TerrainPatch(TerrainType.BUNKER, 50.0, 100.0, 0.0, 100.0),
        ]
        terrain = Terrain(name="Mixed", elevation=elevation, patches=patches)

        engine = TerrainAwareEngine()
        engine.set_terrain(terrain)

        # Bunker should have higher friction
        fairway_friction = engine.get_friction(25.0, 50.0)
        bunker_friction = engine.get_friction(75.0, 50.0)

        assert bunker_friction > fairway_friction

    def test_engine_terrain_restitution(self) -> None:
        """Engine should provide terrain-specific restitution."""
        from src.shared.python.physics.terrain_engine import TerrainAwareEngine

        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [
            TerrainPatch(TerrainType.GREEN, 0.0, 50.0, 0.0, 100.0),
            TerrainPatch(TerrainType.BUNKER, 50.0, 100.0, 0.0, 100.0),
        ]
        terrain = Terrain(name="Mixed", elevation=elevation, patches=patches)

        engine = TerrainAwareEngine()
        engine.set_terrain(terrain)

        # Green should have higher restitution (bouncier)
        green_restitution = engine.get_restitution(25.0, 50.0)
        bunker_restitution = engine.get_restitution(75.0, 50.0)

        assert green_restitution > bunker_restitution


class TestTerrainContactModel:
    """Test terrain contact physics."""

    def test_ground_contact_detection(self) -> None:
        """Detect when object contacts terrain."""
        from src.shared.python.physics.terrain_engine import TerrainContactModel

        terrain = create_flat_terrain("Flat", 100.0, 100.0)
        contact = TerrainContactModel(terrain)

        # Object above ground - no contact
        assert not contact.is_in_contact(50.0, 50.0, z=1.0, radius=0.02)

        # Object at ground level - contact
        assert contact.is_in_contact(50.0, 50.0, z=0.02, radius=0.02)

        # Object below ground - contact (penetration)
        assert contact.is_in_contact(50.0, 50.0, z=-0.01, radius=0.02)

    def test_contact_force_normal(self) -> None:
        """Contact force should be normal to terrain surface."""
        from src.shared.python.physics.terrain_engine import TerrainContactModel

        terrain = create_sloped_terrain(
            "Sloped",
            100.0,
            100.0,
            slope_angle_deg=10.0,
            slope_direction_deg=0.0,
        )
        contact = TerrainContactModel(terrain)

        # Get contact force for object penetrating terrain
        force = contact.compute_contact_force(50.0, 50.0, z=0.0, radius=0.02)

        # Force should have positive Z component (pushing up)
        assert force[2] > 0

        # Force should have positive X component (slope pushes object in -X)
        # Actually on a slope in +X direction, normal points in -X, so force points in -X
        # Wait, the normal points away from the surface, so if slope goes up in +X,
        # normal tilts in -X, and contact force is in direction of normal, so -X
        assert force[0] < 0  # Force tilts opposite to slope

    def test_contact_penetration_force(self) -> None:
        """Contact force should increase with penetration depth."""
        from src.shared.python.physics.terrain_engine import TerrainContactModel

        terrain = create_flat_terrain("Flat", 100.0, 100.0)
        contact = TerrainContactModel(terrain, stiffness=1e5, damping=1e3)

        # Small penetration
        force1 = contact.compute_contact_force(50.0, 50.0, z=-0.001, radius=0.02)

        # Larger penetration
        force2 = contact.compute_contact_force(50.0, 50.0, z=-0.01, radius=0.02)

        # Force should be larger for more penetration
        assert np.linalg.norm(force2) > np.linalg.norm(force1)

    def test_friction_force(self) -> None:
        """Friction force should oppose motion."""
        from src.shared.python.physics.terrain_engine import TerrainContactModel

        terrain = create_flat_terrain("Flat", 100.0, 100.0)
        contact = TerrainContactModel(terrain)

        # Object moving in +X direction
        velocity = np.array([1.0, 0.0, 0.0])

        friction = contact.compute_friction_force(
            50.0, 50.0, z=0.0, radius=0.02, velocity=velocity
        )

        # Friction should oppose motion (be in -X direction)
        assert friction[0] < 0
        assert abs(friction[2]) < abs(friction[0])  # Mostly horizontal


class TestTerrainPhysicsParameters:
    """Test terrain-related physics parameters."""

    def test_terrain_parameters_registered(self) -> None:
        """Terrain parameters should be in registry."""
        from src.shared.python.physics.physics_parameters import get_parameter_registry

        registry = get_parameter_registry()

        # Check terrain-related parameters exist
        assert registry.get("TERRAIN_FRICTION_FAIRWAY") is not None
        assert registry.get("TERRAIN_FRICTION_ROUGH") is not None
        assert registry.get("TERRAIN_FRICTION_GREEN") is not None
        assert registry.get("TERRAIN_FRICTION_BUNKER") is not None

        assert registry.get("TERRAIN_RESTITUTION_FAIRWAY") is not None
        assert registry.get("TERRAIN_RESTITUTION_BUNKER") is not None

    def test_terrain_friction_values(self) -> None:
        """Terrain friction values should be physically reasonable."""
        from src.shared.python.physics.physics_parameters import get_parameter_registry

        registry = get_parameter_registry()

        # Bunker (sand) should have highest friction
        bunker_friction = registry.get("TERRAIN_FRICTION_BUNKER")
        fairway_friction = registry.get("TERRAIN_FRICTION_FAIRWAY")
        green_friction = registry.get("TERRAIN_FRICTION_GREEN")

        assert bunker_friction is not None
        assert bunker_friction.value > fairway_friction.value
        assert green_friction.value < fairway_friction.value


class TestTerrainGeometryGeneration:
    """Test terrain geometry generation for physics engines."""

    def test_generate_heightfield_mesh(self) -> None:
        """Generate mesh from elevation map."""
        from src.shared.python.physics.terrain_engine import TerrainGeometryGenerator

        terrain = create_sloped_terrain(
            "Sloped",
            10.0,
            10.0,
            slope_angle_deg=5.0,
            slope_direction_deg=0.0,
            resolution=1.0,
        )

        generator = TerrainGeometryGenerator(terrain)
        vertices, triangles = generator.generate_mesh()

        # Should have vertices
        assert len(vertices) > 0

        # Should have triangles
        assert len(triangles) > 0

        # Each triangle should reference 3 vertices
        for tri in triangles:
            assert len(tri) == 3
            for idx in tri:
                assert 0 <= idx < len(vertices)

    def test_generate_mujoco_hfield(self) -> None:
        """Generate MuJoCo heightfield data."""
        from src.shared.python.physics.terrain_engine import TerrainGeometryGenerator

        terrain = create_sloped_terrain(
            "Sloped",
            10.0,
            10.0,
            slope_angle_deg=5.0,
            slope_direction_deg=0.0,
            resolution=0.5,
        )

        generator = TerrainGeometryGenerator(terrain)
        hfield_data, hfield_size = generator.generate_mujoco_hfield()

        # Data should be 2D array
        assert hfield_data.ndim == 2

        # Size should match terrain
        assert hfield_size[0] == terrain.elevation.width
        assert hfield_size[1] == terrain.elevation.length

    def test_generate_mujoco_xml_snippet(self) -> None:
        """Generate MuJoCo XML for terrain."""
        from src.shared.python.physics.terrain_engine import TerrainGeometryGenerator

        terrain = create_flat_terrain("Test", 50.0, 100.0)

        generator = TerrainGeometryGenerator(terrain)
        xml = generator.generate_mujoco_xml()

        # Should contain asset and body elements
        assert "<asset>" in xml or "hfield" in xml
        assert "geom" in xml


class TestTerrainEngineIntegration:
    """Test terrain integration with specific physics engines."""

    def test_apply_terrain_to_mock_engine(self) -> None:
        """Apply terrain to a mock physics engine."""
        from src.shared.python.physics.terrain_engine import apply_terrain_to_engine

        # Create a mock engine
        class MockEngine:
            def __init__(self):
                self.ground_height = 0.0
                self.friction = 0.5
                self.terrain_applied = False

            def set_ground_properties(self, height, friction, restitution):
                self.ground_height = height
                self.friction = friction
                self.terrain_applied = True

        engine = MockEngine()
        terrain = create_flat_terrain("Test", 100.0, 100.0)

        apply_terrain_to_engine(engine, terrain, x=50.0, y=50.0)

        assert engine.terrain_applied

    def test_terrain_update_during_simulation(self) -> None:
        """Terrain properties should update as object moves."""
        from src.shared.python.physics.terrain_engine import TerrainAwareEngine

        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [
            TerrainPatch(TerrainType.FAIRWAY, 0.0, 50.0, 0.0, 100.0),
            TerrainPatch(TerrainType.GREEN, 50.0, 100.0, 0.0, 100.0),
        ]
        terrain = Terrain(name="Mixed", elevation=elevation, patches=patches)

        engine = TerrainAwareEngine()
        engine.set_terrain(terrain)

        # Properties should change based on position
        props1 = engine.get_terrain_properties(25.0, 50.0)
        props2 = engine.get_terrain_properties(75.0, 50.0)

        assert props1["terrain_type"] == TerrainType.FAIRWAY
        assert props2["terrain_type"] == TerrainType.GREEN


class TestCompressibleTurf:
    """Test compressible turf/grass model."""

    def test_compressible_material_properties(self) -> None:
        """Materials should have compressibility properties."""
        from src.shared.python.physics.terrain import MATERIALS

        # Check compressibility is defined
        fairway = MATERIALS["fairway"]
        assert hasattr(fairway, "compressibility")
        assert hasattr(fairway, "compression_damping")
        assert hasattr(fairway, "moisture_content")

        # Bunker (sand) should be more compressible than green
        bunker = MATERIALS["bunker"]
        green = MATERIALS["green"]
        assert bunker.compressibility > green.compressibility

    def test_compressible_turf_contact(self) -> None:
        """Test contact force on compressible turf."""
        from src.shared.python.physics.terrain_engine import CompressibleTurfModel

        terrain = create_flat_terrain("Test", 100.0, 100.0, TerrainType.FAIRWAY)
        turf = CompressibleTurfModel(terrain)

        # Ball at ground level
        force = turf.compute_turf_contact_force(50.0, 50.0, z=0.02, radius=0.02)

        # Should have upward force
        assert force[2] >= 0

    def test_compression_state(self) -> None:
        """Test compression state calculation."""
        from src.shared.python.physics.terrain_engine import CompressibleTurfModel

        terrain = create_flat_terrain("Test", 100.0, 100.0, TerrainType.ROUGH)
        turf = CompressibleTurfModel(terrain)

        # Ball penetrating ground
        state = turf.get_compression_state(50.0, 50.0, z=-0.01, radius=0.02)

        assert state["compression_depth"] > 0
        assert state["effective_stiffness"] > 0
        assert 0 <= state["compression_ratio"] <= 1

    def test_lie_quality_calculation(self) -> None:
        """Test ball lie quality on different surfaces."""
        from src.shared.python.physics.terrain_engine import CompressibleTurfModel

        # Fairway should have good lie
        fairway_terrain = create_flat_terrain(
            "Fairway", 100.0, 100.0, TerrainType.FAIRWAY
        )
        fairway_turf = CompressibleTurfModel(fairway_terrain)
        fairway_lie = fairway_turf.compute_lie_quality(50.0, 50.0)

        assert fairway_lie["playability_factor"] > 0.8
        assert fairway_lie["lie_type"] in ["tight", "normal"]

        # Rough should have taller grass
        rough_terrain = create_flat_terrain("Rough", 100.0, 100.0, TerrainType.ROUGH)
        rough_turf = CompressibleTurfModel(rough_terrain)
        rough_lie = rough_turf.compute_lie_quality(50.0, 50.0)

        # Rough has taller grass than fairway
        assert rough_lie["grass_height"] > fairway_lie["grass_height"]
        # Both should produce valid lie types
        assert rough_lie["lie_type"] in ["tight", "normal", "sitting_down", "plugged"]

    def test_energy_absorption(self) -> None:
        """Test energy absorption during impact."""
        from src.shared.python.physics.terrain_engine import CompressibleTurfModel

        terrain = create_flat_terrain("Test", 100.0, 100.0, TerrainType.BUNKER)
        turf = CompressibleTurfModel(terrain)

        # Ball landing with vertical velocity
        impact_velocity = np.array([10.0, 0.0, -20.0])  # m/s
        energy = turf.compute_energy_absorption(50.0, 50.0, impact_velocity)

        assert energy["kinetic_energy"] > 0
        assert energy["absorbed_energy"] > 0
        assert energy["remaining_energy"] < energy["kinetic_energy"]
        assert 0 <= energy["energy_absorption_ratio"] <= 1

        # Bunker should absorb more energy than green
        green_terrain = create_flat_terrain("Green", 100.0, 100.0, TerrainType.GREEN)
        green_turf = CompressibleTurfModel(green_terrain)
        green_energy = green_turf.compute_energy_absorption(50.0, 50.0, impact_velocity)

        assert (
            energy["energy_absorption_ratio"] > green_energy["energy_absorption_ratio"]
        )

    def test_soft_turf_material(self) -> None:
        """Test soft turf material for wet conditions."""
        from src.shared.python.physics.terrain import MATERIALS

        assert "soft_turf" in MATERIALS
        soft = MATERIALS["soft_turf"]

        assert soft.is_compressible
        assert soft.compressibility > 0.3
        assert soft.moisture_content > 0.3

    def test_progressive_stiffening(self) -> None:
        """Test that stiffness increases with compression."""
        from src.shared.python.physics.terrain_engine import CompressibleTurfModel

        terrain = create_flat_terrain("Test", 100.0, 100.0, TerrainType.FAIRWAY)
        turf = CompressibleTurfModel(terrain)

        # Light compression
        state1 = turf.get_compression_state(50.0, 50.0, z=0.01, radius=0.02)

        # Heavy compression
        state2 = turf.get_compression_state(50.0, 50.0, z=-0.01, radius=0.02)

        # Stiffness should increase with compression ratio
        if state1["compression_ratio"] < state2["compression_ratio"]:
            assert state2["effective_stiffness"] >= state1["effective_stiffness"]


class TestTerrainValidation:
    """Test terrain configuration validation."""

    def test_validate_terrain_dimensions(self) -> None:
        """Terrain dimensions should be validated."""
        from src.shared.python.physics.terrain_engine import validate_terrain

        terrain = create_flat_terrain("Test", 100.0, 200.0)
        errors = validate_terrain(terrain)

        assert len(errors) == 0

    def test_validate_terrain_patches_within_bounds(self) -> None:
        """Patches should be within terrain bounds."""
        from src.shared.python.physics.terrain_engine import validate_terrain

        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        # Patch extends beyond terrain
        patches = [TerrainPatch(TerrainType.FAIRWAY, 0.0, 150.0, 0.0, 100.0)]
        terrain = Terrain(name="Test", elevation=elevation, patches=patches)

        errors = validate_terrain(terrain)

        assert len(errors) > 0
        assert any("bounds" in e.lower() for e in errors)

    def test_validate_terrain_resolution(self) -> None:
        """Terrain resolution should be reasonable."""
        from src.shared.python.physics.terrain_engine import validate_terrain

        # Very low resolution terrain
        terrain = create_flat_terrain("Test", 100.0, 100.0, resolution=50.0)
        warnings = validate_terrain(terrain, warn_low_resolution=True)

        # Should warn about low resolution
        assert any("resolution" in w.lower() for w in warnings)
