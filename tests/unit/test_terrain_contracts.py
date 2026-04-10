"""Tests for terrain API correctness (issue #2473).

Covers three semantic bugs:
1. ElevationMap._check_bounds accepts coordinates beyond the actual grid extent.
2. TerrainRegion.to_dict/from_dict drops material physics fields on round-trip.
3. terrain_engine / terrain_mixin return fabricated 0.0 on out-of-bounds queries.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.physics.terrain import (
    ElevationMap,
    SurfaceMaterial,
    TerrainRegion,
    TerrainType,
)


class TestElevationMapBounds:
    """ElevationMap must reject coordinates beyond the actual grid extent."""

    def _make_map(self) -> ElevationMap:
        """10×10m grid with 1m resolution → nodes at 0..9 in each axis."""
        return ElevationMap.flat(width=10.0, length=10.0, resolution=1.0)

    def test_query_at_origin_succeeds(self) -> None:
        """First grid node (0, 0) must be accepted."""
        em = self._make_map()
        em.get_elevation(0.0, 0.0)  # must not raise

    def test_query_at_last_grid_node_succeeds(self) -> None:
        """Last grid node (9, 9) must be accepted."""
        em = self._make_map()
        em.get_elevation(9.0, 9.0)  # must not raise

    def test_query_beyond_grid_extent_raises(self) -> None:
        """Coordinate == width (10.0) is beyond the last node (9.0) — must raise."""
        em = self._make_map()
        with pytest.raises(ValueError, match="out of bounds"):
            em.get_elevation(10.0, 5.0)

    def test_query_beyond_length_raises(self) -> None:
        """Coordinate == length (10.0) is beyond the last node (9.0) — must raise."""
        em = self._make_map()
        with pytest.raises(ValueError, match="out of bounds"):
            em.get_elevation(5.0, 10.0)

    def test_negative_x_raises(self) -> None:
        """Negative X coordinate is always out of bounds."""
        em = self._make_map()
        with pytest.raises(ValueError):
            em.get_elevation(-0.1, 5.0)

    def test_negative_y_raises(self) -> None:
        """Negative Y coordinate is always out of bounds."""
        em = self._make_map()
        with pytest.raises(ValueError):
            em.get_elevation(5.0, -0.1)


class TestTerrainRegionSerializationRoundTrip:
    """TerrainRegion to_dict/from_dict must preserve all SurfaceMaterial fields."""

    def _custom_material(self) -> SurfaceMaterial:
        return SurfaceMaterial(
            name="custom_rough",
            friction_coefficient=0.8,
            rolling_resistance=0.15,
            restitution=0.4,
            hardness=0.6,
            grass_height_m=0.02,
            compressibility=0.1,
            compression_damping=0.5,
            turf_density=500.0,
            moisture_content=0.2,
        )

    def _make_region(self) -> TerrainRegion:
        return TerrainRegion.circle(
            terrain_type=TerrainType.ROUGH,
            center_x=10.0,
            center_y=20.0,
            radius=5.0,
            material=self._custom_material(),
        )

    def test_friction_preserved(self) -> None:
        """friction_coefficient must survive to_dict → from_dict round-trip."""
        region = self._make_region()
        restored = TerrainRegion.from_dict(region.to_dict())
        assert restored.material is not None
        assert restored.material.friction_coefficient == pytest.approx(0.8)

    def test_rolling_resistance_preserved(self) -> None:
        """rolling_resistance must survive round-trip."""
        region = self._make_region()
        restored = TerrainRegion.from_dict(region.to_dict())
        assert restored.material is not None
        assert restored.material.rolling_resistance == pytest.approx(0.15)

    def test_restitution_preserved(self) -> None:
        """restitution must survive round-trip."""
        region = self._make_region()
        restored = TerrainRegion.from_dict(region.to_dict())
        assert restored.material is not None
        assert restored.material.restitution == pytest.approx(0.4)

    def test_hardness_preserved(self) -> None:
        """hardness must survive round-trip."""
        region = self._make_region()
        restored = TerrainRegion.from_dict(region.to_dict())
        assert restored.material is not None
        assert restored.material.hardness == pytest.approx(0.6)

    def test_grass_height_preserved(self) -> None:
        """grass_height_m must survive round-trip."""
        region = self._make_region()
        restored = TerrainRegion.from_dict(region.to_dict())
        assert restored.material is not None
        assert restored.material.grass_height_m == pytest.approx(0.02)

    def test_compressibility_preserved(self) -> None:
        """compressibility must survive round-trip."""
        region = self._make_region()
        restored = TerrainRegion.from_dict(region.to_dict())
        assert restored.material is not None
        assert restored.material.compressibility == pytest.approx(0.1)

    def test_compression_damping_preserved(self) -> None:
        """compression_damping must survive round-trip."""
        region = self._make_region()
        restored = TerrainRegion.from_dict(region.to_dict())
        assert restored.material is not None
        assert restored.material.compression_damping == pytest.approx(0.5)

    def test_turf_density_preserved(self) -> None:
        """turf_density must survive round-trip."""
        region = self._make_region()
        restored = TerrainRegion.from_dict(region.to_dict())
        assert restored.material is not None
        assert restored.material.turf_density == pytest.approx(500.0)

    def test_moisture_content_preserved(self) -> None:
        """moisture_content must survive round-trip."""
        region = self._make_region()
        restored = TerrainRegion.from_dict(region.to_dict())
        assert restored.material is not None
        assert restored.material.moisture_content == pytest.approx(0.2)

    def test_no_material_round_trips_as_none(self) -> None:
        """Region without custom material must round-trip without materialkey."""
        region = TerrainRegion.circle(
            terrain_type=TerrainType.FAIRWAY,
            center_x=0.0,
            center_y=0.0,
            radius=5.0,
        )
        restored = TerrainRegion.from_dict(region.to_dict())
        assert restored.material is None


class TestTerrainEngineOutOfBounds:
    """terrain_engine.get_ground_height must not fabricate 0.0 on out-of-bounds."""

    def test_get_ground_height_raises_on_oob(self) -> None:
        """Out-of-bounds query must raise ValueError, not return 0.0."""
        from src.shared.python.physics.terrain_engine import TerrainAwareEngine

        engine = TerrainAwareEngine()
        elevation = ElevationMap.flat(width=10.0, length=10.0, resolution=1.0)
        from src.shared.python.physics.terrain import Terrain

        terrain = Terrain(name="test", elevation=elevation)
        engine.set_terrain(terrain)

        with pytest.raises(ValueError, match="out of bounds"):
            engine.get_ground_height(100.0, 100.0)

    def test_get_ground_height_in_bounds_still_works(self) -> None:
        """In-bounds query must still return the correct elevation."""
        from src.shared.python.physics.terrain_engine import TerrainAwareEngine

        engine = TerrainAwareEngine()
        elevation = ElevationMap.flat(
            width=10.0, length=10.0, resolution=1.0, base_elevation=3.5
        )
        from src.shared.python.physics.terrain import Terrain

        terrain = Terrain(name="test", elevation=elevation)
        engine.set_terrain(terrain)

        assert engine.get_ground_height(5.0, 5.0) == pytest.approx(3.5)

    def test_get_ground_height_no_terrain_returns_zero(self) -> None:
        """Without a terrain attached, 0.0 is the correct sentinel."""
        from src.shared.python.physics.terrain_engine import TerrainAwareEngine

        engine = TerrainAwareEngine()
        assert engine.get_ground_height(100.0, 100.0) == pytest.approx(0.0)


class TestTerrainMixinOutOfBounds:
    """TerrainMixin.get_ground_height must not fabricate 0.0 on out-of-bounds."""

    def _make_mixin_with_terrain(self) -> Any:
        from src.shared.python.physics.terrain_mixin import TerrainMixin

        class FakeEngine(TerrainMixin):
            pass

        engine = FakeEngine()
        elevation = ElevationMap.flat(width=10.0, length=10.0, resolution=1.0)
        from src.shared.python.physics.terrain import Terrain

        terrain = Terrain(name="test", elevation=elevation)
        engine.set_terrain(terrain)
        return engine

    def test_get_ground_height_raises_on_oob(self) -> None:
        """Out-of-bounds query must raise ValueError, not return 0.0."""
        engine = self._make_mixin_with_terrain()
        with pytest.raises(ValueError, match="out of bounds"):
            engine.get_ground_height(100.0, 100.0)

    def test_get_ground_height_in_bounds_still_works(self) -> None:
        """In-bounds query must still return the correct elevation."""
        engine = self._make_mixin_with_terrain()
        # flat at base_elevation=0.0 by default
        assert engine.get_ground_height(5.0, 5.0) == pytest.approx(0.0)
