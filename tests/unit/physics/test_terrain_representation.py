"""Value-asserting tests for terrain geometry + physics (issue #6994).

Covers ``terrain_representation.py`` (ElevationMap flat/sloped/from_array,
gradient/normal/slope-angle vs analytic, ``_check_bounds`` OOB, TerrainRegion
point-in-polygon edges, to_dict/from_dict round-trips, material/contact-param
precedence) and ``_terrain_physics.py`` (penetration, monotonic contact force,
capped Coulomb friction, lie quality).

All assertions are against closed-form analytic expectations; nothing here
requires Qt/GL.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from src.shared.python.physics import terrain_representation as tr
from src.shared.python.physics._terrain_physics import (
    CompressibleTurfModel,
    TerrainContactModel,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# ElevationMap constructors
# --------------------------------------------------------------------------- #


class TestElevationMapFlat:
    def test_flat_constant_elevation(self) -> None:
        em = tr.ElevationMap.flat(10.0, 10.0, 1.0, base_elevation=2.5)
        assert em.get_elevation(0.0, 0.0) == pytest.approx(2.5)
        assert em.get_elevation(5.0, 7.0) == pytest.approx(2.5)

    def test_flat_zero_gradient_and_vertical_normal(self) -> None:
        em = tr.ElevationMap.flat(10.0, 10.0, 1.0)
        assert em.get_gradient(3.0, 3.0) == (pytest.approx(0.0), pytest.approx(0.0))
        assert np.allclose(em.get_normal(3.0, 3.0), [0.0, 0.0, 1.0])
        assert em.get_slope_angle(3.0, 3.0) == pytest.approx(0.0)

    def test_flat_rejects_nonpositive_dims(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            tr.ElevationMap.flat(0.0, 10.0, 1.0)
        with pytest.raises(ValueError, match="Resolution"):
            tr.ElevationMap.flat(10.0, 10.0, 0.0)


class TestElevationMapSloped:
    def test_sloped_gradient_matches_tan(self) -> None:
        # 10 deg uphill toward +X => dz/dx = tan(10 deg), dz/dy = 0.
        em = tr.ElevationMap.sloped(
            10.0, 10.0, 1.0, slope_angle_deg=10.0, slope_direction_deg=0.0
        )
        dzdx, dzdy = em.get_gradient(5.0, 5.0)
        assert dzdx == pytest.approx(math.tan(math.radians(10.0)), rel=1e-6)
        assert dzdy == pytest.approx(0.0, abs=1e-9)

    def test_sloped_slope_angle_recovered(self) -> None:
        em = tr.ElevationMap.sloped(
            10.0, 10.0, 1.0, slope_angle_deg=15.0, slope_direction_deg=90.0
        )
        assert em.get_slope_angle(5.0, 5.0) == pytest.approx(15.0, rel=1e-6)

    def test_sloped_normal_is_unit_vector(self) -> None:
        em = tr.ElevationMap.sloped(
            10.0, 10.0, 1.0, slope_angle_deg=20.0, slope_direction_deg=45.0
        )
        normal = em.get_normal(5.0, 5.0)
        assert np.linalg.norm(normal) == pytest.approx(1.0)
        assert normal[2] > 0.0  # points upward

    def test_sloped_direction_y(self) -> None:
        # Uphill toward +Y => gradient only in y.
        em = tr.ElevationMap.sloped(
            10.0, 10.0, 1.0, slope_angle_deg=10.0, slope_direction_deg=90.0
        )
        dzdx, dzdy = em.get_gradient(5.0, 5.0)
        assert dzdx == pytest.approx(0.0, abs=1e-9)
        assert dzdy == pytest.approx(math.tan(math.radians(10.0)), rel=1e-6)


class TestElevationMapFromArray:
    def test_from_array_bilinear_interpolation(self) -> None:
        # 2x2 grid, resolution 1.0. Plane z = x (cols are X).
        data = np.array([[0.0, 1.0], [0.0, 1.0]])
        em = tr.ElevationMap.from_array(data, resolution=1.0)
        assert em.get_elevation(0.0, 0.0) == pytest.approx(0.0)
        assert em.get_elevation(1.0, 0.0) == pytest.approx(1.0)
        assert em.get_elevation(0.5, 0.0) == pytest.approx(0.5)

    def test_from_array_dimensions(self) -> None:
        data = np.zeros((3, 4))
        em = tr.ElevationMap.from_array(data, resolution=2.0)
        assert em.width == pytest.approx(8.0)  # 4 cols * 2
        assert em.length == pytest.approx(6.0)  # 3 rows * 2


# --------------------------------------------------------------------------- #
# _check_bounds OOB
# --------------------------------------------------------------------------- #


class TestCheckBounds:
    def test_out_of_bounds_x_raises(self) -> None:
        em = tr.ElevationMap.flat(10.0, 10.0, 1.0)
        with pytest.raises(ValueError, match="X coordinate"):
            em.get_elevation(100.0, 5.0)

    def test_out_of_bounds_y_raises(self) -> None:
        em = tr.ElevationMap.flat(10.0, 10.0, 1.0)
        with pytest.raises(ValueError, match="Y coordinate"):
            em.get_elevation(5.0, -1.0)

    def test_coordinate_beyond_last_node_rejected(self) -> None:
        # 10x10 at res 1.0 => 10 cols => last node at x=9.0; x=10 is OOB.
        em = tr.ElevationMap.flat(10.0, 10.0, 1.0)
        with pytest.raises(ValueError):
            em.get_elevation(10.0, 5.0)
        # Last valid node is fine.
        assert em.get_elevation(9.0, 9.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# TerrainRegion point-in-polygon
# --------------------------------------------------------------------------- #


class TestPointInPolygon:
    SQUARE = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]

    def test_interior_point_inside(self) -> None:
        region = tr.TerrainRegion.polygon(tr.TerrainType.GREEN, self.SQUARE)
        assert region.contains(2.0, 2.0) is True

    def test_exterior_point_outside(self) -> None:
        region = tr.TerrainRegion.polygon(tr.TerrainType.GREEN, self.SQUARE)
        assert region.contains(5.0, 2.0) is False
        assert region.contains(-1.0, 2.0) is False

    def test_triangle_membership(self) -> None:
        tri = tr.TerrainRegion.polygon(
            tr.TerrainType.BUNKER, [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]
        )
        assert tri.contains(0.5, 0.5) is True  # near right-angle corner
        assert tri.contains(3.0, 3.0) is False  # beyond hypotenuse

    def test_circle_region_membership(self) -> None:
        circ = tr.TerrainRegion.circle(tr.TerrainType.GREEN, 5.0, 5.0, 2.0)
        assert circ.contains(5.0, 5.0) is True
        assert circ.contains(6.0, 5.0) is True  # distance 1 < radius 2
        assert circ.contains(8.0, 5.0) is False  # distance 3 > radius 2
        # On-boundary point (distance exactly radius) counts as inside.
        assert circ.contains(7.0, 5.0) is True


# --------------------------------------------------------------------------- #
# to_dict / from_dict round-trips
# --------------------------------------------------------------------------- #


class TestRoundTrips:
    def test_elevation_map_round_trip(self) -> None:
        em = tr.ElevationMap.sloped(
            6.0, 6.0, 1.0, slope_angle_deg=5.0, slope_direction_deg=30.0
        )
        restored = tr.ElevationMap.from_dict(em.to_dict())
        assert restored.resolution == em.resolution
        assert restored.width == em.width
        assert np.allclose(restored.data, em.data)
        assert restored.get_elevation(3.0, 3.0) == pytest.approx(
            em.get_elevation(3.0, 3.0)
        )

    def test_region_round_trip_preserves_material(self) -> None:
        mat = tr.SurfaceMaterial(
            name="custom",
            friction_coefficient=0.9,
            compressibility=0.5,
            moisture_content=0.6,
        )
        region = tr.TerrainRegion.circle(
            tr.TerrainType.BUNKER, 1.0, 2.0, 3.0, material=mat
        )
        restored = tr.TerrainRegion.from_dict(region.to_dict())
        assert restored.terrain_type == tr.TerrainType.BUNKER
        assert restored.shape_type == "circle"
        assert restored.material is not None
        assert restored.material.friction_coefficient == pytest.approx(0.9)
        assert restored.material.compressibility == pytest.approx(0.5)
        assert restored.material.moisture_content == pytest.approx(0.6)

    def test_patch_round_trip(self) -> None:
        patch = tr.TerrainPatch(tr.TerrainType.FAIRWAY, 0.0, 5.0, 0.0, 5.0)
        restored = tr.TerrainPatch.from_dict(patch.to_dict())
        assert restored.terrain_type == tr.TerrainType.FAIRWAY
        assert restored.x_max == pytest.approx(5.0)

    def test_terrain_config_round_trip(self) -> None:
        terrain = tr.create_sloped_terrain(
            "course", 8.0, 8.0, slope_angle_deg=3.0, slope_direction_deg=0.0
        )
        cfg = tr.TerrainConfig.from_terrain(terrain)
        rebuilt = tr.TerrainConfig.from_dict(cfg.to_dict()).to_terrain()
        assert rebuilt.name == "course"
        assert rebuilt.get_elevation(4.0, 4.0) == pytest.approx(
            terrain.get_elevation(4.0, 4.0)
        )


# --------------------------------------------------------------------------- #
# Material & contact-param precedence
# --------------------------------------------------------------------------- #


class TestMaterialPrecedence:
    def _terrain(self) -> tr.Terrain:
        em = tr.ElevationMap.flat(10.0, 10.0, 1.0)
        # Patch covers everything as FAIRWAY; region overrides a circle as GREEN.
        patch = tr.TerrainPatch(tr.TerrainType.FAIRWAY, 0.0, 10.0, 0.0, 10.0)
        region = tr.TerrainRegion.circle(tr.TerrainType.GREEN, 5.0, 5.0, 2.0)
        return tr.Terrain(name="t", elevation=em, patches=[patch], regions=[region])

    def test_region_overrides_patch(self) -> None:
        terrain = self._terrain()
        # Inside the green circle => green material.
        assert terrain.get_material(5.0, 5.0).name == "green"
        # Outside the circle but inside the patch => fairway.
        assert terrain.get_material(0.5, 0.5).name == "fairway"

    def test_terrain_type_resolution(self) -> None:
        terrain = self._terrain()
        assert terrain.get_terrain_type(5.0, 5.0) == tr.TerrainType.GREEN
        assert terrain.get_terrain_type(0.5, 0.5) == tr.TerrainType.FAIRWAY

    def test_contact_params_reflect_material(self) -> None:
        terrain = self._terrain()
        params = terrain.get_contact_params(5.0, 5.0)
        green = tr.MATERIALS["green"]
        assert params["friction"] == pytest.approx(green.friction_coefficient)
        assert params["restitution"] == pytest.approx(green.restitution)
        # Stiffness scales with hardness; harder green > softer default rough.
        assert params["stiffness"] == pytest.approx(1e5 * green.hardness)
        assert params["damping"] > 0.0

    def test_default_material_when_uncovered(self) -> None:
        em = tr.ElevationMap.flat(10.0, 10.0, 1.0)
        terrain = tr.Terrain(
            name="bare", elevation=em, default_type=tr.TerrainType.ROUGH
        )
        assert terrain.get_material(1.0, 1.0).name == "rough"


class TestSurfaceMaterialValidation:
    def test_negative_friction_rejected(self) -> None:
        with pytest.raises(ValueError, match="friction_coefficient"):
            tr.SurfaceMaterial(name="bad", friction_coefficient=-0.1)

    def test_restitution_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="restitution"):
            tr.SurfaceMaterial(name="bad", restitution=1.5)


# --------------------------------------------------------------------------- #
# compute_gravity_on_slope
# --------------------------------------------------------------------------- #


class TestGravityOnSlope:
    def test_flat_slope_all_perpendicular(self) -> None:
        g_par, g_perp = tr.compute_gravity_on_slope(0.0, gravity=9.81)
        assert g_par == pytest.approx(0.0, abs=1e-9)
        assert g_perp == pytest.approx(9.81)

    def test_thirty_degree_components(self) -> None:
        g_par, g_perp = tr.compute_gravity_on_slope(30.0, gravity=10.0)
        assert g_par == pytest.approx(10.0 * 0.5)  # sin 30 = 0.5
        assert g_perp == pytest.approx(10.0 * math.cos(math.radians(30.0)))


# =========================================================================== #
# _terrain_physics.py
# =========================================================================== #


def _green_terrain() -> tr.Terrain:
    return tr.create_flat_terrain(
        "green", 10.0, 10.0, terrain_type=tr.TerrainType.GREEN, resolution=1.0
    )


class TestContactModelPenetration:
    def test_penetration_zero_above_ground(self) -> None:
        model = TerrainContactModel(terrain=_green_terrain())
        assert model.compute_penetration(5.0, 5.0, z=1.0, radius=0.0) == 0.0

    def test_penetration_positive_below_ground(self) -> None:
        model = TerrainContactModel(terrain=_green_terrain())
        # ground at z=0, object centre at -0.3 => penetration 0.3.
        assert model.compute_penetration(5.0, 5.0, z=-0.3) == pytest.approx(0.3)

    def test_penetration_accounts_for_radius(self) -> None:
        model = TerrainContactModel(terrain=_green_terrain())
        # contact point = z - radius = 0.1 - 0.3 = -0.2 => penetration 0.2.
        assert model.compute_penetration(5.0, 5.0, z=0.1, radius=0.3) == pytest.approx(
            0.2
        )

    def test_is_in_contact(self) -> None:
        model = TerrainContactModel(terrain=_green_terrain())
        assert model.is_in_contact(5.0, 5.0, z=-0.1) is True
        assert model.is_in_contact(5.0, 5.0, z=0.5) is False


class TestContactForce:
    def test_no_force_without_penetration(self) -> None:
        model = TerrainContactModel(terrain=_green_terrain())
        force = model.compute_contact_force(5.0, 5.0, z=1.0)
        assert np.allclose(force, np.zeros(3))

    def test_force_monotonic_with_penetration(self) -> None:
        model = TerrainContactModel(terrain=_green_terrain())
        shallow = np.linalg.norm(model.compute_contact_force(5.0, 5.0, z=-0.1))
        deep = np.linalg.norm(model.compute_contact_force(5.0, 5.0, z=-0.3))
        assert deep > shallow > 0.0

    def test_force_acts_along_normal_on_flat(self) -> None:
        model = TerrainContactModel(terrain=_green_terrain())
        force = model.compute_contact_force(5.0, 5.0, z=-0.2)
        # On flat ground the normal is +Z, so force is purely vertical.
        assert force[0] == pytest.approx(0.0, abs=1e-6)
        assert force[1] == pytest.approx(0.0, abs=1e-6)
        assert force[2] > 0.0


class TestFrictionForce:
    def test_friction_capped_at_mu_times_normal(self) -> None:
        terrain = _green_terrain()
        model = TerrainContactModel(terrain=terrain)
        velocity = np.array([2.0, 0.0, 0.0])
        normal_force = np.array([0.0, 0.0, 100.0])
        friction = model.compute_friction_force(
            5.0,
            5.0,
            z=-0.1,
            radius=0.0,
            velocity=velocity,
            normal_force=normal_force,
        )
        mu = terrain.get_material(5.0, 5.0).friction_coefficient
        assert np.linalg.norm(friction) == pytest.approx(mu * 100.0)

    def test_friction_opposes_motion(self) -> None:
        model = TerrainContactModel(terrain=_green_terrain())
        velocity = np.array([3.0, 0.0, 0.0])
        friction = model.compute_friction_force(
            5.0,
            5.0,
            z=-0.1,
            radius=0.0,
            velocity=velocity,
            normal_force=np.array([0.0, 0.0, 50.0]),
        )
        assert friction[0] < 0.0  # opposes +X motion

    def test_zero_friction_when_stationary(self) -> None:
        model = TerrainContactModel(terrain=_green_terrain())
        friction = model.compute_friction_force(
            5.0,
            5.0,
            z=-0.1,
            radius=0.0,
            velocity=np.zeros(3),
            normal_force=np.array([0.0, 0.0, 50.0]),
        )
        assert np.allclose(friction, np.zeros(3))

    def test_zero_friction_without_normal_force(self) -> None:
        model = TerrainContactModel(terrain=_green_terrain())
        # No penetration => derived normal force ~0 => no friction.
        friction = model.compute_friction_force(
            5.0, 5.0, z=1.0, radius=0.0, velocity=np.array([2.0, 0.0, 0.0])
        )
        assert np.allclose(friction, np.zeros(3))


class TestCompressibleTurf:
    def test_lie_quality_keys_and_ranges(self) -> None:
        model = CompressibleTurfModel(terrain=_green_terrain())
        lie = model.compute_lie_quality(5.0, 5.0)
        assert set(lie) >= {
            "lie_type",
            "sitting_depth",
            "grass_interference",
            "playability_factor",
            "grass_height",
            "terrain_type",
        }
        assert 0.0 <= lie["playability_factor"] <= 1.0
        assert lie["sitting_depth"] >= 0.0

    def test_firm_green_gives_tight_lie(self) -> None:
        model = CompressibleTurfModel(terrain=_green_terrain())
        lie = model.compute_lie_quality(5.0, 5.0)
        # A firm green barely compresses => shallow sitting depth => tight lie.
        assert lie["lie_type"] == "tight"
        # Near-perfect playability: ball sits almost entirely above the grass.
        assert lie["playability_factor"] > 0.99

    def test_soft_turf_sits_deeper_than_green(self) -> None:
        green = CompressibleTurfModel(terrain=_green_terrain())
        soft_terrain = tr.Terrain(
            name="soft",
            elevation=tr.ElevationMap.flat(10.0, 10.0, 1.0),
            patches=[
                tr.TerrainPatch(
                    tr.TerrainType.ROUGH,
                    0.0,
                    10.0,
                    0.0,
                    10.0,
                    material=tr.MATERIALS["soft_turf"],
                )
            ],
        )
        soft = CompressibleTurfModel(terrain=soft_terrain)
        deep = soft.compute_lie_quality(5.0, 5.0)["sitting_depth"]
        shallow = green.compute_lie_quality(5.0, 5.0)["sitting_depth"]
        assert deep >= shallow

    def test_compression_state_clamped_to_max(self) -> None:
        model = CompressibleTurfModel(terrain=_green_terrain())
        state = model.get_compression_state(5.0, 5.0, z=-1.0, radius=0.0)
        # Huge penetration is clamped to the material's max compression depth.
        assert state["compression_depth"] <= state["max_compression"] + 1e-9
        assert 0.0 <= state["compression_ratio"] <= 1.0
