"""Invariants of the BunkerShot3D domain value objects (issue #8608, W1).

ADR-0032 decision 1: narrow value objects replace ``BunkerShotConfig``'s flat
delegating accessors, and each validates its own invariants. These tests pin
those invariants, including the properties that must hold for *any* admissible
value (hypothesis), not only for the handful of numbers a fixture happens to
use.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from bunkershot3d.domain import (
    BoundaryCondition,
    ContactMaterial,
    DomainBox,
    GrainPopulation,
    SolverSettings,
    SwingCondition,
    TrajectorySource,
)
from bunkershot3d.exceptions import DomainInvariantError
from bunkershot3d.geometry import DeliveryCondition

pytestmark = pytest.mark.unit

_EXTENT = st.floats(
    min_value=1.0e-4, max_value=1.0e3, allow_nan=False, allow_infinity=False
)
_FRACTION = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

_NON_FINITE = [math.nan, math.inf, -math.inf]


def _box(**overrides: object) -> DomainBox:
    fields: dict[str, object] = {
        "length_x_m": 0.4,
        "width_y_m": 0.3,
        "depth_z_m": 0.1,
    }
    fields.update(overrides)
    return DomainBox(**fields)  # type: ignore[arg-type]


def _grains(**overrides: object) -> GrainPopulation:
    fields: dict[str, object] = {
        "count": 50_000,
        "diameter_mean_m": 4.0e-4,
        "diameter_sigma_log": 0.1,
        "density_kg_m3": 2650.0,
    }
    fields.update(overrides)
    return GrainPopulation(**fields)  # type: ignore[arg-type]


def _material(**overrides: object) -> ContactMaterial:
    fields: dict[str, object] = {
        "friction": 0.5,
        "restitution": 0.3,
        "youngs_modulus_pa": 7.0e10,
        "poisson_ratio": 0.17,
    }
    fields.update(overrides)
    return ContactMaterial(**fields)  # type: ignore[arg-type]


class TestAllValueObjectsAreFrozen:
    @pytest.mark.parametrize(
        "instance",
        [
            _box(),
            _grains(),
            _material(),
            SolverSettings(output_rate_hz=1000.0),
            SwingCondition(clubhead_speed_mps=25.0, duration_s=0.02),
            TrajectorySource(file="swing.csv", duration_s=0.02),
        ],
        ids=lambda obj: type(obj).__name__,
    )
    def test_assignment_is_refused(self, instance: object) -> None:
        field = dataclasses.fields(instance)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field, 1.0)


class TestDomainBox:
    def test_extents_are_reported_in_authoring_order(self) -> None:
        assert _box().extents_m == (0.4, 0.3, 0.1)

    def test_half_extents_are_half_the_extents(self) -> None:
        assert _box().half_extents_m == (0.2, 0.15, 0.05)

    def test_default_boundary_is_fixed(self) -> None:
        assert _box().boundary is BoundaryCondition.FIXED

    def test_boundary_accepts_its_string_spelling(self) -> None:
        assert _box(boundary="periodic").boundary is BoundaryCondition.PERIODIC

    def test_unknown_boundary_is_refused(self) -> None:
        with pytest.raises(DomainInvariantError, match="boundary"):
            _box(boundary="absorbing")

    @pytest.mark.parametrize("axis", ["length_x_m", "width_y_m", "depth_z_m"])
    def test_non_positive_extent_is_refused(self, axis: str) -> None:
        with pytest.raises(DomainInvariantError, match=axis):
            _box(**{axis: 0.0})

    @pytest.mark.parametrize("axis", ["length_x_m", "width_y_m", "depth_z_m"])
    @pytest.mark.parametrize("bad", _NON_FINITE)
    def test_non_finite_extent_is_refused(self, axis: str, bad: float) -> None:
        with pytest.raises(DomainInvariantError, match=axis):
            _box(**{axis: bad})

    @given(_EXTENT, _EXTENT, _EXTENT)
    def test_volume_is_the_product_of_the_extents(
        self, lx: float, ly: float, lz: float
    ) -> None:
        box = DomainBox(length_x_m=lx, width_y_m=ly, depth_z_m=lz)
        assert box.volume_m3 == pytest.approx(lx * ly * lz, rel=1e-12)

    @given(_EXTENT, _EXTENT, _EXTENT)
    def test_volume_is_positive_for_every_admissible_box(
        self, lx: float, ly: float, lz: float
    ) -> None:
        assert DomainBox(length_x_m=lx, width_y_m=ly, depth_z_m=lz).volume_m3 > 0.0

    def test_it_is_the_numerical_box_not_the_sand_patch(self) -> None:
        """The docstring must point at ``sand.BunkerBedGeometry`` so the two
        rectangular-extent objects do not drift into duplicates."""
        assert "BunkerBedGeometry" in (DomainBox.__doc__ or "")


class TestGrainPopulation:
    def test_mean_radius_is_half_the_mean_diameter(self) -> None:
        assert _grains().radius_mean_m == pytest.approx(2.0e-4)

    def test_effective_count_is_the_count_when_not_coarse_grained(self) -> None:
        assert _grains().effective_count == 50_000

    def test_coarse_graining_reduces_the_effective_count(self) -> None:
        assert _grains(coarse_graining_factor=10.0).effective_count == 5_000

    def test_effective_count_never_falls_below_one(self) -> None:
        assert _grains(count=2, coarse_graining_factor=1000.0).effective_count == 1

    def test_zero_count_is_refused(self) -> None:
        with pytest.raises(DomainInvariantError, match="count"):
            _grains(count=0)

    def test_negative_sigma_is_refused(self) -> None:
        with pytest.raises(DomainInvariantError, match="sigma"):
            _grains(diameter_sigma_log=-0.1)

    def test_a_monodisperse_population_is_allowed(self) -> None:
        assert _grains(diameter_sigma_log=0.0).diameter_sigma_log == 0.0

    def test_coarse_graining_below_one_is_refused(self) -> None:
        with pytest.raises(DomainInvariantError, match="coarse_graining_factor"):
            _grains(coarse_graining_factor=0.5)

    @pytest.mark.parametrize("field", ["diameter_mean_m", "density_kg_m3"])
    def test_non_positive_physical_quantity_is_refused(self, field: str) -> None:
        with pytest.raises(DomainInvariantError, match=field):
            _grains(**{field: 0.0})

    @given(
        st.floats(min_value=1e-6, max_value=1e-2, allow_nan=False),
        st.floats(min_value=500.0, max_value=20_000.0, allow_nan=False),
    )
    def test_mean_grain_mass_matches_a_sphere_of_the_mean_diameter(
        self, diameter: float, density: float
    ) -> None:
        grains = GrainPopulation(
            count=10,
            diameter_mean_m=diameter,
            diameter_sigma_log=0.0,
            density_kg_m3=density,
        )
        expected = density * (math.pi / 6.0) * diameter**3
        assert grains.mean_grain_mass_kg == pytest.approx(expected, rel=1e-12)

    @given(st.integers(min_value=1, max_value=10**7), st.floats(1.0, 1e4))
    def test_effective_count_never_exceeds_the_configured_count(
        self, count: int, factor: float
    ) -> None:
        assume(math.isfinite(factor))
        grains = GrainPopulation(
            count=count,
            diameter_mean_m=1.0e-3,
            diameter_sigma_log=0.0,
            density_kg_m3=2650.0,
            coarse_graining_factor=factor,
        )
        assert 1 <= grains.effective_count <= count


class TestContactMaterial:
    @pytest.mark.parametrize("value", [-0.01, 1.01])
    def test_friction_outside_the_unit_interval_is_refused(self, value: float) -> None:
        with pytest.raises(DomainInvariantError, match="friction"):
            _material(friction=value)

    @pytest.mark.parametrize("value", [-0.01, 1.01])
    def test_restitution_outside_the_unit_interval_is_refused(
        self, value: float
    ) -> None:
        with pytest.raises(DomainInvariantError, match="restitution"):
            _material(restitution=value)

    @pytest.mark.parametrize("value", [0.0, 0.5, 0.6])
    def test_poisson_ratio_outside_the_open_interval_is_refused(
        self, value: float
    ) -> None:
        with pytest.raises(DomainInvariantError, match="poisson"):
            _material(poisson_ratio=value)

    def test_non_positive_modulus_is_refused(self) -> None:
        with pytest.raises(DomainInvariantError, match="youngs_modulus_pa"):
            _material(youngs_modulus_pa=0.0)

    @given(
        st.floats(min_value=1.0e3, max_value=1.0e12, allow_nan=False),
        st.floats(min_value=0.01, max_value=0.49, allow_nan=False),
    )
    def test_shear_modulus_matches_the_isotropic_relation(
        self, modulus: float, poisson: float
    ) -> None:
        material = ContactMaterial(
            friction=0.5,
            restitution=0.3,
            youngs_modulus_pa=modulus,
            poisson_ratio=poisson,
        )
        assert material.shear_modulus_pa == pytest.approx(
            modulus / (2.0 * (1.0 + poisson)), rel=1e-12
        )

    @given(
        st.floats(min_value=1.0e3, max_value=1.0e12, allow_nan=False),
        st.floats(min_value=0.01, max_value=0.49, allow_nan=False),
    )
    def test_shear_modulus_is_always_below_youngs_modulus(
        self, modulus: float, poisson: float
    ) -> None:
        material = ContactMaterial(
            friction=0.5,
            restitution=0.3,
            youngs_modulus_pa=modulus,
            poisson_ratio=poisson,
        )
        assert 0.0 < material.shear_modulus_pa < modulus

    @given(_FRACTION, _FRACTION)
    def test_any_unit_interval_pair_is_admissible(
        self, friction: float, restitution: float
    ) -> None:
        material = _material(friction=friction, restitution=restitution)
        assert material.friction == friction
        assert material.restitution == restitution


class TestSolverSettings:
    def test_output_period_is_the_reciprocal_of_the_rate(self) -> None:
        assert SolverSettings(output_rate_hz=500.0).output_period_s == pytest.approx(
            2.0e-3
        )

    def test_default_downsampling_keeps_every_grain(self) -> None:
        assert SolverSettings(output_rate_hz=1000.0).downsample_grains == 1

    def test_non_positive_rate_is_refused(self) -> None:
        with pytest.raises(DomainInvariantError, match="output_rate_hz"):
            SolverSettings(output_rate_hz=0.0)

    def test_zero_downsampling_is_refused(self) -> None:
        with pytest.raises(DomainInvariantError, match="downsample_grains"):
            SolverSettings(output_rate_hz=1000.0, downsample_grains=0)

    @given(st.floats(min_value=1.0e-3, max_value=1.0e9, allow_nan=False))
    def test_rate_and_period_round_trip(self, rate: float) -> None:
        settings = SolverSettings(output_rate_hz=rate)
        assert 1.0 / settings.output_period_s == pytest.approx(rate, rel=1e-12)


class TestSwingCondition:
    def test_it_composes_the_geometry_delivery_condition(self) -> None:
        """#8609 already models face-open / shaft-lean / attack angle; this
        object must reuse it rather than declare a second copy."""
        swing = SwingCondition(clubhead_speed_mps=25.0, duration_s=0.02)
        assert isinstance(swing.delivery, DeliveryCondition)

    def test_a_supplied_delivery_is_preserved(self) -> None:
        delivery = DeliveryCondition(
            face_open_deg=20.0, shaft_lean_deg=6.0, attack_angle_deg=-8.0
        )
        swing = SwingCondition(
            clubhead_speed_mps=25.0, duration_s=0.02, delivery=delivery
        )
        assert swing.delivery == delivery

    def test_non_positive_speed_is_refused(self) -> None:
        with pytest.raises(DomainInvariantError, match="clubhead_speed_mps"):
            SwingCondition(clubhead_speed_mps=0.0, duration_s=0.02)

    def test_non_positive_duration_is_refused(self) -> None:
        with pytest.raises(DomainInvariantError, match="duration_s"):
            SwingCondition(clubhead_speed_mps=25.0, duration_s=0.0)

    def test_an_implausible_speed_is_refused(self) -> None:
        with pytest.raises(DomainInvariantError, match="clubhead_speed_mps"):
            SwingCondition(clubhead_speed_mps=500.0, duration_s=0.02)

    def test_a_greenside_bunker_shot_is_inertially_dominated(self) -> None:
        """ADR-0032: the depth and inertial terms cross at 6.8 m/s and a
        greenside shot is delivered at 20-27 m/s."""
        assert SwingCondition(
            clubhead_speed_mps=25.0, duration_s=0.02
        ).is_inertially_dominated

    def test_the_legacy_five_metres_per_second_is_not(self) -> None:
        """The hard-coded 5 m/s the old code used sits below the crossover."""
        assert not SwingCondition(
            clubhead_speed_mps=5.0, duration_s=0.02
        ).is_inertially_dominated

    @given(st.floats(min_value=0.01, max_value=100.0, allow_nan=False))
    def test_inertial_dominance_is_monotone_in_speed(self, speed: float) -> None:
        from bunkershot3d.domain import DRFT_INERTIAL_CROSSOVER_MPS

        swing = SwingCondition(clubhead_speed_mps=speed, duration_s=0.02)
        assert swing.is_inertially_dominated == (speed > DRFT_INERTIAL_CROSSOVER_MPS)


class TestTrajectorySource:
    def test_it_carries_the_file_and_the_duration(self) -> None:
        source = TrajectorySource(file="swing.csv", duration_s=0.07)
        assert source.file == "swing.csv"
        assert source.duration_s == 0.07

    def test_an_empty_file_is_refused(self) -> None:
        with pytest.raises(DomainInvariantError, match="file"):
            TrajectorySource(file="  ", duration_s=0.07)

    def test_non_positive_duration_is_refused(self) -> None:
        with pytest.raises(DomainInvariantError, match="duration_s"):
            TrajectorySource(file="swing.csv", duration_s=-1.0)
