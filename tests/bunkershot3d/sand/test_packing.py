"""Packing / compaction state tests (issue #8610)."""

from __future__ import annotations

import pytest
from bunkershot3d.sand.exceptions import PackingStateError
from bunkershot3d.sand.packing import (
    RANDOM_CLOSE_PACKING_SOLID_FRACTION,
    RANDOM_LOOSE_PACKING_SOLID_FRACTION,
    SAND_VOID_RATIO_MAX,
    SAND_VOID_RATIO_MIN,
    Angularity,
    PackingState,
    solid_fraction_from_void_ratio,
    void_ratio_from_solid_fraction,
)
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.unit


class TestVoidRatioAlgebra:
    def test_round_trip(self) -> None:
        for phi in (0.50, 0.55, 0.60, 0.64):
            e = void_ratio_from_solid_fraction(phi)
            assert solid_fraction_from_void_ratio(e) == pytest.approx(phi)

    def test_packing_limits_map_to_the_module_void_ratio_bounds(self) -> None:
        assert void_ratio_from_solid_fraction(
            RANDOM_CLOSE_PACKING_SOLID_FRACTION
        ) == pytest.approx(SAND_VOID_RATIO_MIN)
        assert void_ratio_from_solid_fraction(
            RANDOM_LOOSE_PACKING_SOLID_FRACTION
        ) == pytest.approx(SAND_VOID_RATIO_MAX)

    @pytest.mark.parametrize("phi", [0.0, 1.0, 1.2, -0.1])
    def test_solid_fraction_outside_the_open_unit_interval_raises(
        self, phi: float
    ) -> None:
        with pytest.raises(PackingStateError, match="solid fraction"):
            void_ratio_from_solid_fraction(phi)

    def test_negative_void_ratio_raises(self) -> None:
        with pytest.raises(PackingStateError, match="void ratio"):
            solid_fraction_from_void_ratio(-0.2)


class TestPackingState:
    def test_from_solid_fraction_matches_the_quikrete_analogue(self) -> None:
        packing = PackingState.from_solid_fraction(
            particle_density_kg_m3=2600.0, solid_fraction=0.60
        )
        assert packing.solid_fraction == pytest.approx(0.60)
        assert packing.void_ratio == pytest.approx(2.0 / 3.0)
        assert packing.porosity == pytest.approx(0.40)
        assert packing.dry_bulk_density_kg_m3 == pytest.approx(1560.0)

    def test_relative_density_at_the_packing_limits(self) -> None:
        loose = PackingState(
            particle_density_kg_m3=2600.0,
            void_ratio=SAND_VOID_RATIO_MAX,
            void_ratio_min=SAND_VOID_RATIO_MIN,
            void_ratio_max=SAND_VOID_RATIO_MAX,
        )
        dense = PackingState(
            particle_density_kg_m3=2600.0,
            void_ratio=SAND_VOID_RATIO_MIN,
            void_ratio_min=SAND_VOID_RATIO_MIN,
            void_ratio_max=SAND_VOID_RATIO_MAX,
        )
        assert loose.relative_density == pytest.approx(0.0)
        assert dense.relative_density == pytest.approx(1.0)
        assert dense.relative_density_percent == pytest.approx(100.0)

    def test_from_relative_density_round_trips(self) -> None:
        for dr in (0.0, 0.125, 0.5, 0.875, 1.0):
            packing = PackingState.from_relative_density(
                particle_density_kg_m3=2600.0, relative_density=dr
            )
            assert packing.relative_density == pytest.approx(dr, abs=1e-12)

    def test_from_dry_bulk_density_round_trips(self) -> None:
        packing = PackingState.from_dry_bulk_density(
            particle_density_kg_m3=2600.0, dry_bulk_density_kg_m3=1560.0
        )
        assert packing.solid_fraction == pytest.approx(0.60)

    def test_denser_packing_means_higher_bulk_density(self) -> None:
        loose = PackingState.from_relative_density(
            particle_density_kg_m3=2600.0, relative_density=0.125
        )
        dense = PackingState.from_relative_density(
            particle_density_kg_m3=2600.0, relative_density=0.875
        )
        assert dense.dry_bulk_density_kg_m3 > loose.dry_bulk_density_kg_m3
        assert dense.void_ratio < loose.void_ratio

    def test_state_is_frozen(self) -> None:
        packing = PackingState.from_solid_fraction(2600.0, 0.60)
        with pytest.raises((AttributeError, TypeError)):
            packing.void_ratio = 0.5  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"particle_density_kg_m3": 0.0}, "particle density"),
            ({"void_ratio": -0.1}, "void ratio"),
            ({"void_ratio_min": 1.5}, "void_ratio_min"),
            ({"void_ratio": 2.0}, "outside"),
        ],
    )
    def test_invalid_states_raise(self, kwargs: dict, match: str) -> None:
        base = {
            "particle_density_kg_m3": 2600.0,
            "void_ratio": 0.66,
            "void_ratio_min": SAND_VOID_RATIO_MIN,
            "void_ratio_max": SAND_VOID_RATIO_MAX,
        }
        base.update(kwargs)
        with pytest.raises(PackingStateError, match=match):
            PackingState(**base)


class TestAngularity:
    def test_angular_shapes_are_flagged_desirable(self) -> None:
        assert Angularity.ANGULAR.is_usga_desirable
        assert Angularity.VERY_ANGULAR.is_usga_desirable
        assert not Angularity.ROUNDED.is_usga_desirable

    def test_ordering_is_rounded_to_angular(self) -> None:
        order = [a.shape_index for a in Angularity]
        assert order == sorted(order)


@settings(deadline=None, max_examples=100)
@given(
    st.floats(min_value=0.0, max_value=1.0, allow_subnormal=False),
    st.floats(min_value=1500.0, max_value=3000.0, allow_subnormal=False),
)
def test_relative_density_is_monotone_in_bulk_density(dr: float, rho_s: float) -> None:
    packing = PackingState.from_relative_density(
        particle_density_kg_m3=rho_s, relative_density=dr
    )
    assert 0.0 <= packing.relative_density <= 1.0
    assert packing.dry_bulk_density_kg_m3 == pytest.approx(
        rho_s * packing.solid_fraction
    )
    assert packing.porosity + packing.solid_fraction == pytest.approx(1.0)
