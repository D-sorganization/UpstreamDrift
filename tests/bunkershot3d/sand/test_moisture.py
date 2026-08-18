"""Moisture-regime and cavitation-cap tests (issue #8610).

Two regimes, modelled separately (ADR-0032):

* damp / capillary -- apparent cohesion of order 1-10 kPa from menisci;
* saturated / cavitating -- shear-band dilation suction, hard-capped at
  about -100 kPa gauge.

The cavitation cap is the load-bearing test in this file. Without it a
poroelastic estimate invents multi-MPa suction and overpredicts club force
severalfold.
"""

from __future__ import annotations

import math

import pytest
from bunkershot3d.sand.exceptions import MoistureRegimeError
from bunkershot3d.sand.moisture import (
    ATMOSPHERIC_PRESSURE_PA,
    CAVITATION_PORE_PRESSURE_PA,
    CAVITATION_SUCTION_LIMIT_PA,
    DRY_SATURATION_CEILING,
    SATURATED_SATURATION_FLOOR,
    WATER_DENSITY_KG_M3,
    WATER_SURFACE_TENSION_N_PER_M,
    MoistureRegime,
    MoistureState,
    capillary_apparent_cohesion_pa,
    capillary_suction_pa,
    cavitation_limited_strength_gain_pa,
    clamp_pore_pressure_pa,
    clamp_suction_pa,
    classify_regime,
    degree_of_saturation,
)
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.unit

_PHI_RAD = math.radians(34.0)


class TestCavitationCap:
    def test_limit_is_about_minus_one_hundred_kilopascal_gauge(self) -> None:
        assert CAVITATION_PORE_PRESSURE_PA < 0.0
        assert pytest.approx(-1.0e5, rel=0.03) == CAVITATION_PORE_PRESSURE_PA
        assert (
            pytest.approx(-CAVITATION_PORE_PRESSURE_PA) == CAVITATION_SUCTION_LIMIT_PA
        )

    def test_limit_is_derived_from_vapour_pressure_not_a_magic_number(self) -> None:
        assert CAVITATION_PORE_PRESSURE_PA < 0.0
        assert abs(CAVITATION_PORE_PRESSURE_PA) < ATMOSPHERIC_PRESSURE_PA

    @pytest.mark.parametrize("requested_pa", [-1.0e6, -5.0e6, -2.5e7, -1e9])
    def test_multi_megapascal_suction_is_clamped(self, requested_pa: float) -> None:
        assert clamp_pore_pressure_pa(requested_pa) == pytest.approx(
            CAVITATION_PORE_PRESSURE_PA
        )

    def test_pressures_above_the_limit_pass_through(self) -> None:
        assert clamp_pore_pressure_pa(-5.0e4) == pytest.approx(-5.0e4)
        assert clamp_pore_pressure_pa(0.0) == 0.0
        assert clamp_pore_pressure_pa(2.0e5) == pytest.approx(2.0e5)

    def test_suction_magnitude_clamp(self) -> None:
        assert clamp_suction_pa(5.0e6) == pytest.approx(CAVITATION_SUCTION_LIMIT_PA)
        assert clamp_suction_pa(1.0e4) == pytest.approx(1.0e4)

    def test_negative_suction_magnitude_is_refused(self) -> None:
        with pytest.raises(MoistureRegimeError, match="non-negative"):
            clamp_suction_pa(-1.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_input_raises_rather_than_propagating(self, bad: float) -> None:
        with pytest.raises(MoistureRegimeError, match="finite"):
            clamp_pore_pressure_pa(bad)

    def test_clamp_is_a_raise_not_an_assert(self) -> None:
        """``python -O`` strips ``assert``; the guard must survive it."""
        import inspect

        from bunkershot3d.sand import moisture

        source = inspect.getsource(moisture)
        assert "assert " not in source

    def test_capped_dilation_gain_matches_the_published_order(self) -> None:
        """~65 kPa of extra shear strength (ADR-0032), not megapascals."""
        gain = cavitation_limited_strength_gain_pa(
            requested_suction_pa=5.0e6, friction_angle_rad=_PHI_RAD
        )
        assert gain == pytest.approx(6.7e4, rel=0.05)
        uncapped = 5.0e6 * math.tan(_PHI_RAD)
        assert gain < uncapped / 40.0


class TestRegimeClassification:
    def test_degree_of_saturation_formula(self) -> None:
        s = degree_of_saturation(
            gravimetric_water_content=0.05,
            void_ratio=0.60,
            particle_density_kg_m3=2600.0,
        )
        expected = 0.05 * 2600.0 / (0.60 * WATER_DENSITY_KG_M3)
        assert s == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("saturation", "regime"),
        [
            (0.0, MoistureRegime.DRY),
            (DRY_SATURATION_CEILING / 2.0, MoistureRegime.DRY),
            (0.2, MoistureRegime.DAMP_CAPILLARY),
            (0.8, MoistureRegime.DAMP_CAPILLARY),
            (SATURATED_SATURATION_FLOOR, MoistureRegime.SATURATED),
            (1.0, MoistureRegime.SATURATED),
        ],
    )
    def test_classification_boundaries(
        self, saturation: float, regime: MoistureRegime
    ) -> None:
        assert classify_regime(saturation) is regime

    @pytest.mark.parametrize("saturation", [-0.01, 1.01])
    def test_saturation_outside_zero_one_raises(self, saturation: float) -> None:
        with pytest.raises(MoistureRegimeError, match="saturation"):
            classify_regime(saturation)

    def test_declared_regime_must_match_the_saturation(self) -> None:
        with pytest.raises(MoistureRegimeError, match="declared"):
            MoistureState(
                gravimetric_water_content=0.20,
                degree_of_saturation=0.60,
                regime=MoistureRegime.DRY,
                meniscus_radius_m=1.2e-5,
            )

    def test_factory_classifies_explicitly(self) -> None:
        state = MoistureState.from_water_content(
            gravimetric_water_content=0.05,
            void_ratio=0.60,
            particle_density_kg_m3=2600.0,
            meniscus_radius_m=1.2e-5,
        )
        assert state.regime is MoistureRegime.DAMP_CAPILLARY
        assert state.degree_of_saturation == pytest.approx(
            degree_of_saturation(0.05, 0.60, 2600.0)
        )


class TestCapillaryRegime:
    def test_suction_is_two_sigma_over_r(self) -> None:
        r = 1.2e-5
        assert capillary_suction_pa(r) == pytest.approx(
            2.0 * WATER_SURFACE_TENSION_N_PER_M / r
        )

    def test_a_vanishing_meniscus_radius_is_clamped_not_infinite(self) -> None:
        assert capillary_suction_pa(1.0e-12) == pytest.approx(
            CAVITATION_SUCTION_LIMIT_PA
        )

    def test_zero_or_negative_radius_raises(self) -> None:
        with pytest.raises(MoistureRegimeError, match="meniscus radius"):
            capillary_suction_pa(0.0)

    def test_apparent_cohesion_lands_in_the_one_to_ten_kilopascal_band(self) -> None:
        """The published band for damp sand (research digest section 3)."""
        for saturation in (0.15, 0.35, 0.60):
            for radius_m in (2.0e-5, 1.2e-5, 6.0e-6):
                cohesion = capillary_apparent_cohesion_pa(
                    suction_pa=capillary_suction_pa(radius_m),
                    saturation=saturation,
                    friction_angle_rad=_PHI_RAD,
                )
                assert 5.0e2 <= cohesion <= 1.0e4

    def test_cohesion_vanishes_at_zero_saturation(self) -> None:
        assert (
            capillary_apparent_cohesion_pa(
                suction_pa=1.2e4, saturation=0.0, friction_angle_rad=_PHI_RAD
            )
            == 0.0
        )


class TestRegimeDispatch:
    def _damp(self) -> MoistureState:
        return MoistureState.from_water_content(
            gravimetric_water_content=0.05,
            void_ratio=0.60,
            particle_density_kg_m3=2600.0,
            meniscus_radius_m=1.2e-5,
        )

    def _dry(self) -> MoistureState:
        return MoistureState.from_water_content(
            gravimetric_water_content=0.0,
            void_ratio=0.60,
            particle_density_kg_m3=2600.0,
            meniscus_radius_m=1.2e-5,
        )

    def _saturated(self) -> MoistureState:
        return MoistureState.from_water_content(
            gravimetric_water_content=0.24,
            void_ratio=0.66,
            particle_density_kg_m3=2600.0,
            meniscus_radius_m=1.2e-5,
        )

    def test_dry_has_no_cohesion_and_no_suction(self) -> None:
        dry = self._dry()
        assert dry.regime is MoistureRegime.DRY
        assert dry.matric_suction_pa == 0.0
        assert dry.cohesive_strength_pa(_PHI_RAD) == 0.0

    def test_damp_uses_the_capillary_model(self) -> None:
        damp = self._damp()
        assert damp.regime is MoistureRegime.DAMP_CAPILLARY
        assert damp.matric_suction_pa > 0.0
        expected = capillary_apparent_cohesion_pa(
            suction_pa=damp.matric_suction_pa,
            saturation=damp.degree_of_saturation,
            friction_angle_rad=_PHI_RAD,
        )
        assert damp.cohesive_strength_pa(_PHI_RAD) == pytest.approx(expected)

    def test_saturated_requires_an_explicit_dilation_suction(self) -> None:
        saturated = self._saturated()
        assert saturated.regime is MoistureRegime.SATURATED
        with pytest.raises(MoistureRegimeError, match="dilation"):
            saturated.cohesive_strength_pa(_PHI_RAD)

    def test_saturated_gain_is_cavitation_capped(self) -> None:
        saturated = self._saturated()
        gain = saturated.cohesive_strength_pa(_PHI_RAD, dilation_suction_pa=5.0e6)
        assert gain == pytest.approx(6.7e4, rel=0.05)

    def test_damp_rejects_a_dilation_suction_argument(self) -> None:
        with pytest.raises(MoistureRegimeError, match="only"):
            self._damp().cohesive_strength_pa(_PHI_RAD, dilation_suction_pa=1.0e5)


@settings(deadline=None, max_examples=200)
@given(
    st.floats(
        min_value=-1.0e9,
        max_value=1.0e9,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )
)
def test_clamped_pore_pressure_never_goes_below_the_cavitation_floor(
    requested_pa: float,
) -> None:
    assert clamp_pore_pressure_pa(requested_pa) >= CAVITATION_PORE_PRESSURE_PA


@settings(deadline=None, max_examples=200)
@given(
    st.floats(
        min_value=0.0,
        max_value=1.0e9,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
    st.floats(min_value=0.1, max_value=0.9, allow_subnormal=False),
)
def test_strength_gain_is_bounded_by_the_capped_suction(
    requested_suction_pa: float, friction_ratio: float
) -> None:
    phi_rad = math.radians(20.0 + 30.0 * friction_ratio)
    gain = cavitation_limited_strength_gain_pa(requested_suction_pa, phi_rad)
    assert 0.0 <= gain <= CAVITATION_SUCTION_LIMIT_PA * math.tan(phi_rad)
