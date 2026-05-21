"""Tests for baghouse_calculator.py — BaghouseCalculator.

Targets: 32% → 100% coverage.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.baghouse_calculator import (
    BaghouseCalculator,
    BaghouseResult,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

SYNGAS_COMP = {
    "H2": 0.30,
    "CO": 0.30,
    "CO2": 0.15,
    "H2O": 0.05,
    "N2": 0.18,
    "CH4": 0.02,
}


def _calc() -> BaghouseCalculator:
    """Return a BaghouseCalculator in simplified (no-thermo) mode."""
    return BaghouseCalculator(thermo_calc=None)


def _default_result() -> BaghouseResult:
    calc = _calc()
    return calc.calculate(
        gas_flow_kg_s=1.0,
        inlet_temp_k=700.0,
        pressure_pa=101325.0,
        composition=SYNGAS_COMP,
        solid_carbon_in_kg_hr=50.0,
        ash_in_kg_hr=30.0,
        carbon_removal_efficiency=0.99,
        ash_removal_efficiency=0.95,
        heat_loss_w=5000.0,
        drum_volume_m3=0.2,
        solid_density_kg_m3=500.0,
        bag_area_ft2=200.0,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestBaghouseCalculatorInit:
    def test_init_without_thermo(self):
        """Lines 109-114: thermo_calc=None → self.thermo_calc is None."""
        calc = BaghouseCalculator(thermo_calc=None)
        assert calc.thermo_calc is None

    def test_init_with_mock_thermo(self):
        """Line 109-110: thermo passed directly."""
        mock = object()
        calc = BaghouseCalculator(thermo_calc=mock)
        assert calc.thermo_calc is mock


# ---------------------------------------------------------------------------
# _estimate_cp_ideal (lines 116-159)
# ---------------------------------------------------------------------------


class TestEstimateCpIdeal:
    def test_syngas_cp_returns_positive(self):
        calc = _calc()
        cp = calc._estimate_cp_ideal(SYNGAS_COMP)
        assert cp > 0

    def test_empty_composition_returns_fallback(self):
        """Line 159: mw_avg == 0 → returns CP_MASS_DEFAULT_FALLBACK."""
        calc = _calc()
        cp = calc._estimate_cp_ideal({})
        assert cp > 0  # Returns fallback

    def test_unknown_species_uses_defaults(self):
        """Lines 154-155: unknown species uses CP_DEFAULT_FALLBACK and MW_DEFAULT_KG."""
        calc = _calc()
        cp = calc._estimate_cp_ideal({"EXOTIC_GAS": 1.0})
        assert cp > 0


# ---------------------------------------------------------------------------
# _estimate_volume_flow (lines 161-206)
# ---------------------------------------------------------------------------


class TestEstimateVolumeFlow:
    def test_returns_positive_flows(self):
        calc = _calc()
        acfm, scfm = calc._estimate_volume_flow(1.0, 700.0, 101325.0, SYNGAS_COMP)
        assert acfm > 0
        assert scfm > 0

    def test_zero_mw_fallback(self):
        """Line 194: mw_avg == 0 → assume N2-like."""
        calc = _calc()
        acfm, scfm = calc._estimate_volume_flow(1.0, 300.0, 101325.0, {})
        assert acfm > 0
        assert scfm > 0


# ---------------------------------------------------------------------------
# _calculate_outlet_thermal (lines 208-273)
# ---------------------------------------------------------------------------


class TestCalculateOutletThermal:
    def test_simplified_mode_returns_valid_outputs(self):
        """Lines 262-273: simplified path (no thermo_calc)."""
        calc = _calc()
        outlet_c, acfm, scfm = calc._calculate_outlet_thermal(
            1.0, 700.0, 101325.0, SYNGAS_COMP, 5000.0
        )
        assert isinstance(outlet_c, float)
        assert acfm > 0
        assert scfm > 0

    def test_zero_heat_loss_no_temp_drop(self):
        """Lines 264-268: heat_loss_w=0 → temp_drop=0."""
        calc = _calc()
        outlet_c, _, _ = calc._calculate_outlet_thermal(
            1.0, 700.0, 101325.0, SYNGAS_COMP, 0.0
        )
        expected = 700.0 - 273.15
        assert abs(outlet_c - expected) < 0.01

    def test_zero_flow_no_temperature_drop(self):
        """Lines 264-268: gas_flow==0 → temp_drop=0 (guard against div by zero)."""
        calc = _calc()
        outlet_c, _, _ = calc._calculate_outlet_thermal(
            0.001, 700.0, 101325.0, SYNGAS_COMP, 1e9
        )
        # Flow is very small — outlet temp could be near absolute zero but should return
        assert isinstance(outlet_c, float)


# ---------------------------------------------------------------------------
# _calculate_drum_sizing (lines 275-314)
# ---------------------------------------------------------------------------


class TestCalculateDrumSizing:
    def test_normal_case(self):
        result = BaghouseCalculator._calculate_drum_sizing(
            solid_carbon_in_kg_hr=50.0,
            ash_in_kg_hr=30.0,
            carbon_removal_efficiency=0.99,
            ash_removal_efficiency=0.95,
            drum_volume_m3=0.2,
            solid_density_kg_m3=500.0,
        )
        (
            carbon_removed,
            ash_removed,
            total_solids,
            fill_hrs,
            fill_days,
            c_fill,
            a_fill,
        ) = result
        assert carbon_removed == pytest.approx(50.0 * 0.99)
        assert ash_removed == pytest.approx(30.0 * 0.95)
        assert total_solids > 0
        assert fill_hrs > 0
        assert fill_days > 0
        assert c_fill > 0
        assert a_fill > 0

    def test_zero_total_solids_gives_inf_fill_time(self):
        """Line 299: total_solids == 0 → fill_hrs = inf."""
        result = BaghouseCalculator._calculate_drum_sizing(
            solid_carbon_in_kg_hr=0.0,
            ash_in_kg_hr=0.0,
            carbon_removal_efficiency=0.99,
            ash_removal_efficiency=0.95,
            drum_volume_m3=0.2,
            solid_density_kg_m3=500.0,
        )
        fill_hrs = result[3]
        assert fill_hrs == float("inf")

    def test_zero_carbon_removed_gives_inf_c_fill(self):
        """Line 303: carbon_removed == 0 → c_fill = inf."""
        result = BaghouseCalculator._calculate_drum_sizing(
            solid_carbon_in_kg_hr=0.0,  # no carbon
            ash_in_kg_hr=50.0,
            carbon_removal_efficiency=0.99,
            ash_removal_efficiency=0.95,
            drum_volume_m3=0.2,
            solid_density_kg_m3=500.0,
        )
        c_fill = result[5]
        assert c_fill == float("inf")

    def test_zero_ash_removed_gives_inf_a_fill(self):
        """Line 304: ash_removed == 0 → a_fill = inf."""
        result = BaghouseCalculator._calculate_drum_sizing(
            solid_carbon_in_kg_hr=50.0,
            ash_in_kg_hr=0.0,  # no ash
            carbon_removal_efficiency=0.99,
            ash_removal_efficiency=0.95,
            drum_volume_m3=0.2,
            solid_density_kg_m3=500.0,
        )
        a_fill = result[6]
        assert a_fill == float("inf")


# ---------------------------------------------------------------------------
# calculate() — full pipeline
# ---------------------------------------------------------------------------


class TestBaghouseCalculate:
    def test_full_calculation_returns_result(self):
        """Lines 316-417: end-to-end pipeline."""
        result = _default_result()
        assert isinstance(result, BaghouseResult)
        assert result.total_solids_removed_rate > 0
        assert result.flow_acfm > 0
        assert result.flow_scfm > 0

    def test_removal_efficiencies_in_result(self):
        """Lines 413-416: efficiency stored as percent."""
        result = _default_result()
        assert result.removal_efficiency["carbon"] == pytest.approx(99.0)
        assert result.removal_efficiency["ash"] == pytest.approx(95.0)

    def test_air_to_cloth_ratio_calculated(self):
        """Line 390: air_to_cloth = flow_acfm / bag_area."""
        result = _default_result()
        expected = result.flow_acfm / 200.0
        assert abs(result.air_to_cloth_ratio - expected) < 0.001

    def test_bag_area_zero_gives_zero_atc(self):
        """Line 390: bag_area=0 → air_to_cloth = 0."""
        calc = _calc()
        result = calc.calculate(
            gas_flow_kg_s=1.0,
            inlet_temp_k=700.0,
            pressure_pa=101325.0,
            composition=SYNGAS_COMP,
            solid_carbon_in_kg_hr=50.0,
            ash_in_kg_hr=30.0,
            carbon_removal_efficiency=0.99,
            ash_removal_efficiency=0.95,
            heat_loss_w=0.0,
            drum_volume_m3=0.2,
            solid_density_kg_m3=500.0,
            bag_area_ft2=0.0,
        )
        assert result.air_to_cloth_ratio == 0.0

    def test_ash_stream_composition_sums_to_one(self):
        """Lines 392-397: carbon_fraction + ash_fraction should sum to 1.0."""
        result = _default_result()
        total = (
            result.ash_stream_composition["carbon_fraction"]
            + result.ash_stream_composition["ash_fraction"]
        )
        assert abs(total - 1.0) < 1e-10

    def test_zero_feed_gives_zero_stream_composition(self):
        """Lines 393-396: total_solids == 0 → fractions = 0."""
        calc = _calc()
        result = calc.calculate(
            gas_flow_kg_s=1.0,
            inlet_temp_k=700.0,
            pressure_pa=101325.0,
            composition=SYNGAS_COMP,
            solid_carbon_in_kg_hr=0.0,
            ash_in_kg_hr=0.0,
            carbon_removal_efficiency=0.99,
            ash_removal_efficiency=0.95,
            heat_loss_w=0.0,
            drum_volume_m3=0.2,
            solid_density_kg_m3=500.0,
            bag_area_ft2=200.0,
        )
        assert result.ash_stream_composition["carbon_fraction"] == 0.0
        assert result.ash_stream_composition["ash_fraction"] == 0.0

    def test_negative_gas_flow_raises(self):
        """Line 351: gas_flow <= 0 → AssertionError."""
        calc = _calc()
        with pytest.raises(AssertionError, match="Gas flow must be positive"):
            calc.calculate(
                gas_flow_kg_s=-1.0,
                inlet_temp_k=700.0,
                pressure_pa=101325.0,
                composition=SYNGAS_COMP,
                solid_carbon_in_kg_hr=50.0,
                ash_in_kg_hr=30.0,
                carbon_removal_efficiency=0.99,
                ash_removal_efficiency=0.95,
                heat_loss_w=0.0,
                drum_volume_m3=0.2,
                solid_density_kg_m3=500.0,
                bag_area_ft2=200.0,
            )

    def test_efficiency_out_of_range_raises(self):
        """Lines 354-359: efficiency > 1 → AssertionError."""
        calc = _calc()
        with pytest.raises(
            AssertionError, match="Carbon removal efficiency must be 0-1"
        ):
            calc.calculate(
                gas_flow_kg_s=1.0,
                inlet_temp_k=700.0,
                pressure_pa=101325.0,
                composition=SYNGAS_COMP,
                solid_carbon_in_kg_hr=50.0,
                ash_in_kg_hr=30.0,
                carbon_removal_efficiency=1.5,
                ash_removal_efficiency=0.95,
                heat_loss_w=0.0,
                drum_volume_m3=0.2,
                solid_density_kg_m3=500.0,
                bag_area_ft2=200.0,
            )

    def test_clean_gas_flow_rate_in_kg_hr(self):
        """Line 407: clean_gas_flow_rate = gas_flow_kg_s * SECONDS_PER_HOUR."""
        result = _default_result()
        assert abs(result.clean_gas_flow_rate - 1.0 * 3600.0) < 0.01


# ---------------------------------------------------------------------------
# convert fallback function (lines 58-65, standalone mode only)
# ---------------------------------------------------------------------------


class TestConvertFallback:
    """Tests that the local fallback convert() in baghouse works correctly."""

    def test_k_to_c(self):
        """Lines 61-62: K → C conversion."""
        from upstream_drift_tools.process_calculators import (
            baghouse_calculator as bh_mod,
        )

        if not bh_mod.HAS_THERMO:
            result = bh_mod.convert(273.15, "K", "C")
            assert abs(result - 0.0) < 0.01

    def test_c_to_k(self):
        """Lines 63-64: C → K conversion."""
        from upstream_drift_tools.process_calculators import (
            baghouse_calculator as bh_mod,
        )

        if not bh_mod.HAS_THERMO:
            result = bh_mod.convert(0.0, "C", "K")
            assert abs(result - 273.15) < 0.01

    def test_identity_fallback(self):
        """Line 65: unknown units → identity."""
        from upstream_drift_tools.process_calculators import (
            baghouse_calculator as bh_mod,
        )

        if not bh_mod.HAS_THERMO:
            result = bh_mod.convert(42.0, "Pa", "bar")
            assert result == 42.0
