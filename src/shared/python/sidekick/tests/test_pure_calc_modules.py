# ruff: noqa: E501
"""Tests for pure-math upstream_drift_tools modules to reach 100% coverage.

Targets (all stateless/pure-logic, no external dependencies):
- calculators.thermo.thermo_properties (ThermoPropertiesCalculator, ThermoResult)
- calculators.conversion.core (convert_temperature, _require_positive_finite,
  convert_via_table, standard_to_actual_flow, etc.)
- process_calculators.electrode_advancement_calculator
- process_calculators.water_vapor_pressure_calculator
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# ThermoPropertiesCalculator
# ---------------------------------------------------------------------------


class TestThermoPropertiesCalculator:
    """Complete coverage of thermo_properties.py."""

    def _calc(self):
        from upstream_drift_tools.calculators.thermo.thermo_properties import (
            ThermoPropertiesCalculator,
        )

        return ThermoPropertiesCalculator()

    def test_basic_calculation_air(self):
        """Standard air at 25 C, 101.325 kPa → non-trivial density."""
        calc = self._calc()
        result = calc.calculate(
            temperature_c=25.0,
            pressure_kpa=101.325,
            composition={"N2": 79.0, "O2": 21.0},
        )
        assert result.temperature_k == pytest.approx(298.15, abs=0.01)
        assert result.pressure_pa == pytest.approx(101325.0, rel=1e-6)
        assert result.density_kg_m3 > 0
        assert result.molecular_weight_g_mol > 0
        assert result.molar_volume_m3_mol > 0

    def test_enthalpy_at_reference_temperature(self):
        """At 298.15 C → temp_k ≈ 571.3 K → enthalpy = cp*(571.3-298.15)."""
        calc = self._calc()
        result = calc.calculate(
            temperature_c=298.15,
            pressure_kpa=101.325,
            composition={"N2": 100.0},
        )
        # enthalpy should be positive (above 298 K reference)
        assert result.enthalpy_j_mol > 0

    def test_entropy_and_gibbs_calculation(self):
        """Entropy and Gibbs energy should be computed without error."""
        calc = self._calc()
        result = calc.calculate(
            temperature_c=500.0,
            pressure_kpa=200.0,
            composition={"CO2": 50.0, "N2": 50.0},
        )
        assert math.isfinite(result.entropy_j_molk)
        assert math.isfinite(result.gibbs_energy_j_mol)

    def test_normalization_of_composition(self):
        """Composition fractions need not sum to 1; should be normalized."""
        calc = self._calc()
        # Supply values summing to 200 (not 100)
        r1 = calc.calculate(
            temperature_c=100.0,
            pressure_kpa=101.325,
            composition={"N2": 79.0, "O2": 21.0},
        )
        r2 = calc.calculate(
            temperature_c=100.0,
            pressure_kpa=101.325,
            composition={"N2": 158.0, "O2": 42.0},  # × 2 → same fractions
        )
        assert r1.molecular_weight_g_mol == pytest.approx(r2.molecular_weight_g_mol)

    def test_unknown_species_uses_defaults(self):
        """Unknown species fall back to MW=28, Cp=29."""
        calc = self._calc()
        result = calc.calculate(
            temperature_c=100.0,
            pressure_kpa=101.325,
            composition={"UNKNOWN_GAS": 100.0},
        )
        assert result.molecular_weight_g_mol == pytest.approx(28.0, rel=0.01)
        assert result.cp_j_molk == pytest.approx(29.0, rel=0.01)

    def test_zero_total_fraction_uses_default_total(self):
        """If all composition values are 0, normalise to total=1 (guard)."""
        calc = self._calc()
        result = calc.calculate(
            temperature_c=100.0,
            pressure_kpa=101.325,
            composition={"N2": 0.0},
        )
        # Should return a result without error
        assert math.isfinite(result.density_kg_m3)

    def test_gamma_greater_than_one(self):
        """Cp/Cv ratio should be > 1."""
        calc = self._calc()
        result = calc.calculate(
            temperature_c=25.0,
            pressure_kpa=101.325,
            composition={"N2": 79.0, "O2": 21.0},
        )
        assert result.gamma > 1.0

    def test_thermo_result_dataclass(self):
        """ThermoResult fields are accessible and typed correctly."""
        from upstream_drift_tools.calculators.thermo.thermo_properties import (
            ThermoResult,
        )

        r = ThermoResult(
            temperature_k=300.0,
            pressure_pa=101325.0,
            molecular_weight_g_mol=28.0,
            molar_volume_m3_mol=0.0246,
            density_kg_m3=1.14,
            enthalpy_j_mol=50.0,
            entropy_j_molk=2.0,
            gibbs_energy_j_mol=-550.0,
            cp_j_molk=29.0,
            cv_j_molk=20.7,
            gamma=1.4,
        )
        assert r.database_used == "ideal_gas"
        assert r.gamma == pytest.approx(1.4)


# ---------------------------------------------------------------------------
# calculators.conversion.core
# ---------------------------------------------------------------------------


class TestConversionCore:
    """Cover all branches of the functional helpers in conversion/core.py."""

    def test_require_positive_finite_passes_positive(self):
        from upstream_drift_tools.calculators.conversion.core import (
            _require_positive_finite,
        )

        _require_positive_finite(1.0, "x")  # Should not raise

    def test_require_positive_finite_rejects_zero(self):
        from upstream_drift_tools.calculators.conversion.core import (
            _require_positive_finite,
        )

        with pytest.raises(ValueError, match="positive and finite"):
            _require_positive_finite(0.0, "x")

    def test_require_positive_finite_rejects_negative(self):
        from upstream_drift_tools.calculators.conversion.core import (
            _require_positive_finite,
        )

        with pytest.raises(ValueError, match="positive and finite"):
            _require_positive_finite(-1.0, "x")

    def test_require_positive_finite_rejects_inf(self):
        from upstream_drift_tools.calculators.conversion.core import (
            _require_positive_finite,
        )

        with pytest.raises(ValueError, match="positive and finite"):
            _require_positive_finite(float("inf"), "x")

    def test_convert_via_table_same_unit(self):
        from upstream_drift_tools.calculators.conversion.core import convert_via_table

        table = {"m": 1.0, "km": 1000.0}
        assert convert_via_table(5.0, "m", "m", table) == 5.0

    def test_convert_via_table_different_units(self):
        from upstream_drift_tools.calculators.conversion.core import convert_via_table

        table = {"m": 1.0, "km": 1000.0}
        result = convert_via_table(1.0, "km", "m", table)
        assert result == pytest.approx(1000.0)

    def test_temperature_same_unit(self):
        from upstream_drift_tools.calculators.conversion.core import (
            convert_temperature,
        )

        assert convert_temperature(100.0, "C", "C") == 100.0

    def test_temperature_kelvin_to_celsius(self):
        from upstream_drift_tools.calculators.conversion.core import (
            convert_temperature,
        )

        result = convert_temperature(273.15, "K", "C")
        assert result == pytest.approx(0.0, abs=0.01)

    def test_temperature_celsius_to_kelvin(self):
        from upstream_drift_tools.calculators.conversion.core import (
            convert_temperature,
        )

        result = convert_temperature(0.0, "C", "K")
        assert result == pytest.approx(273.15, abs=0.01)

    def test_temperature_fahrenheit_to_kelvin(self):
        from upstream_drift_tools.calculators.conversion.core import (
            convert_temperature,
        )

        result = convert_temperature(32.0, "F", "K")
        assert result == pytest.approx(273.15, abs=0.01)

    def test_temperature_rankine_to_kelvin(self):
        from upstream_drift_tools.calculators.conversion.core import (
            convert_temperature,
        )

        result = convert_temperature(491.67, "R", "K")
        assert result == pytest.approx(273.15, abs=0.1)

    def test_temperature_kelvin_to_fahrenheit(self):
        from upstream_drift_tools.calculators.conversion.core import (
            convert_temperature,
        )

        result = convert_temperature(273.15, "K", "F")
        assert result == pytest.approx(32.0, abs=0.01)

    def test_temperature_kelvin_to_rankine(self):
        from upstream_drift_tools.calculators.conversion.core import (
            convert_temperature,
        )

        result = convert_temperature(273.15, "K", "R")
        assert result == pytest.approx(491.67, abs=0.1)

    def test_temperature_unknown_from_unit(self):
        from upstream_drift_tools.calculators.conversion.core import (
            convert_temperature,
        )

        with pytest.raises(ValueError, match="Unknown temperature unit"):
            convert_temperature(100.0, "X", "K")

    def test_temperature_unknown_to_unit(self):
        from upstream_drift_tools.calculators.conversion.core import (
            convert_temperature,
        )

        with pytest.raises(ValueError, match="Unknown temperature unit"):
            convert_temperature(100.0, "K", "X")

    def test_standard_to_actual_flow(self):
        from upstream_drift_tools.calculators.conversion.core import (
            standard_to_actual_flow,
        )
        from upstream_drift_tools.calculators.conversion.tables import StandardCondition

        # scfm=100 at standard → actual should differ at elevated T
        result = standard_to_actual_flow(100.0, 400.0, 101325.0, StandardCondition.NTP)
        assert result > 100.0  # Higher T → more volume

    def test_standard_to_actual_flow_rejects_bad_temperature(self):
        from upstream_drift_tools.calculators.conversion.core import (
            standard_to_actual_flow,
        )
        from upstream_drift_tools.calculators.conversion.tables import StandardCondition

        with pytest.raises(ValueError, match="positive and finite"):
            standard_to_actual_flow(100.0, 0.0, 101325.0, StandardCondition.NTP)

    def test_actual_to_standard_flow(self):
        from upstream_drift_tools.calculators.conversion.core import (
            actual_to_standard_flow,
        )
        from upstream_drift_tools.calculators.conversion.tables import StandardCondition

        result = actual_to_standard_flow(100.0, 293.15, 101325.0, StandardCondition.NTP)
        assert result > 0

    def test_actual_to_standard_flow_rejects_bad_pressure(self):
        from upstream_drift_tools.calculators.conversion.core import (
            actual_to_standard_flow,
        )
        from upstream_drift_tools.calculators.conversion.tables import StandardCondition

        with pytest.raises(ValueError, match="positive and finite"):
            actual_to_standard_flow(100.0, 300.0, 0.0, StandardCondition.NTP)

    def test_scfm_to_standard_m3_hr_same_standard(self):
        """When standard == reference, no temperature/pressure correction needed."""
        from upstream_drift_tools.calculators.conversion.core import (
            scfm_to_standard_m3_per_hour,
        )
        from upstream_drift_tools.calculators.conversion.tables import StandardCondition

        result = scfm_to_standard_m3_per_hour(
            100.0, StandardCondition.NTP, StandardCondition.NTP
        )
        # Should be 100 * SCFM_TO_CU_METER_PER_HOUR_AT_60F
        assert result > 0
        # Different-standard path (correction branch)
        result2 = scfm_to_standard_m3_per_hour(
            100.0, StandardCondition.STP, StandardCondition.NTP
        )
        assert result2 > 0

    def test_standard_m3_hr_to_scfm_same_standard(self):
        from upstream_drift_tools.calculators.conversion.core import (
            standard_m3_per_hour_to_scfm,
        )
        from upstream_drift_tools.calculators.conversion.tables import StandardCondition

        result = standard_m3_per_hour_to_scfm(
            100.0, StandardCondition.NTP, StandardCondition.NTP
        )
        assert result > 0
        # Different reference → uses correction branch
        result2 = standard_m3_per_hour_to_scfm(
            100.0, StandardCondition.STP, StandardCondition.NTP
        )
        assert result2 > 0


# ---------------------------------------------------------------------------
# ElectrodeAdvancementCalculator
# ---------------------------------------------------------------------------


class TestElectrodeAdvancementCalculator:
    """Cover all lines of electrode_advancement_calculator.py."""

    def test_init_default_consumption_rate(self):
        from upstream_drift_tools.process_calculators.electrode_advancement_calculator import (
            ElectrodeAdvancementCalculator,
        )

        calc = ElectrodeAdvancementCalculator()
        assert calc.consumption_rate == pytest.approx(0.5)

    def test_calculate_consumption_basic(self):
        from upstream_drift_tools.process_calculators.electrode_advancement_calculator import (
            ElectrodeAdvancementCalculator,
        )

        calc = ElectrodeAdvancementCalculator()
        result = calc.calculate_consumption(current_ka=10.0, time_hrs=2.0)
        # 0.5 * 10 * 2 = 10.0
        assert result == pytest.approx(10.0)

    def test_calculate_consumption_zero_time(self):
        from upstream_drift_tools.process_calculators.electrode_advancement_calculator import (
            ElectrodeAdvancementCalculator,
        )

        calc = ElectrodeAdvancementCalculator()
        assert calc.calculate_consumption(current_ka=5.0, time_hrs=0.0) == 0.0


# ---------------------------------------------------------------------------
# WaterVaporPressureCalculator
# ---------------------------------------------------------------------------


class TestWaterVaporPressureCalculator:
    """Cover the wrapper around SyngasWaterCalculator."""

    def test_calculate_vapor_pressure_basic(self):
        from upstream_drift_tools.process_calculators.water_vapor_pressure_calculator import (
            WaterVaporPressureCalculator,
        )

        calc = WaterVaporPressureCalculator()
        pressure_pa = calc.calculate_vapor_pressure(temperature_c=25.0)
        # At 25°C, vapor pressure of water ≈ 3169 Pa
        assert pressure_pa == pytest.approx(3169.0, rel=0.1)

    def test_calculate_vapor_pressure_at_100c(self):
        from upstream_drift_tools.process_calculators.water_vapor_pressure_calculator import (
            WaterVaporPressureCalculator,
        )

        calc = WaterVaporPressureCalculator()
        pressure_pa = calc.calculate_vapor_pressure(temperature_c=100.0)
        # At 100°C, vapor pressure ≈ 101325 Pa (1 atm)
        assert pressure_pa == pytest.approx(101325.0, rel=0.15)

    def test_calculate_vapor_pressure_with_method(self):
        from upstream_drift_tools.process_calculators.water_vapor_pressure_calculator import (
            WaterVaporPressureCalculator,
        )

        calc = WaterVaporPressureCalculator()
        pressure_auto = calc.calculate_vapor_pressure(temperature_c=50.0, method="auto")
        assert pressure_auto > 0
