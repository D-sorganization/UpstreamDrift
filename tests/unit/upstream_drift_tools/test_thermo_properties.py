"""Tests for src.shared.python.upstream_drift_tools.calculators.thermo.thermo_properties (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.upstream_drift_tools.calculators.thermo.thermo_properties import (
    MOLAR_CP_298,
    MOLECULAR_WEIGHTS,
    R_GAS,
    ThermoPropertiesCalculator,
    ThermoResult,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_molecular_weights_nonempty(self) -> None:
        assert len(MOLECULAR_WEIGHTS) > 0

    def test_n2_molecular_weight(self) -> None:
        assert abs(MOLECULAR_WEIGHTS["N2"] - 28.014) < 0.001

    def test_h2_molecular_weight(self) -> None:
        assert abs(MOLECULAR_WEIGHTS["H2"] - 2.016) < 0.001

    def test_molar_cp_nonempty(self) -> None:
        assert len(MOLAR_CP_298) > 0

    def test_n2_molar_cp(self) -> None:
        assert abs(MOLAR_CP_298["N2"] - 29.12) < 0.01

    def test_r_gas_value(self) -> None:
        assert abs(R_GAS - 8.314) < 0.001

    def test_same_species_in_both_tables(self) -> None:
        assert set(MOLECULAR_WEIGHTS.keys()) == set(MOLAR_CP_298.keys())


# ---------------------------------------------------------------------------
# ThermoResult dataclass
# ---------------------------------------------------------------------------


class TestThermoResult:
    def _make(self) -> ThermoResult:
        return ThermoResult(
            temperature_k=373.15,
            pressure_pa=101325.0,
            molecular_weight_g_mol=28.014,
            molar_volume_m3_mol=0.030,
            density_kg_m3=0.93,
            enthalpy_j_mol=2189.0,
            entropy_j_molk=5.0,
            gibbs_energy_j_mol=-662.0,
            cp_j_molk=29.12,
            cv_j_molk=20.81,
            gamma=1.4,
        )

    def test_thermo_properties_construct(self) -> None:
        r = self._make()
        assert r.temperature_k == 373.15

    def test_default_database(self) -> None:
        r = self._make()
        assert r.database_used == "ideal_gas"

    def test_all_fields_accessible(self) -> None:
        r = self._make()
        assert r.density_kg_m3 > 0
        assert r.gamma > 0


# ---------------------------------------------------------------------------
# ThermoPropertiesCalculator.calculate
# ---------------------------------------------------------------------------


class TestCalculate:
    _CALC = ThermoPropertiesCalculator()
    _AIR = {"N2": 79.0, "O2": 21.0}

    def test_returns_thermo_result(self) -> None:
        result = self._CALC.calculate(25.0, 101.325, self._AIR)
        assert isinstance(result, ThermoResult)

    def test_temperature_k_correct(self) -> None:
        result = self._CALC.calculate(0.0, 101.325, self._AIR)
        assert abs(result.temperature_k - 273.15) < 0.01

    def test_pressure_pa_correct(self) -> None:
        result = self._CALC.calculate(25.0, 101.325, self._AIR)
        assert abs(result.pressure_pa - 101325.0) < 1.0

    def test_density_positive(self) -> None:
        result = self._CALC.calculate(25.0, 101.325, self._AIR)
        assert result.density_kg_m3 > 0.0

    def test_thermo_properties_higher_pressure_higher_density(self) -> None:
        low = self._CALC.calculate(25.0, 101.325, self._AIR)
        high = self._CALC.calculate(25.0, 202.65, self._AIR)
        assert high.density_kg_m3 > low.density_kg_m3

    def test_higher_temperature_lower_density(self) -> None:
        cold = self._CALC.calculate(0.0, 101.325, self._AIR)
        hot = self._CALC.calculate(500.0, 101.325, self._AIR)
        assert hot.density_kg_m3 < cold.density_kg_m3

    def test_gamma_greater_than_one(self) -> None:
        result = self._CALC.calculate(25.0, 101.325, self._AIR)
        assert result.gamma > 1.0

    def test_cp_greater_than_cv(self) -> None:
        result = self._CALC.calculate(25.0, 101.325, self._AIR)
        assert result.cp_j_molk > result.cv_j_molk

    def test_gibbs_equals_h_minus_ts(self) -> None:
        result = self._CALC.calculate(25.0, 101.325, self._AIR)
        expected = result.enthalpy_j_mol - result.temperature_k * result.entropy_j_molk
        assert abs(result.gibbs_energy_j_mol - expected) < 0.01

    def test_molar_volume_ideal_gas(self) -> None:
        result = self._CALC.calculate(25.0, 101.325, self._AIR)
        # PV = RT for 1 mol => V = RT/P
        expected = R_GAS * result.temperature_k / result.pressure_pa
        assert abs(result.molar_volume_m3_mol - expected) < 1e-6

    def test_pure_n2(self) -> None:
        result = self._CALC.calculate(25.0, 101.325, {"N2": 1.0})
        assert abs(result.molecular_weight_g_mol - MOLECULAR_WEIGHTS["N2"]) < 0.001

    def test_unnormalized_composition_same_result(self) -> None:
        r1 = self._CALC.calculate(25.0, 101.325, {"N2": 79.0, "O2": 21.0})
        r2 = self._CALC.calculate(25.0, 101.325, {"N2": 0.79, "O2": 0.21})
        assert abs(r1.density_kg_m3 - r2.density_kg_m3) < 1e-6

    def test_single_species_ch4(self) -> None:
        result = self._CALC.calculate(25.0, 101.325, {"CH4": 1.0})
        assert abs(result.molecular_weight_g_mol - MOLECULAR_WEIGHTS["CH4"]) < 0.001
        assert result.density_kg_m3 > 0.0

    def test_enthalpy_increases_with_temperature(self) -> None:
        low = self._CALC.calculate(25.0, 101.325, self._AIR)
        high = self._CALC.calculate(500.0, 101.325, self._AIR)
        assert high.enthalpy_j_mol > low.enthalpy_j_mol

    def test_entropy_increases_with_temperature_at_const_p(self) -> None:
        low = self._CALC.calculate(100.0, 101.325, self._AIR)
        high = self._CALC.calculate(500.0, 101.325, self._AIR)
        assert high.entropy_j_molk > low.entropy_j_molk

    def test_reference_temperature_enthalpy_near_zero(self) -> None:
        # At T=298.15 C? No — 298.15 K is about 25°C. At 25°C, enthalpy relative
        # to 298.15 K reference is near zero (T=298.15 K, delta = 0).
        result = self._CALC.calculate(25.0, 101.325, {"N2": 1.0})
        # enthalpy = cp * (T_K - 298.15) and T_K = 298.15 here
        assert abs(result.enthalpy_j_mol) < 0.1
