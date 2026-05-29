import pytest
from sidekick.calculators.thermo.thermo_properties import (
    ThermoPropertiesCalculator,
    ThermoResult,
)


def test_thermo_properties_calculator() -> None:
    calc = ThermoPropertiesCalculator()

    # Standard condition test (25C, 1 atm, mostly N2)
    result = calc.calculate(
        temperature_c=25.0,
        pressure_kpa=101.325,
        composition={"N2": 0.79, "O2": 0.21},
    )

    assert isinstance(result, ThermoResult)
    assert result.temperature_k == pytest.approx(298.15)
    assert result.pressure_pa == pytest.approx(101325.0)
    assert result.molecular_weight_g_mol > 28.0 and result.molecular_weight_g_mol < 32.0
    assert result.density_kg_m3 > 1.0 and result.density_kg_m3 < 1.3
    assert result.cp_j_molk > 29.0

    # Test zero total composition handling
    zero_comp_result = calc.calculate(
        temperature_c=0.0,
        pressure_kpa=100.0,
        composition={"N2": 0.0},
    )
    # The code normalizes total to 1.0 if it's <= 0, keeping fractions 0/1.0 = 0.
    # The default MW for pure unknown is 0.0 because it's missing, but it uses sum().
    # Without any valid species, MW = 0.0. Let's verify no crash happens.
    assert zero_comp_result.molecular_weight_g_mol == 0.0

    # Test unknown species
    unknown_result = calc.calculate(
        temperature_c=100.0,
        pressure_kpa=200.0,
        composition={"UnknownGas": 1.0},
    )
    # The code defaults completely missing specs to MW=28.0, Cp=29.0
    assert unknown_result.molecular_weight_g_mol == 28.0
    assert unknown_result.cp_j_molk == 29.0
    assert unknown_result.gamma > 1.0
