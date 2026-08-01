import pytest
from sidekick.process_calculators.pressure_drop_calculator.utils.gas_properties import (
    GAS_DATABASE,
    calculate_compressibility_factor,
    calculate_gas_properties,
    calculate_heat_capacity_ratio,
    calculate_ideal_gas_cp,
    calculate_ideal_gas_density,
    calculate_mixture_cp,
    calculate_mixture_molecular_weight,
    calculate_mixture_viscosity_simple,
    calculate_mixture_viscosity_wilke,
    calculate_pure_gas_viscosity_lucas,
    calculate_pure_gas_viscosity_sutherland,
    calculate_real_gas_density,
    calculate_speed_of_sound,
)


def test_unknown_only_compressibility_falls_back_to_ideal_gas() -> None:
    assert (
        calculate_compressibility_factor(
            {"UnobtainiumGas": 1.0},
            temperature=350.0,
            pressure=250_000.0,
        )
        == 1.0
    )


def test_known_gas_property_helpers_return_physical_values() -> None:
    composition = {"N2": 0.79, "O2": 0.21}

    molecular_weight = calculate_mixture_molecular_weight(composition)
    cp = calculate_mixture_cp(composition, temperature=350.0)
    gamma = calculate_heat_capacity_ratio(composition, temperature=350.0)
    ideal_density = calculate_ideal_gas_density(
        molecular_weight,
        temperature=350.0,
        pressure=101_325.0,
    )
    z_factor = calculate_compressibility_factor(
        composition,
        temperature=350.0,
        pressure=101_325.0,
    )
    real_density = calculate_real_gas_density(
        molecular_weight,
        temperature=350.0,
        pressure=101_325.0,
        compressibility=z_factor,
    )
    speed_of_sound = calculate_speed_of_sound(
        composition,
        temperature=350.0,
        molecular_weight=molecular_weight,
    )

    assert molecular_weight == pytest.approx(28.85, rel=0.01)
    assert cp > 0.0
    assert 1.0 < gamma < 1.7
    assert ideal_density > 0.0
    assert real_density > 0.0
    assert 0.1 <= z_factor <= 1.5
    assert speed_of_sound > 0.0


def test_pure_and_mixture_viscosity_helpers_return_physical_values() -> None:
    sutherland_viscosity = calculate_pure_gas_viscosity_sutherland(
        temperature=350.0,
    )
    lucas_viscosity = calculate_pure_gas_viscosity_lucas(
        temperature=450.0,
        pressure=500_000.0,
        props=GAS_DATABASE["H2O"],
    )
    wilke_viscosity = calculate_mixture_viscosity_wilke(
        {"N2": 0.5, "O2": 0.5},
        temperature=350.0,
        pressure=101_325.0,
    )
    simple_viscosity = calculate_mixture_viscosity_simple(
        {"N2": 0.5, "O2": 0.5},
        temperature=350.0,
    )

    assert sutherland_viscosity > 0.0
    assert lucas_viscosity > 0.0
    assert wilke_viscosity > 0.0
    assert simple_viscosity > 0.0


def test_complete_property_calculation_returns_expected_keys() -> None:
    properties = calculate_gas_properties(
        {"H2": 0.25, "CO": 0.35, "CO2": 0.15, "N2": 0.25},
        temperature=700.0,
        pressure=500_000.0,
        use_compressibility=True,
    )

    assert set(properties) == {
        "molecular_weight",
        "density",
        "viscosity",
        "compressibility_factor",
        "heat_capacity_ratio",
        "speed_of_sound",
        "cp",
    }
    assert all(value > 0.0 for value in properties.values())


def test_unknown_ideal_gas_cp_uses_air_fallback() -> None:
    assert calculate_ideal_gas_cp("UnobtainiumGas", temperature=350.0) == pytest.approx(
        calculate_ideal_gas_cp("Air", temperature=350.0)
    )


def test_wilke_viscosity_rejects_unknown_species() -> None:
    with pytest.raises(ValueError, match="Unknown gas species: UnobtainiumGas"):
        calculate_mixture_viscosity_wilke(
            {"N2": 0.5, "UnobtainiumGas": 0.5},
            temperature=350.0,
            pressure=250_000.0,
        )


def test_simple_viscosity_rejects_species_without_sutherland_data() -> None:
    with pytest.raises(ValueError, match="Unknown gas species: H2O"):
        calculate_mixture_viscosity_simple({"H2O": 1.0}, temperature=350.0)


def test_complete_property_calculation_rejects_unknown_species() -> None:
    with pytest.raises(ValueError, match="Unknown gas species: UnobtainiumGas"):
        calculate_gas_properties(
            {"UnobtainiumGas": 1.0},
            temperature=350.0,
            pressure=250_000.0,
        )
