"""Tests for sidekick.process_calculators.constants (Issues #1949, #1744)."""

from __future__ import annotations

from sidekick.process_calculators.constants import (
    ATM_PA,
    MOLECULAR_WEIGHTS,
    R_GAS_J_MOL_K,
    celsius_to_kelvin,
    fahrenheit_to_kelvin,
    get_molecular_weight,
    kelvin_to_celsius,
    kelvin_to_fahrenheit,
)

# ---------------------------------------------------------------------------
# Physical constants sanity checks
# ---------------------------------------------------------------------------


class TestConstants:
    def test_r_gas_approx(self) -> None:
        assert abs(R_GAS_J_MOL_K - 8.314) < 0.01

    def test_atm_pa_approx(self) -> None:
        assert abs(ATM_PA - 101325.0) < 1.0

    def test_molecular_weights_nonempty(self) -> None:
        assert len(MOLECULAR_WEIGHTS) > 0

    def test_n2_in_molecular_weights(self) -> None:
        assert "N2" in MOLECULAR_WEIGHTS


# ---------------------------------------------------------------------------
# Temperature conversions
# ---------------------------------------------------------------------------


class TestCelsiusToKelvin:
    def test_zero_celsius(self) -> None:
        assert abs(celsius_to_kelvin(0.0) - 273.15) < 0.001

    def test_one_hundred_celsius(self) -> None:
        assert abs(celsius_to_kelvin(100.0) - 373.15) < 0.001

    def test_negative_celsius(self) -> None:
        assert abs(celsius_to_kelvin(-273.15) - 0.0) < 0.001


class TestKelvinToCelsius:
    def test_process_constants_roundtrip(self) -> None:
        original = 25.0
        assert abs(kelvin_to_celsius(celsius_to_kelvin(original)) - original) < 1e-6

    def test_freezing(self) -> None:
        assert abs(kelvin_to_celsius(273.15) - 0.0) < 0.001


class TestFahrenheitToKelvin:
    def test_freezing_point(self) -> None:
        assert abs(fahrenheit_to_kelvin(32.0) - 273.15) < 0.01

    def test_boiling_point(self) -> None:
        assert abs(fahrenheit_to_kelvin(212.0) - 373.15) < 0.01


class TestKelvinToFahrenheit:
    def test_process_constants_roundtrip(self) -> None:
        original = 350.0
        assert (
            abs(fahrenheit_to_kelvin(kelvin_to_fahrenheit(original)) - original) < 1e-6
        )

    def test_freezing(self) -> None:
        assert abs(kelvin_to_fahrenheit(273.15) - 32.0) < 0.1


# ---------------------------------------------------------------------------
# get_molecular_weight
# ---------------------------------------------------------------------------


class TestGetMolecularWeight:
    def test_n2_known(self) -> None:
        mw = get_molecular_weight("N2")
        assert mw > 0

    def test_h2_known(self) -> None:
        mw = get_molecular_weight("H2")
        assert mw > 0
        assert mw < 0.01  # H2 is light: ~0.002 kg/mol

    def test_process_constants_case_insensitive(self) -> None:
        mw_upper = get_molecular_weight("CO2")
        mw_lower = get_molecular_weight("co2")
        assert abs(mw_upper - mw_lower) < 1e-10

    def test_unknown_species_returns_air_default(self) -> None:
        mw = get_molecular_weight("UNKNOWN_GAS_XYZ")
        # Should return MW_AIR fallback
        assert mw > 0
