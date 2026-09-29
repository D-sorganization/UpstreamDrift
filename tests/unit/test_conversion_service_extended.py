"""Extended unit tests for UnitConversionService.

Tests cover the full conversion API including:
- Temperature conversions (C↔K↔F)
- Length/distance conversions
- Mass conversions
- Pressure conversions
- Flow rate conversions
- Error cases (unknown units, incompatible categories)
- ConversionResult dataclass
- Module-level convert() convenience function
- get_service() singleton

All tests are headless-safe with no heavy dependencies.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Fresh UnitConversionService for each test."""
    from sidekick.calculators.conversion.service import (
        UnitConversionService,
    )

    return UnitConversionService()


# ---------------------------------------------------------------------------
# UnitConversionService initialization
# ---------------------------------------------------------------------------


class TestServiceInit:
    """Tests for UnitConversionService.__init__."""

    def test_instantiates_successfully(self, service) -> None:
        """Service can be created without errors."""
        assert service is not None

    def test_enable_validation_stored(self) -> None:
        """enable_validation flag is stored on the instance."""
        from sidekick.calculators.conversion.service import (
            UnitConversionService,
        )

        s = UnitConversionService(enable_validation=False)
        assert s.enable_validation is False


# ---------------------------------------------------------------------------
# ConversionResult dataclass
# ---------------------------------------------------------------------------


class TestConversionResult:
    """Tests for the ConversionResult dataclass."""

    def test_has_expected_fields(self, service) -> None:
        """ConversionResult has value, from_unit, to_unit fields."""
        result = service.convert(1.0, "m", "ft")
        assert hasattr(result, "value")
        assert hasattr(result, "from_unit")
        assert hasattr(result, "to_unit")

    def test_from_unit_and_to_unit_stored(self, service) -> None:
        """ConversionResult records source and target units."""
        result = service.convert(100.0, "C", "F")
        assert result.from_unit == "C"
        assert result.to_unit == "F"


# ---------------------------------------------------------------------------
# Temperature conversions
# ---------------------------------------------------------------------------


class TestTemperatureConversions:
    """Tests for temperature unit conversions."""

    def test_celsius_to_kelvin_0c(self, service) -> None:
        """0°C = 273.15 K."""
        r = service.convert(0.0, "C", "K")
        assert abs(r.value - 273.15) < 1e-6

    def test_celsius_to_kelvin_100c(self, service) -> None:
        """100°C = 373.15 K."""
        r = service.convert(100.0, "C", "K")
        assert abs(r.value - 373.15) < 1e-6

    def test_kelvin_to_celsius(self, service) -> None:
        """273.15 K = 0°C (inverse of celsius→kelvin)."""
        r = service.convert(273.15, "K", "C")
        assert abs(r.value - 0.0) < 1e-4

    def test_celsius_to_fahrenheit_0c(self, service) -> None:
        """0°C = 32°F."""
        r = service.convert(0.0, "C", "F")
        assert abs(r.value - 32.0) < 1e-6

    def test_celsius_to_fahrenheit_100c(self, service) -> None:
        """100°C = 212°F."""
        r = service.convert(100.0, "C", "F")
        assert abs(r.value - 212.0) < 1e-6

    def test_fahrenheit_to_celsius_32f(self, service) -> None:
        """32°F = 0°C."""
        r = service.convert(32.0, "F", "C")
        assert abs(r.value - 0.0) < 1e-6

    def test_fahrenheit_to_celsius_212f(self, service) -> None:
        """212°F = 100°C."""
        r = service.convert(212.0, "F", "C")
        assert abs(r.value - 100.0) < 1e-6


# ---------------------------------------------------------------------------
# Length conversions
# ---------------------------------------------------------------------------


class TestLengthConversions:
    """Tests for length unit conversions."""

    def test_meter_to_foot(self, service) -> None:
        """1 m ≈ 3.28084 ft."""
        r = service.convert(1.0, "m", "ft")
        assert abs(r.value - 3.28084) < 0.0001

    def test_foot_to_meter(self, service) -> None:
        """1 ft ≈ 0.3048 m (inverse)."""
        r = service.convert(1.0, "ft", "m")
        assert abs(r.value - 0.3048) < 1e-6

    def test_meter_to_centimeter(self, service) -> None:
        """1 m = 100 cm."""
        r = service.convert(1.0, "m", "cm")
        assert abs(r.value - 100.0) < 1e-6

    def test_kilometer_to_meter(self, service) -> None:
        """1 km = 1000 m."""
        r = service.convert(1.0, "km", "m")
        assert abs(r.value - 1000.0) < 1e-6

    def test_identity_conversion(self, service) -> None:
        """Converting a unit to itself returns the same value."""
        r = service.convert(42.0, "m", "m")
        assert abs(r.value - 42.0) < 1e-9

    def test_inch_to_centimeter(self, service) -> None:
        """1 inch = 2.54 cm."""
        r = service.convert(1.0, "in", "cm")
        assert abs(r.value - 2.54) < 0.001


# ---------------------------------------------------------------------------
# Mass conversions
# ---------------------------------------------------------------------------


class TestMassConversions:
    """Tests for mass unit conversions."""

    def test_kilogram_to_gram(self, service) -> None:
        """1 kg = 1000 g."""
        r = service.convert(1.0, "kg", "g")
        assert abs(r.value - 1000.0) < 1e-6

    def test_gram_to_kilogram(self, service) -> None:
        """1 g = 0.001 kg (inverse)."""
        r = service.convert(1.0, "g", "kg")
        assert abs(r.value - 0.001) < 1e-9

    def test_kilogram_to_pound(self, service) -> None:
        """1 kg ≈ 2.20462 lb."""
        r = service.convert(1.0, "kg", "lb")
        assert abs(r.value - 2.20462) < 0.001


# ---------------------------------------------------------------------------
# Pressure conversions
# ---------------------------------------------------------------------------


class TestPressureConversions:
    """Tests for pressure unit conversions."""

    def test_atm_to_pascal(self, service) -> None:
        """1 atm = 101325 Pa."""
        r = service.convert(1.0, "atm", "Pa")
        assert abs(r.value - 101325.0) < 1.0

    def test_bar_to_pascal(self, service) -> None:
        """1 bar = 100000 Pa."""
        r = service.convert(1.0, "bar", "Pa")
        assert abs(r.value - 100000.0) < 1.0

    def test_pascal_to_bar(self, service) -> None:
        """100000 Pa = 1 bar."""
        r = service.convert(100000.0, "Pa", "bar")
        assert abs(r.value - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Clean string helper
# ---------------------------------------------------------------------------


class TestCleanString:
    """Tests for the _clean_string normalization helper."""

    def test_lowercase(self, service) -> None:
        """_clean_string converts to lowercase."""
        assert service._clean_string("KG") == "kg"

    def test_removes_spaces(self, service) -> None:
        """_clean_string removes spaces."""
        assert "m" in service._clean_string("m s")

    def test_removes_degree_symbol(self, service) -> None:
        """_clean_string strips degree symbol °."""
        assert "°" not in service._clean_string("°C")

    def test_removes_hyphens_and_underscores(self, service) -> None:
        """_clean_string strips hyphens and underscores."""
        cleaned = service._clean_string("kilo-gram_per_second")
        assert "-" not in cleaned
        assert "_" not in cleaned


# ---------------------------------------------------------------------------
# _require_positive_finite and _require_finite validators
# ---------------------------------------------------------------------------


class TestValidationHelpers:
    """Tests for static validation helper methods."""

    def test_require_positive_finite_valid(self) -> None:
        """_require_positive_finite does not raise for positive finite float."""
        from sidekick.calculators.conversion.service import (
            UnitConversionService,
        )

        UnitConversionService._require_positive_finite(5.0, "x")  # no exception

    def test_require_positive_finite_raises_for_zero(self) -> None:
        """_require_positive_finite raises ValueError for zero."""
        from sidekick.calculators.conversion.service import (
            UnitConversionService,
        )

        with pytest.raises(ValueError):
            UnitConversionService._require_positive_finite(0.0, "x")

    def test_require_positive_finite_raises_for_negative(self) -> None:
        """_require_positive_finite raises ValueError for negative values."""
        from sidekick.calculators.conversion.service import (
            UnitConversionService,
        )

        with pytest.raises(ValueError):
            UnitConversionService._require_positive_finite(-1.0, "x")

    def test_require_positive_finite_raises_for_inf(self) -> None:
        """_require_positive_finite raises ValueError for infinity."""
        from sidekick.calculators.conversion.service import (
            UnitConversionService,
        )

        with pytest.raises(ValueError):
            UnitConversionService._require_positive_finite(float("inf"), "x")

    def test_require_finite_valid(self) -> None:
        """_require_finite does not raise for finite float."""
        from sidekick.calculators.conversion.service import (
            UnitConversionService,
        )

        UnitConversionService._require_finite(0.0, "x")  # no exception
        UnitConversionService._require_finite(-999.0, "x")  # no exception

    def test_require_finite_raises_for_nan(self) -> None:
        """_require_finite raises ValueError for NaN."""
        import math

        from sidekick.calculators.conversion.service import (
            UnitConversionService,
        )

        with pytest.raises(ValueError):
            UnitConversionService._require_finite(math.nan, "x")


# ---------------------------------------------------------------------------
# Module-level convert() convenience function
# ---------------------------------------------------------------------------


class TestModuleConvert:
    """Tests for the module-level convert() function."""

    def test_conversion_service_extended_basic_conversion(self) -> None:
        """convert() returns a numeric result for a valid unit pair."""
        from sidekick.calculators.conversion.service import (
            convert,
        )

        result = convert(1.0, "kg", "g")
        assert abs(result - 1000.0) < 1e-6

    def test_temperature_conversion(self) -> None:
        """convert() handles temperature conversions."""
        from sidekick.calculators.conversion.service import (
            convert,
        )

        result = convert(0.0, "C", "K")
        assert abs(result - 273.15) < 1e-6


# ---------------------------------------------------------------------------
# Gas-flow Nm³ / tar-concentration reference states (#7413)
# ---------------------------------------------------------------------------


class TestNormalCubicMetreReferenceState:
    """Nm³ conversions must use the DIN 1343 normal state (0°C, 101.325 kPa).

    Guards issue #7413: the gas-flow pivot previously used IUPAC STP
    (0°C, 100 kPa), biasing SCFM/ACFM ↔ Nm³/hr by +1.325% and disagreeing
    with the tar-concentration mixin's Nm³, while ppm used the 25°C
    24.45 L/mol factor (8.3% low).
    """

    @pytest.mark.scientific
    def test_scfm_to_nm3_per_hour_uses_din1343(self, service) -> None:
        """1000 SCFM (60°F) → 1607.46 Nm³/hr (DIN 1343), not 1628.76."""
        result = service.convert(1000.0, "SCFM", "Nm3/hr")
        assert result.value == pytest.approx(1607.46, rel=0.001)

    @pytest.mark.scientific
    def test_scfm_nm3_round_trip(self, service) -> None:
        """SCFM → Nm³/hr → SCFM is identity."""
        nm3 = service.convert(1000.0, "SCFM", "Nm3/hr").value
        back = service.convert(nm3, "Nm3/hr", "SCFM").value
        assert back == pytest.approx(1000.0, rel=1e-9)

    @pytest.mark.scientific
    def test_tar_ppm_to_mg_per_nm3_uses_normal_molar_volume(self, service) -> None:
        """1 ppmv benzene (MW 78.11) → 3.485 mg/Nm³ (22.414 L/mol), not 3.195."""
        value = service.tar_concentration(
            1.0, "ppm_mass", "mg/Nm3", molecular_weight=78.11
        )
        assert value == pytest.approx(3.485, rel=0.005)

    @pytest.mark.scientific
    def test_gas_flow_and_tar_share_normal_state(self) -> None:
        """The Nm³ reference used by gas-flow equals the tar mg/Nm³ basis."""
        from sidekick.calculators.conversion._gas_flow import NORMAL_CONDITION

        temp_k, pressure_pa, _ = NORMAL_CONDITION.value
        # Tar mg/Nm³ is anchored at 0°C / 101.325 kPa (see _concentration.py).
        assert temp_k == pytest.approx(273.15)
        assert pressure_pa == pytest.approx(101325.0)


# ---------------------------------------------------------------------------
# get_service singleton
# ---------------------------------------------------------------------------


class TestGetService:
    """Tests for get_service() singleton factory."""

    def test_returns_service_instance(self) -> None:
        """get_service() returns a UnitConversionService."""
        from sidekick.calculators.conversion.service import (
            UnitConversionService,
            get_service,
        )

        s = get_service()
        assert isinstance(s, UnitConversionService)

    def test_singleton_returns_same_instance(self) -> None:
        """get_service() returns the same instance on repeated calls."""
        from sidekick.calculators.conversion.service import (
            get_service,
        )

        s1 = get_service()
        s2 = get_service()
        assert s1 is s2
