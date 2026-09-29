"""Tests for sidekick.utils.unit_constants (Issues #1949, #1744)."""

from __future__ import annotations

from sidekick.utils.unit_constants import (
    ATMOSPHERE_TO_PASCAL,
    CELSIUS_OFFSET,
    FOOT_TO_METER,
    HOUR_TO_SECOND,
    INCH_TO_METER,
    KILOWATT_TO_WATT,
    LB_TO_KG,
    METER_TO_METER,
    MINUTE_TO_SECOND,
    MW_HYDROGEN,
    MW_NITROGEN,
    MW_OXYGEN,
    MW_WATER_VAPOR,
    PSI_TO_PASCAL,
    STANDARD_GRAVITY,
    STP_TEMPERATURE_K,
    UNIT_ALIASES,
    WATT_HOUR_TO_JOULE,
)


class TestLengthConversions:
    def test_meter_to_meter_is_one(self) -> None:
        assert METER_TO_METER == 1.0

    def test_foot_to_meter_approx(self) -> None:
        assert abs(FOOT_TO_METER - 0.3048) < 1e-6

    def test_inch_to_meter_approx(self) -> None:
        assert abs(INCH_TO_METER - 0.0254) < 1e-6

    def test_twelve_inches_equals_one_foot(self) -> None:
        assert abs(12.0 * INCH_TO_METER - FOOT_TO_METER) < 1e-10


class TestPressureConversions:
    def test_atmosphere_to_pascal(self) -> None:
        assert abs(ATMOSPHERE_TO_PASCAL - 101325.0) < 1.0

    def test_psi_to_pascal_approx(self) -> None:
        assert abs(PSI_TO_PASCAL - 6894.757) < 1.0


class TestTimeConversions:
    def test_minute_to_second(self) -> None:
        assert MINUTE_TO_SECOND == 60.0

    def test_hour_to_second(self) -> None:
        assert HOUR_TO_SECOND == 3600.0

    def test_sixty_minutes_equals_one_hour(self) -> None:
        assert abs(60.0 * MINUTE_TO_SECOND - HOUR_TO_SECOND) < 1e-10


class TestMassConversions:
    def test_lb_to_kg_approx(self) -> None:
        assert abs(LB_TO_KG - 0.45359237) < 1e-6


class TestEnergyConversions:
    def test_watt_hour_to_joule(self) -> None:
        assert abs(WATT_HOUR_TO_JOULE - 3600.0) < 1e-6

    def test_kilowatt_to_watt(self) -> None:
        assert KILOWATT_TO_WATT == 1000.0


class TestPhysicalConstants:
    def test_standard_gravity(self) -> None:
        assert abs(STANDARD_GRAVITY - 9.80665) < 1e-5

    def test_celsius_offset(self) -> None:
        assert abs(CELSIUS_OFFSET - 273.15) < 1e-6

    def test_stp_temperature_kelvin(self) -> None:
        # STP is 0°C = 273.15 K
        assert abs(STP_TEMPERATURE_K - 273.15) < 1e-6


class TestMolecularWeights:
    def test_water_vapor_mw(self) -> None:
        assert abs(MW_WATER_VAPOR - 18.015) < 0.01

    def test_hydrogen_mw(self) -> None:
        assert abs(MW_HYDROGEN - 2.016) < 0.01

    def test_nitrogen_mw(self) -> None:
        assert abs(MW_NITROGEN - 28.014) < 0.01

    def test_oxygen_mw_approx_32(self) -> None:
        assert abs(MW_OXYGEN - 31.999) < 0.01


class TestUnitAliases:
    def test_unit_constants_is_dict(self) -> None:
        assert isinstance(UNIT_ALIASES, dict)

    def test_unit_constants_nonempty(self) -> None:
        assert len(UNIT_ALIASES) > 0

    def test_meter_key_exists(self) -> None:
        assert "m" in UNIT_ALIASES

    def test_values_are_lists(self) -> None:
        for key, val in UNIT_ALIASES.items():
            assert isinstance(val, list), f"Expected list for key {key!r}"
