"""Reference data for unit conversion tables and gas properties.

These tables are deliberately defined as immutable module-level mappings so that conversion
logic can operate on declarative data rather than constructing dictionaries at runtime.
Keeping the factors here also makes it easy to document provenance for physical constants
and provides a single source of truth that can be reused in documentation or validation
tooling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from ...utils.unit_constants import (
    ACRE_TO_SQ_METER,
    ANGSTROM_TO_METER,
    ATMOSPHERE_TO_PASCAL,
    BAR_TO_PASCAL,
    BTU_PER_FOOT_HOUR_FAHRENHEIT_TO_W_PER_M_K,
    BTU_PER_HOUR_TO_WATT,
    BTU_PER_POUND_FAHRENHEIT_TO_J_PER_KG_K,
    BTU_PER_SQ_FOOT_HOUR_FAHRENHEIT_TO_W_PER_M2_K,
    BTU_TO_JOULE,
    CAL_PER_CM_SECOND_CELSIUS_TO_W_PER_M_K,
    CAL_PER_GRAM_CELSIUS_TO_J_PER_KG_K,
    CALORIE_PER_SECOND_TO_WATT,
    CALORIE_TO_JOULE,
    CENTIMETER_TO_METER,
    CENTIPOISE_TO_PASCAL_SECOND,
    CENTISTOKE_TO_SQ_METER_PER_SECOND,
    CM_H2O_TO_PASCAL,
    CU_CENTIMETER_TO_CU_METER,
    CU_FOOT_TO_CU_METER,
    CU_INCH_TO_CU_METER,
    CU_MILLIMETER_TO_CU_METER,
    DAY_TO_SECOND,
    DENSITY_STP_AIR,
    DENSITY_STP_CO,
    DENSITY_STP_CO2,
    DENSITY_STP_HYDROGEN,
    DENSITY_STP_METHANE,
    DENSITY_STP_NITROGEN,
    DENSITY_STP_OXYGEN,
    DENSITY_STP_WATER_VAPOR,
    ELECTRON_VOLT_TO_JOULE,
    ERG_TO_JOULE,
    FOOT_H2O_TO_PASCAL,
    FOOT_POUND_PER_SECOND_TO_WATT,
    FOOT_TO_METER,
    GIGAJOULE_TO_JOULE,
    GIGAPASCAL_TO_PASCAL,
    GIGAWATT_TO_WATT,
    GRAIN_TO_KILOGRAM,
    GRAM_PER_CU_CM_TO_KG_PER_CU_METER,
    GRAM_PER_LITER_TO_KG_PER_CU_METER,
    GRAM_PER_SECOND_TO_KG_PER_SECOND,
    GRAM_TO_KILOGRAM,
    HECTARE_TO_SQ_METER,
    HORSEPOWER_TO_WATT,
    HOUR_TO_SECOND,
    IMPERIAL_GALLON_TO_CU_METER,
    INCH_H2O_TO_PASCAL,
    INCH_HG_TO_PASCAL,
    INCH_TO_METER,
    JOULE_PER_KG_KELVIN,
    KCAL_PER_HOUR_TO_WATT,
    KG_PER_HOUR_TO_KG_PER_SECOND,
    KG_PER_MINUTE_TO_KG_PER_SECOND,
    KILOCALORIE_TO_JOULE,
    KILOJOULE_TO_JOULE,
    KILOMETER_TO_METER,
    KILOPASCAL_TO_PASCAL,
    KILOWATT_HOUR_TO_JOULE,
    KILOWATT_TO_WATT,
    LITER_TO_CU_METER,
    LONG_TON_TO_KILOGRAM,
    MEGAJOULE_TO_JOULE,
    MEGAPASCAL_TO_PASCAL,
    MEGAWATT_HOUR_TO_JOULE,
    MEGAWATT_TO_WATT,
    METRIC_HORSEPOWER_TO_WATT,
    METRIC_TON_TO_KILOGRAM,
    MICROMETER_TO_METER,
    MIL_TO_METER,
    MILE_TO_METER,
    MILLIBAR_TO_PASCAL,
    MILLIGRAM_TO_KILOGRAM,
    MILLILITER_TO_CU_METER,
    MILLIMETER_TO_METER,
    MINUTE_TO_SECOND,
    MMBTU_PER_HOUR_TO_WATT,
    MMHG_TO_PASCAL,
    NANOMETER_TO_METER,
    OUNCE_TO_KILOGRAM,
    PASCAL_SECOND_TO_PASCAL_SECOND,
    POISE_TO_PASCAL_SECOND,
    POUND_PER_CU_FOOT_TO_KG_PER_CU_METER,
    POUND_PER_FOOT_SECOND_TO_PASCAL_SECOND,
    POUND_PER_GALLON_TO_KG_PER_CU_METER,
    POUND_PER_HOUR_TO_KG_PER_SECOND,
    POUND_PER_MINUTE_TO_KG_PER_SECOND,
    POUND_PER_SECOND_TO_KG_PER_SECOND,
    POUND_TO_KILOGRAM,
    PSI_TO_PASCAL,
    SCFM_60F_TEMPERATURE_K,
    SCFM_70F_TEMPERATURE_K,
    SCFM_PRESSURE_PA,
    SHORT_TON_TO_KILOGRAM,
    SLUG_TO_KILOGRAM,
    SQ_CENTIMETER_TO_SQ_METER,
    SQ_FOOT_TO_SQ_METER,
    SQ_INCH_TO_SQ_METER,
    SQ_KILOMETER_TO_SQ_METER,
    SQ_MILLIMETER_TO_SQ_METER,
    SQ_YARD_TO_SQ_METER,
    STOKE_TO_SQ_METER_PER_SECOND,
    STP_OLD_PRESSURE_PA,
    STP_PRESSURE_PA,
    STP_TEMPERATURE_K,
    THERM_TO_JOULE,
    TORR_TO_PASCAL,
    US_BARREL_TO_CU_METER,
    US_FLUID_OUNCE_TO_CU_METER,
    US_GALLON_TO_CU_METER,
    US_PINT_TO_CU_METER,
    US_QUART_TO_CU_METER,
    WATT_HOUR_TO_JOULE,
    YARD_TO_METER,
)


@dataclass(frozen=True)
class GasProperties:
    """Physical properties for supported gases with source metadata."""

    name: str
    molecular_weight: float
    density_stp: float
    specific_heat_ratio: float
    critical_temp: float
    critical_pressure: float
    source: str


class StandardCondition(Enum):
    """Standardized gas conditions with explicit documentation."""

    # STP (IUPAC 1982+): 0°C, 100 kPa (1 bar)
    STP = (STP_TEMPERATURE_K, STP_PRESSURE_PA, "STP (0°C, 100 kPa)")

    # Old STP (Pre-1982): 0°C, 101.325 kPa (1 atm)
    # Use validation if dealing with legacy data before 1982.
    STP_OLD = (STP_TEMPERATURE_K, STP_OLD_PRESSURE_PA, "Old STP (0°C, 101.325 kPa)")

    # SCFM: Typically 60°F, 1 atm
    SCFM_60F = (SCFM_60F_TEMPERATURE_K, SCFM_PRESSURE_PA, "SCFM at 60°F, 14.696 psia")
    SCFM_70F = (SCFM_70F_TEMPERATURE_K, SCFM_PRESSURE_PA, "SCFM at 70°F, 14.696 psia")

    # NTP: Normal Temperature and Pressure (20°C, 1 atm)
    NTP = (293.15, ATMOSPHERE_TO_PASCAL, "NTP (20°C, 101.325 kPa)")

    # ISO 13443: 15°C, 1 atm (101.325 kPa)
    ISO_GAS = (288.15, ATMOSPHERE_TO_PASCAL, "ISO 13443 (15°C, 101.325 kPa)")

    # SATP: Standard Ambient Temperature and Pressure (25°C, 1 bar)
    SATP = (298.15, 100000.0, "SATP (25°C, 1 bar)")

    # SI Standard: Explicitly 0°C, 1 bar (Same as current STP)
    STP_SI = (STP_TEMPERATURE_K, 100000.0, "SI Standard (0°C, 1 bar)")


GAS_DATABASE: Mapping[str, GasProperties] = MappingProxyType(
    {
        "air": GasProperties(
            "Air",
            28.97,
            DENSITY_STP_AIR,
            1.400,
            132.5,
            3_771_000.0,
            source="NASA Glenn thermodynamic data, Dry Air mixture",
        ),
        "nitrogen": GasProperties(
            "Nitrogen",
            28.014,
            DENSITY_STP_NITROGEN,
            1.400,
            126.2,
            3_394_000.0,
            source="NIST Chemistry WebBook, Nitrogen properties",
        ),
        "oxygen": GasProperties(
            "Oxygen",
            31.999,
            DENSITY_STP_OXYGEN,
            1.395,
            154.6,
            5_043_000.0,
            source="NIST Chemistry WebBook, Oxygen properties",
        ),
        "hydrogen": GasProperties(
            "Hydrogen",
            2.016,
            DENSITY_STP_HYDROGEN,
            1.405,
            33.2,
            1_296_000.0,
            source="NIST Chemistry WebBook, Hydrogen properties",
        ),
        "methane": GasProperties(
            "Methane",
            16.043,
            DENSITY_STP_METHANE,
            1.321,
            190.6,
            4_600_000.0,
            source="NIST Chemistry WebBook, Methane properties",
        ),
        "co": GasProperties(
            "Carbon Monoxide",
            28.01,
            DENSITY_STP_CO,
            1.400,
            132.9,
            3_494_000.0,
            source="NIST Chemistry WebBook, Carbon Monoxide properties",
        ),
        "co2": GasProperties(
            "Carbon Dioxide",
            44.01,
            DENSITY_STP_CO2,
            1.289,
            304.1,
            7_377_000.0,
            source="NIST Chemistry WebBook, Carbon Dioxide properties",
        ),
        "h2o": GasProperties(
            "Water Vapor",
            18.015,
            DENSITY_STP_WATER_VAPOR,
            1.330,
            647.1,
            22_064_000.0,
            source="IAPWS IF-97 steam tables",
        ),
        "syngas_typical": GasProperties(
            "Syngas",
            20.0,
            0.893,
            1.35,
            150.0,
            4_000_000.0,
            source="Representative biomass-derived syngas (project baseline)",
        ),
    }
)

LENGTH_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "m": 1.0,
        "cm": CENTIMETER_TO_METER,
        "mm": MILLIMETER_TO_METER,
        "um": MICROMETER_TO_METER,
        "nm": NANOMETER_TO_METER,
        "Å": ANGSTROM_TO_METER,
        "mil": MIL_TO_METER,
        "km": KILOMETER_TO_METER,
        "in": INCH_TO_METER,
        "ft": FOOT_TO_METER,
        "yd": YARD_TO_METER,
        "mi": MILE_TO_METER,
    }
)

VOLUME_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "m3": 1.0,
        "L": LITER_TO_CU_METER,
        "mL": MILLILITER_TO_CU_METER,
        "cm3": CU_CENTIMETER_TO_CU_METER,
        "mm3": CU_MILLIMETER_TO_CU_METER,
        "ft3": CU_FOOT_TO_CU_METER,
        "in3": CU_INCH_TO_CU_METER,
        "gal": US_GALLON_TO_CU_METER,
        "imp_gal": IMPERIAL_GALLON_TO_CU_METER,
        "qt": US_QUART_TO_CU_METER,
        "pt": US_PINT_TO_CU_METER,
        "fl_oz": US_FLUID_OUNCE_TO_CU_METER,
        "bbl": US_BARREL_TO_CU_METER,
    }
)

MASS_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "kg": 1.0,
        "g": GRAM_TO_KILOGRAM,
        "mg": MILLIGRAM_TO_KILOGRAM,
        "µg": 1.0e-9,
        "tonne": METRIC_TON_TO_KILOGRAM,
        "lb": POUND_TO_KILOGRAM,
        "oz": OUNCE_TO_KILOGRAM,
        "ton": SHORT_TON_TO_KILOGRAM,
        "long_ton": LONG_TON_TO_KILOGRAM,
        "slug": SLUG_TO_KILOGRAM,
        "grain": GRAIN_TO_KILOGRAM,
    }
)

PRESSURE_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "Pa": 1.0,
        "kPa": KILOPASCAL_TO_PASCAL,
        "MPa": MEGAPASCAL_TO_PASCAL,
        "GPa": GIGAPASCAL_TO_PASCAL,
        "bar": BAR_TO_PASCAL,
        "atm": ATMOSPHERE_TO_PASCAL,
        "mbar": MILLIBAR_TO_PASCAL,
        "psi": PSI_TO_PASCAL,
        "torr": TORR_TO_PASCAL,
        "mmHg": MMHG_TO_PASCAL,
        "inHg": INCH_HG_TO_PASCAL,
        "inH2O": INCH_H2O_TO_PASCAL,
        "ftH2O": FOOT_H2O_TO_PASCAL,
        "cmH2O": CM_H2O_TO_PASCAL,
    }
)

ENERGY_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "J": 1.0,
        "kJ": KILOJOULE_TO_JOULE,
        "MJ": MEGAJOULE_TO_JOULE,
        "GJ": GIGAJOULE_TO_JOULE,
        "BTU": BTU_TO_JOULE,
        "cal": CALORIE_TO_JOULE,
        "kcal": KILOCALORIE_TO_JOULE,
        "Wh": WATT_HOUR_TO_JOULE,
        "kWh": KILOWATT_HOUR_TO_JOULE,
        "MWh": MEGAWATT_HOUR_TO_JOULE,
        "therm": THERM_TO_JOULE,
        "erg": ERG_TO_JOULE,
        "eV": ELECTRON_VOLT_TO_JOULE,
    }
)

POWER_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "W": 1.0,
        "kW": KILOWATT_TO_WATT,
        "MW": MEGAWATT_TO_WATT,
        "GW": GIGAWATT_TO_WATT,
        "hp": HORSEPOWER_TO_WATT,
        "metric_hp": METRIC_HORSEPOWER_TO_WATT,
        "BTU/hr": BTU_PER_HOUR_TO_WATT,
        "cal/s": CALORIE_PER_SECOND_TO_WATT,
        "kcal/hr": KCAL_PER_HOUR_TO_WATT,
        "MMBTU/hr": MMBTU_PER_HOUR_TO_WATT,
        "ft·lbf/s": FOOT_POUND_PER_SECOND_TO_WATT,
    }
)

MASS_FLOW_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "kg/s": 1.0,
        "kg/min": KG_PER_MINUTE_TO_KG_PER_SECOND,
        "kg/hr": KG_PER_HOUR_TO_KG_PER_SECOND,
        "kg/day": 1.0 / DAY_TO_SECOND,
        "g/s": GRAM_PER_SECOND_TO_KG_PER_SECOND,
        "g/min": GRAM_PER_SECOND_TO_KG_PER_SECOND / 60.0,
        "g/hr": GRAM_PER_SECOND_TO_KG_PER_SECOND / 3600.0,
        "g/day": GRAM_PER_SECOND_TO_KG_PER_SECOND / DAY_TO_SECOND,
        "lb/hr": POUND_PER_HOUR_TO_KG_PER_SECOND,
        "lb/min": POUND_PER_MINUTE_TO_KG_PER_SECOND,
        "lb/s": POUND_PER_SECOND_TO_KG_PER_SECOND,
        "lb/day": POUND_TO_KILOGRAM / DAY_TO_SECOND,
        "tonne/hr": METRIC_TON_TO_KILOGRAM / 3600.0,
        "tonne/day": METRIC_TON_TO_KILOGRAM / DAY_TO_SECOND,
        "ton/hr": SHORT_TON_TO_KILOGRAM / 3600.0,
        "ton/day": SHORT_TON_TO_KILOGRAM / DAY_TO_SECOND,
    }
)

AREA_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "m2": 1.0,
        "cm2": SQ_CENTIMETER_TO_SQ_METER,
        "mm2": SQ_MILLIMETER_TO_SQ_METER,
        "km2": SQ_KILOMETER_TO_SQ_METER,
        "in2": SQ_INCH_TO_SQ_METER,
        "ft2": SQ_FOOT_TO_SQ_METER,
        "yd2": SQ_YARD_TO_SQ_METER,
        "acre": ACRE_TO_SQ_METER,
        "hectare": HECTARE_TO_SQ_METER,
    }
)

TIME_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "s": 1.0,
        "min": MINUTE_TO_SECOND,
        "hr": HOUR_TO_SECOND,
        "day": DAY_TO_SECOND,
    }
)

VOLUMETRIC_FLOW_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "m3/s": 1.0,
        "m3/min": 1.0 / MINUTE_TO_SECOND,
        "m3/hr": 1.0 / HOUR_TO_SECOND,
        "m3/day": 1.0 / DAY_TO_SECOND,
        "L/s": LITER_TO_CU_METER,
        "L/min": LITER_TO_CU_METER / MINUTE_TO_SECOND,
        "L/hr": LITER_TO_CU_METER / HOUR_TO_SECOND,
        "L/day": LITER_TO_CU_METER / DAY_TO_SECOND,
        "ft3/s": CU_FOOT_TO_CU_METER,
        "ft3/min": CU_FOOT_TO_CU_METER / MINUTE_TO_SECOND,
        "ft3/hr": CU_FOOT_TO_CU_METER / HOUR_TO_SECOND,
        "ft3/day": CU_FOOT_TO_CU_METER / DAY_TO_SECOND,
        "gal/min": US_GALLON_TO_CU_METER / MINUTE_TO_SECOND,
        "gal/hr": US_GALLON_TO_CU_METER / HOUR_TO_SECOND,
        "gal/day": US_GALLON_TO_CU_METER / DAY_TO_SECOND,
        "imp_gal/min": IMPERIAL_GALLON_TO_CU_METER / MINUTE_TO_SECOND,
        "imp_gal/hr": IMPERIAL_GALLON_TO_CU_METER / HOUR_TO_SECOND,
        "imp_gal/day": IMPERIAL_GALLON_TO_CU_METER / DAY_TO_SECOND,
        "gpm": US_GALLON_TO_CU_METER / MINUTE_TO_SECOND,
        "gph": US_GALLON_TO_CU_METER / HOUR_TO_SECOND,
        "bbl/day": US_BARREL_TO_CU_METER / DAY_TO_SECOND,
        "bbl/hr": US_BARREL_TO_CU_METER / HOUR_TO_SECOND,
    }
)

DENSITY_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "kg/m3": 1.0,
        "g/L": GRAM_PER_LITER_TO_KG_PER_CU_METER,
        "g/cm3": GRAM_PER_CU_CM_TO_KG_PER_CU_METER,
        "lb/ft3": POUND_PER_CU_FOOT_TO_KG_PER_CU_METER,
        "lb/gal": POUND_PER_GALLON_TO_KG_PER_CU_METER,
        "kg/L": 1000.0,
    }
)

DYNAMIC_VISCOSITY_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "Pa·s": PASCAL_SECOND_TO_PASCAL_SECOND,
        "mPa·s": CENTIPOISE_TO_PASCAL_SECOND,
        "cP": CENTIPOISE_TO_PASCAL_SECOND,
        "P": POISE_TO_PASCAL_SECOND,
        "lb/ft·s": POUND_PER_FOOT_SECOND_TO_PASCAL_SECOND,
    }
)

KINEMATIC_VISCOSITY_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "m2/s": 1.0,
        "cSt": CENTISTOKE_TO_SQ_METER_PER_SECOND,
        "cm2/s": STOKE_TO_SQ_METER_PER_SECOND,
        "St": STOKE_TO_SQ_METER_PER_SECOND,
        "ft2/s": SQ_FOOT_TO_SQ_METER,
    }
)

THERMAL_CONDUCTIVITY_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "W/m·K": 1.0,
        "BTU/(ft·hr·°F)": BTU_PER_FOOT_HOUR_FAHRENHEIT_TO_W_PER_M_K,
        "cal/(cm·s·°C)": CAL_PER_CM_SECOND_CELSIUS_TO_W_PER_M_K,
    }
)

HEAT_TRANSFER_COEFF_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "W/m2·K": 1.0,
        "BTU/(ft2·hr·°F)": BTU_PER_SQ_FOOT_HOUR_FAHRENHEIT_TO_W_PER_M2_K,
    }
)

SPECIFIC_HEAT_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "J/kg·K": JOULE_PER_KG_KELVIN,
        "kJ/kg·K": KILOJOULE_TO_JOULE,
        "BTU/lb·°F": BTU_PER_POUND_FAHRENHEIT_TO_J_PER_KG_K,
        "cal/g·°C": CAL_PER_GRAM_CELSIUS_TO_J_PER_KG_K,
    }
)

SPECIFIC_ENERGY_FACTORS: Mapping[str, float] = MappingProxyType(
    {
        "J/kg": 1.0,
        "kJ/kg": KILOJOULE_TO_JOULE,
        "MJ/kg": MEGAJOULE_TO_JOULE,
        "GJ/kg": GIGAJOULE_TO_JOULE,
        "BTU/lb": BTU_TO_JOULE / POUND_TO_KILOGRAM,
        "cal/g": CALORIE_TO_JOULE / GRAM_TO_KILOGRAM,
        "kcal/kg": KILOCALORIE_TO_JOULE,
    }
)

CATEGORY_TABLES: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "length": LENGTH_FACTORS,
        "volume": VOLUME_FACTORS,
        "mass": MASS_FACTORS,
        "pressure": PRESSURE_FACTORS,
        "energy": ENERGY_FACTORS,
        "power": POWER_FACTORS,
        "mass_flow": MASS_FLOW_FACTORS,
        "area": AREA_FACTORS,
        "time": TIME_FACTORS,
        "volumetric_flow": VOLUMETRIC_FLOW_FACTORS,
        "density": DENSITY_FACTORS,
        "dynamic_viscosity": DYNAMIC_VISCOSITY_FACTORS,
        "kinematic_viscosity": KINEMATIC_VISCOSITY_FACTORS,
        "thermal_conductivity": THERMAL_CONDUCTIVITY_FACTORS,
        "heat_transfer": HEAT_TRANSFER_COEFF_FACTORS,
        "specific_heat": SPECIFIC_HEAT_FACTORS,
        "specific_energy": SPECIFIC_ENERGY_FACTORS,
    }
)

HEATING_VALUE_CONVERSIONS: Mapping[str, float | None] = MappingProxyType(
    {
        "mj/kg": 1.0,
        "kj/kg": 0.001,
        "j/kg": 1e-6,
        "cal/g": 0.004184,
        "kcal/kg": 0.004184,
        # Factor based on IT BTU (1055.056 J) / lb (0.45359237 kg) -> J/kg / 1e6 -> MJ/kg
        "btu/lb": 0.002326,
        "mj/nm³": None,
        "mj/nm3": None,
        "btu/scf": None,
        "kwh/kg": 3.6,
        "kwh/nm³": None,
        "kwh/nm3": None,
    }
)

CONCENTRATION_CONVERSIONS: Mapping[str, float | None] = MappingProxyType(
    {
        "mg/nm³": 1.0,
        "mg/nm3": 1.0,
        "g/nm³": 1000.0,
        "g/nm3": 1000.0,
        "µg/nm³": 0.001,
        "ug/nm3": 0.001,
        "mg/m³": None,
        "mg/m3": None,
        "g/m³": None,
        "g/m3": None,
        "gr/scf": 2288.35,
        "ppm_mass": None,
    }
)

PERFORMANCE_UNITS: Mapping[str, list[str]] = MappingProxyType(
    {
        "efficiency": ["%", "fraction"],
        "carbon_conversion": ["%", "fraction"],
        "specific_production": ["nm³/kg", "nm3/kg", "scf/lb"],
    }
)

# Optional alias mapping for unit normalization in conversion service.
# Keys are canonical units; values are accepted aliases for that unit.
UNIT_ALIASES: Mapping[str, list[str]] = MappingProxyType(
    {
        "m": ["meter", "meters", "metre", "metres"],
        "kg": ["kilogram", "kilograms"],
        "s": ["sec", "second", "seconds"],
        "Pa": ["pascal", "pascals"],
        "kPa": ["kilopascal", "kilopascals"],
        "SCFM": ["scf/min"],
        "ACFM": ["acf/min"],
        "Nm3/hr": ["nm^3/hr", "normal_m3_per_hour"],
        "Nm³/hr": ["nm3h"],
        "C": ["degc", "celsius"],
        "F": ["degf", "fahrenheit"],
        "K": ["kelvin"],
    }
)
