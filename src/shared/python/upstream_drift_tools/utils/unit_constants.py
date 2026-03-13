#!/usr/bin/env python3
"""Unit Conversion Constants

NIST-standard conversion factors and physical constants for unit conversions.
Loaded dynamically from JSON configuration for reversibility.
"""

import json
from pathlib import Path
from typing import Final


# Load Configuration dynamically
def _load_config() -> dict:
    """Load constants from external JSON config."""
    # Find assets across development and built application
    base_dir = Path(__file__).resolve().parent
    # Check parent hierarchy up to 6 levels for assets
    for _ in range(7):
        config_path = base_dir / "assets" / "config" / "unit_constants.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        base_dir = base_dir.parent

    return {}


_CONFIG = _load_config()


def _t(key: str, default: float) -> float:
    return float(_CONFIG.get(key, default))


def _t_str(key: str, default: str) -> str:
    return str(_CONFIG.get(key, default))


def _t_int(key: str, default: int) -> int:
    return int(_CONFIG.get(key, default))


def _t_dict_str(key: str, default: dict) -> dict[str, float]:
    d = _CONFIG.get(key, default)
    return {k: float(v) for k, v in d.items()}


def _t_dict_list(key: str, default: dict) -> dict[str, list[str]]:
    d = _CONFIG.get(key, default)
    return {str(k): [str(x) for x in v] for k, v in d.items()}


def _t_dict_tuple(key: str, default: dict) -> dict[str, tuple[float, float]]:
    d = _CONFIG.get(key, default)
    return {str(k): (float(v[0]), float(v[1])) for k, v in d.items()}


ACRE_TO_SQ_METER: Final[float] = _t("ACRE_TO_SQ_METER", 4046.8564224)
ANGSTROM_TO_METER: Final[float] = _t("ANGSTROM_TO_METER", 1e-10)
ATMOSPHERE_TO_PASCAL: Final[float] = _t("ATMOSPHERE_TO_PASCAL", 101325.0)
ATM_TO_KPA: Final[float] = _t("ATM_TO_KPA", 101.325)
AVOGADRO_NUMBER: Final[float] = _t("AVOGADRO_NUMBER", 6.02214076e23)
BAR_TO_KPA: Final[float] = _t("BAR_TO_KPA", 100.0)
BAR_TO_PASCAL: Final[float] = _t("BAR_TO_PASCAL", 100000.0)
BOLTZMANN_CONSTANT: Final[float] = _t("BOLTZMANN_CONSTANT", 1.380649e-23)
BTU_PER_FOOT_HOUR_FAHRENHEIT_TO_W_PER_M_K: Final[float] = _t(
    "BTU_PER_FOOT_HOUR_FAHRENHEIT_TO_W_PER_M_K", 1.7307346664
)
BTU_PER_HOUR_TO_WATT: Final[float] = _t("BTU_PER_HOUR_TO_WATT", 0.2930710701722222)
BTU_PER_LB_TO_J_PER_KG: Final[float] = _t("BTU_PER_LB_TO_J_PER_KG", 2326.0)
BTU_PER_LB_TO_MJ_PER_KG: Final[float] = _t("BTU_PER_LB_TO_MJ_PER_KG", 0.002326)
BTU_PER_POUND_FAHRENHEIT_TO_J_PER_KG_K: Final[float] = _t(
    "BTU_PER_POUND_FAHRENHEIT_TO_J_PER_KG_K", 4186.8
)
BTU_PER_SQ_FOOT_HOUR_FAHRENHEIT_TO_W_PER_M2_K: Final[float] = _t(
    "BTU_PER_SQ_FOOT_HOUR_FAHRENHEIT_TO_W_PER_M2_K", 5.6782633411
)
BTU_TO_JOULE: Final[float] = _t("BTU_TO_JOULE", 1055.05585262)
CALORIE_PER_SECOND_TO_WATT: Final[float] = _t("CALORIE_PER_SECOND_TO_WATT", 4.184)
CALORIE_TO_JOULE: Final[float] = _t("CALORIE_TO_JOULE", 4.184)
CAL_PER_CM_SECOND_CELSIUS_TO_W_PER_M_K: Final[float] = _t(
    "CAL_PER_CM_SECOND_CELSIUS_TO_W_PER_M_K", 418.4
)
CAL_PER_GRAM_CELSIUS_TO_J_PER_KG_K: Final[float] = _t(
    "CAL_PER_GRAM_CELSIUS_TO_J_PER_KG_K", 4186.8
)
CELSIUS_OFFSET: Final[float] = _t("CELSIUS_OFFSET", 273.15)
CENTIMETER_TO_METER: Final[float] = _t("CENTIMETER_TO_METER", 0.01)
CENTIPOISE_TO_PASCAL_SECOND: Final[float] = _t("CENTIPOISE_TO_PASCAL_SECOND", 0.001)
CENTISTOKE_TO_SQ_METER_PER_SECOND: Final[float] = _t(
    "CENTISTOKE_TO_SQ_METER_PER_SECOND", 1e-06
)
CM_H2O_TO_PASCAL: Final[float] = _t("CM_H2O_TO_PASCAL", 98.0665)
CP_WATER_LIQUID: Final[float] = _t("CP_WATER_LIQUID", 4181.3)
CP_WATER_VAPOR: Final[float] = _t("CP_WATER_VAPOR", 1858.9)
CU_CENTIMETER_TO_CU_METER: Final[float] = _t("CU_CENTIMETER_TO_CU_METER", 1e-06)
CU_FOOT_TO_CU_METER: Final[float] = _t("CU_FOOT_TO_CU_METER", 0.028316846592)
CU_INCH_TO_CU_METER: Final[float] = _t("CU_INCH_TO_CU_METER", 1.6387064e-05)
CU_METER_TO_CU_METER: Final[float] = _t("CU_METER_TO_CU_METER", 1.0)
CU_MILLIMETER_TO_CU_METER: Final[float] = _t("CU_MILLIMETER_TO_CU_METER", 1e-09)
DAY_TO_SECOND: Final[float] = _t("DAY_TO_SECOND", 86400.0)
DENSITY_STP_AIR: Final[float] = _t("DENSITY_STP_AIR", 1.2922)
DENSITY_STP_CO: Final[float] = _t("DENSITY_STP_CO", 1.25)
DENSITY_STP_CO2: Final[float] = _t("DENSITY_STP_CO2", 1.9768)
DENSITY_STP_HYDROGEN: Final[float] = _t("DENSITY_STP_HYDROGEN", 0.08988)
DENSITY_STP_METHANE: Final[float] = _t("DENSITY_STP_METHANE", 0.7168)
DENSITY_STP_NITROGEN: Final[float] = _t("DENSITY_STP_NITROGEN", 1.2506)
DENSITY_STP_OXYGEN: Final[float] = _t("DENSITY_STP_OXYGEN", 1.4289)
DENSITY_STP_WATER_VAPOR: Final[float] = _t("DENSITY_STP_WATER_VAPOR", 0.00485)
DENSITY_WATER_STD: Final[float] = _t("DENSITY_WATER_STD", 997.0)
ELECTRON_VOLT_TO_JOULE: Final[float] = _t("ELECTRON_VOLT_TO_JOULE", 1.602176634e-19)
ERG_TO_JOULE: Final[float] = _t("ERG_TO_JOULE", 1e-07)
FOOT_H2O_TO_PASCAL: Final[float] = _t("FOOT_H2O_TO_PASCAL", 2989.07)
FOOT_POUND_PER_SECOND_TO_WATT: Final[float] = _t(
    "FOOT_POUND_PER_SECOND_TO_WATT", 1.3558179483314003
)
FOOT_TO_METER: Final[float] = _t("FOOT_TO_METER", 0.3048)
GIGAJOULE_TO_JOULE: Final[float] = _t("GIGAJOULE_TO_JOULE", 1000000000.0)
GIGAPASCAL_TO_PASCAL: Final[float] = _t("GIGAPASCAL_TO_PASCAL", 1000000000.0)
GIGAWATT_TO_WATT: Final[float] = _t("GIGAWATT_TO_WATT", 1000000000.0)
GRAIN_TO_KILOGRAM: Final[float] = _t("GRAIN_TO_KILOGRAM", 6.479891e-05)
GRAM_PER_CU_CM_TO_KG_PER_CU_METER: Final[float] = _t(
    "GRAM_PER_CU_CM_TO_KG_PER_CU_METER", 1000.0
)
GRAM_PER_LITER_TO_KG_PER_CU_METER: Final[float] = _t(
    "GRAM_PER_LITER_TO_KG_PER_CU_METER", 1.0
)
GRAM_PER_SECOND_TO_KG_PER_SECOND: Final[float] = _t(
    "GRAM_PER_SECOND_TO_KG_PER_SECOND", 0.001
)
GRAM_TO_KILOGRAM: Final[float] = _t("GRAM_TO_KILOGRAM", 0.001)
HECTARE_TO_SQ_METER: Final[float] = _t("HECTARE_TO_SQ_METER", 10000.0)
HORSEPOWER_TO_WATT: Final[float] = _t("HORSEPOWER_TO_WATT", 745.6998715822702)
HOURS_PER_DAY: Final[int] = _t_int("HOURS_PER_DAY", 24)
HOUR_TO_SECOND: Final[float] = _t("HOUR_TO_SECOND", 3600.0)
H_VAP_WATER: Final[float] = _t("H_VAP_WATER", 2257000.0)
IMPERIAL_GALLON_TO_CU_METER: Final[float] = _t(
    "IMPERIAL_GALLON_TO_CU_METER", 0.00454609
)
INCH_H2O_TO_PASCAL: Final[float] = _t("INCH_H2O_TO_PASCAL", 249.082)
INCH_HG_TO_PASCAL: Final[float] = _t("INCH_HG_TO_PASCAL", 3386.389)
INCH_TO_METER: Final[float] = _t("INCH_TO_METER", 0.0254)
JOULE_PER_KG_KELVIN: Final[float] = _t("JOULE_PER_KG_KELVIN", 1.0)
JOULE_TO_JOULE: Final[float] = _t("JOULE_TO_JOULE", 1.0)
KCAL_PER_HOUR_TO_WATT: Final[float] = _t("KCAL_PER_HOUR_TO_WATT", 1.163)
KG_PER_CU_METER_TO_KG_PER_CU_METER: Final[float] = _t(
    "KG_PER_CU_METER_TO_KG_PER_CU_METER", 1.0
)
KG_PER_HOUR_TO_KG_PER_SECOND: Final[float] = _t(
    "KG_PER_HOUR_TO_KG_PER_SECOND", 0.0002777777777777778
)
KG_PER_MINUTE_TO_KG_PER_SECOND: Final[float] = _t(
    "KG_PER_MINUTE_TO_KG_PER_SECOND", 0.016666666666666666
)
KG_PER_SECOND_TO_KG_PER_SECOND: Final[float] = _t("KG_PER_SECOND_TO_KG_PER_SECOND", 1.0)
KG_TO_LB: Final[float] = _t("KG_TO_LB", 2.2046226218487757)
KILOCALORIE_TO_JOULE: Final[float] = _t("KILOCALORIE_TO_JOULE", 4184.0)
KILOGRAM_TO_KILOGRAM: Final[float] = _t("KILOGRAM_TO_KILOGRAM", 1.0)
KILOJOULE_TO_JOULE: Final[float] = _t("KILOJOULE_TO_JOULE", 1000.0)
KILOMETER_TO_METER: Final[float] = _t("KILOMETER_TO_METER", 1000.0)
KILOPASCAL_TO_PASCAL: Final[float] = _t("KILOPASCAL_TO_PASCAL", 1000.0)
KILOWATT_HOUR_TO_JOULE: Final[float] = _t("KILOWATT_HOUR_TO_JOULE", 3600000.0)
KILOWATT_TO_WATT: Final[float] = _t("KILOWATT_TO_WATT", 1000.0)
LB_TO_G: Final[float] = _t("LB_TO_G", 453.59237)
LB_TO_KG: Final[float] = _t("LB_TO_KG", 0.45359237)
LITER_TO_CU_METER: Final[float] = _t("LITER_TO_CU_METER", 0.001)
LONG_TON_TO_KILOGRAM: Final[float] = _t("LONG_TON_TO_KILOGRAM", 1016.0469088)
MEGAJOULE_TO_JOULE: Final[float] = _t("MEGAJOULE_TO_JOULE", 1000000.0)
MEGAPASCAL_TO_PASCAL: Final[float] = _t("MEGAPASCAL_TO_PASCAL", 1000000.0)
MEGAWATT_HOUR_TO_JOULE: Final[float] = _t("MEGAWATT_HOUR_TO_JOULE", 3600000000.0)
MEGAWATT_TO_WATT: Final[float] = _t("MEGAWATT_TO_WATT", 1000000.0)
METER_TO_METER: Final[float] = _t("METER_TO_METER", 1.0)
METRIC_HORSEPOWER_TO_WATT: Final[float] = _t("METRIC_HORSEPOWER_TO_WATT", 735.49875)
METRIC_TON_TO_KILOGRAM: Final[float] = _t("METRIC_TON_TO_KILOGRAM", 1000.0)
MICROMETER_TO_METER: Final[float] = _t("MICROMETER_TO_METER", 1e-06)
MILE_TO_METER: Final[float] = _t("MILE_TO_METER", 1609.344)
MILLIBAR_TO_PASCAL: Final[float] = _t("MILLIBAR_TO_PASCAL", 100.0)
MILLIGRAM_TO_KILOGRAM: Final[float] = _t("MILLIGRAM_TO_KILOGRAM", 1e-06)
MILLILITER_TO_CU_METER: Final[float] = _t("MILLILITER_TO_CU_METER", 1e-06)
MILLIMETER_TO_METER: Final[float] = _t("MILLIMETER_TO_METER", 0.001)
MIL_TO_METER: Final[float] = _t("MIL_TO_METER", 2.54e-05)
MINUTE_TO_SECOND: Final[float] = _t("MINUTE_TO_SECOND", 60.0)
MIN_BASIS_MOLES: Final[float] = _t("MIN_BASIS_MOLES", 1.0)
MMBTU_PER_HOUR_TO_WATT: Final[float] = _t("MMBTU_PER_HOUR_TO_WATT", 293071.0701722222)
MMHG_TO_PASCAL: Final[float] = _t("MMHG_TO_PASCAL", 133.322387415)
MOLAR_VOLUME_STP: Final[float] = _t("MOLAR_VOLUME_STP", 0.02271095)
MOLAR_VOLUME_STP_OLD: Final[float] = _t("MOLAR_VOLUME_STP_OLD", 0.022413969545)
MW_AIR: Final[float] = _t("MW_AIR", 28.9647)
MW_AMMONIA: Final[float] = _t("MW_AMMONIA", 17.0305)
MW_CARBON_DIOXIDE: Final[float] = _t("MW_CARBON_DIOXIDE", 44.0095)
MW_CARBON_MONOXIDE: Final[float] = _t("MW_CARBON_MONOXIDE", 28.0101)
MW_HYDROGEN: Final[float] = _t("MW_HYDROGEN", 2.01588)
MW_HYDROGEN_SULFIDE: Final[float] = _t("MW_HYDROGEN_SULFIDE", 34.0809)
MW_METHANE: Final[float] = _t("MW_METHANE", 16.0425)
MW_NITROGEN: Final[float] = _t("MW_NITROGEN", 28.0134)
MW_OXYGEN: Final[float] = _t("MW_OXYGEN", 31.9988)
MW_WATER_VAPOR: Final[float] = _t("MW_WATER_VAPOR", 18.01528)
NANOMETER_TO_METER: Final[float] = _t("NANOMETER_TO_METER", 1e-09)
NTP_PRESSURE_PA: Final[float] = _t("NTP_PRESSURE_PA", 101325.0)
NTP_TEMPERATURE_K: Final[float] = _t("NTP_TEMPERATURE_K", 293.15)
OUNCE_TO_KILOGRAM: Final[float] = _t("OUNCE_TO_KILOGRAM", 0.028349523125)
PASCAL_SECOND_TO_PASCAL_SECOND: Final[float] = _t("PASCAL_SECOND_TO_PASCAL_SECOND", 1.0)
PASCAL_TO_PASCAL: Final[float] = _t("PASCAL_TO_PASCAL", 1.0)
POISE_TO_PASCAL_SECOND: Final[float] = _t("POISE_TO_PASCAL_SECOND", 0.1)
POUND_PER_CU_FOOT_TO_KG_PER_CU_METER: Final[float] = _t(
    "POUND_PER_CU_FOOT_TO_KG_PER_CU_METER", 16.01846337396
)
POUND_PER_FOOT_SECOND_TO_PASCAL_SECOND: Final[float] = _t(
    "POUND_PER_FOOT_SECOND_TO_PASCAL_SECOND", 1.4881639436
)
POUND_PER_GALLON_TO_KG_PER_CU_METER: Final[float] = _t(
    "POUND_PER_GALLON_TO_KG_PER_CU_METER", 119.8264273
)
POUND_PER_HOUR_TO_KG_PER_SECOND: Final[float] = _t(
    "POUND_PER_HOUR_TO_KG_PER_SECOND", 0.00012599788055555556
)
POUND_PER_MINUTE_TO_KG_PER_SECOND: Final[float] = _t(
    "POUND_PER_MINUTE_TO_KG_PER_SECOND", 0.007559872833333333
)
POUND_PER_SECOND_TO_KG_PER_SECOND: Final[float] = _t(
    "POUND_PER_SECOND_TO_KG_PER_SECOND", 0.45359237
)
POUND_TO_KILOGRAM: Final[float] = _t("POUND_TO_KILOGRAM", 0.45359237)
PSI_TO_KPA: Final[float] = _t("PSI_TO_KPA", 6.894757293168)
PSI_TO_PASCAL: Final[float] = _t("PSI_TO_PASCAL", 6894.757293168)
RANKINE_RATIO: Final[float] = _t("RANKINE_RATIO", 0.5555555555555556)
R_UNIVERSAL: Final[float] = _t("R_UNIVERSAL", 8.314462618)
R_UNIVERSAL_KMOL: Final[float] = _t("R_UNIVERSAL_KMOL", 8314.462618)
SATP_PRESSURE_PA: Final[float] = _t("SATP_PRESSURE_PA", 100000.0)
SATP_TEMPERATURE_K: Final[float] = _t("SATP_TEMPERATURE_K", 298.15)
SCFM_60F_TEMPERATURE_K: Final[float] = _t("SCFM_60F_TEMPERATURE_K", 288.706)
SCFM_70F_TEMPERATURE_K: Final[float] = _t("SCFM_70F_TEMPERATURE_K", 294.261)
SCFM_PRESSURE_PA: Final[float] = _t("SCFM_PRESSURE_PA", 101325.0)
SCFM_TO_CU_METER_PER_HOUR_AT_60F: Final[float] = _t(
    "SCFM_TO_CU_METER_PER_HOUR_AT_60F", 1.69901079552
)
SECOND_TO_SECOND: Final[float] = _t("SECOND_TO_SECOND", 1.0)
SHORT_TON_TO_KILOGRAM: Final[float] = _t("SHORT_TON_TO_KILOGRAM", 907.18474)
SLUG_TO_KILOGRAM: Final[float] = _t("SLUG_TO_KILOGRAM", 14.59390294)
SQ_CENTIMETER_TO_SQ_METER: Final[float] = _t("SQ_CENTIMETER_TO_SQ_METER", 0.0001)
SQ_FOOT_PER_SECOND_TO_SQ_METER_PER_SECOND: Final[float] = _t(
    "SQ_FOOT_PER_SECOND_TO_SQ_METER_PER_SECOND", 0.09290304
)
SQ_FOOT_TO_SQ_METER: Final[float] = _t("SQ_FOOT_TO_SQ_METER", 0.09290304)
SQ_INCH_TO_SQ_METER: Final[float] = _t("SQ_INCH_TO_SQ_METER", 0.00064516)
SQ_KILOMETER_TO_SQ_METER: Final[float] = _t("SQ_KILOMETER_TO_SQ_METER", 1000000.0)
SQ_METER_PER_SECOND_TO_SQ_METER_PER_SECOND: Final[float] = _t(
    "SQ_METER_PER_SECOND_TO_SQ_METER_PER_SECOND", 1.0
)
SQ_METER_TO_SQ_METER: Final[float] = _t("SQ_METER_TO_SQ_METER", 1.0)
SQ_MILLIMETER_TO_SQ_METER: Final[float] = _t("SQ_MILLIMETER_TO_SQ_METER", 1e-06)
SQ_YARD_TO_SQ_METER: Final[float] = _t("SQ_YARD_TO_SQ_METER", 0.83612736)
STANDARD_GRAVITY: Final[float] = _t("STANDARD_GRAVITY", 9.80665)
STOKE_TO_SQ_METER_PER_SECOND: Final[float] = _t("STOKE_TO_SQ_METER_PER_SECOND", 0.0001)
STP_OLD_PRESSURE_PA: Final[float] = _t("STP_OLD_PRESSURE_PA", 101325.0)
STP_PRESSURE_PA: Final[float] = _t("STP_PRESSURE_PA", 100000.0)
STP_TEMPERATURE_K: Final[float] = _t("STP_TEMPERATURE_K", 273.15)
THERM_TO_JOULE: Final[float] = _t("THERM_TO_JOULE", 105505585.262)
TORR_TO_PASCAL: Final[float] = _t("TORR_TO_PASCAL", 133.322387415)
TPD_TO_LB: Final[float] = _t("TPD_TO_LB", 2204.6226218487755)
UNIT_ALIASES: dict[str, list[str]] = _t_dict_list(
    "UNIT_ALIASES",
    {
        "m": ["meter", "meters", "metre", "metres"],
        "cm": ["centimeter", "centimeters", "centimetre", "centimetres"],
        "mm": ["millimeter", "millimeters", "millimetre", "millimetres"],
        "um": ["µm", "micrometer", "micrometre", "micron"],
        "nm": ["nanometer", "nanometers", "nanometre", "nanometres"],
        "Å": ["angstrom", "ångström", "a"],
        "mil": ["thou"],
        "km": ["kilometer", "kilometers", "kilometre", "kilometres"],
        "ft": ["foot", "feet", "ft"],
        "in": ["inch", "inches", "in"],
        "yd": ["yard", "yards"],
        "mi": ["mile", "miles"],
        "m2": ["m^2", "square meter", "square metre"],
        "cm2": ["cm^2", "square centimeter", "square centimetre"],
        "mm2": ["square millimeter", "square millimetre"],
        "km2": ["square kilometer", "square kilometre"],
        "in2": ["square inch", "sq in"],
        "ft2": ["square foot", "sq ft"],
        "yd2": ["square yard", "sq yd"],
        "acre": ["acres"],
        "hectare": ["hectares"],
        "m3": ["m³", "m^3", "cubic meter", "cubic metre", "cu m"],
        "L": ["l", "liter", "litre", "liters", "litres"],
        "mL": ["ml", "milliliter", "millilitre"],
        "cm3": ["cm³", "cm^3", "cubic centimeter", "cubic centimetre", "cc"],
        "mm3": ["mm³", "mm^3", "cubic millimeter", "cubic millimetre"],
        "ft3": ["ft³", "ft^3", "cubic foot", "cubic feet", "cu ft"],
        "in3": ["in³", "in^3", "cubic inch", "cu in"],
        "gal": ["gallon", "gallons", "us gallon"],
        "imp_gal": ["imperial gallon", "uk gallon"],
        "qt": ["quart", "quarts"],
        "pt": ["pint", "pints"],
        "fl_oz": ["fluid ounce", "fluid ounces", "fl oz"],
        "bbl": ["barrel", "barrels"],
        "kg": ["kilogram", "kilograms"],
        "g": ["gram", "grams"],
        "mg": ["milligram", "milligrams"],
        "µg": ["ug", "microgram", "micrograms"],
        "lb": ["pound", "pounds", "lbs"],
        "oz": ["ounce", "ounces"],
        "ton": ["short ton", "us ton"],
        "tonne": ["metric ton", "metric tons", "t"],
        "long_ton": ["long ton", "uk ton"],
        "slug": ["slugs"],
        "grain": ["grains", "gr"],
        "s": ["sec", "second", "seconds"],
        "min": ["minute", "minutes"],
        "hr": ["hour", "hours", "h"],
        "day": ["days", "d"],
        "K": ["kelvin", "k"],
        "C": ["celsius", "degC", "°C"],
        "F": ["fahrenheit", "degF", "°F"],
        "R": ["rankine", "degR", "°R"],
        "Pa": ["pascal", "pascals"],
        "kPa": ["kilopascal", "kilopascals"],
        "MPa": ["megapascal", "megapascals"],
        "GPa": ["gigapascal", "gigapascals"],
        "bar": ["bars"],
        "atm": ["atmosphere", "atmospheres"],
        "psi": ["pounds per square inch"],
        "mbar": ["millibar", "millibars"],
        "torr": ["torr"],
        "mmHg": ["mm hg", "millimeter of mercury"],
        "inHg": ["inch of mercury", "in hg"],
        "inH2O": ["inch of water", "in h2o"],
        "ftH2O": ["foot of water", "ft h2o"],
        "cmH2O": ["centimeter of water", "cm h2o"],
        "kg/s": ["kilogram per second"],
        "kg/min": ["kilogram per minute"],
        "kg/hr": ["kg/h", "kilogram per hour"],
        "kg/day": ["kilogram per day", "kg/d"],
        "g/s": ["gram per second"],
        "g/min": ["gram per minute"],
        "g/hr": ["gram per hour"],
        "g/day": ["gram per day"],
        "lb/s": ["pound per second"],
        "lb/min": ["pound per minute", "lb/min"],
        "lb/hr": ["lb/h", "pound per hour"],
        "lb/day": ["pound per day", "lb/d"],
        "ton/hr": ["short ton per hour"],
        "tonne/hr": ["metric ton per hour"],
        "tonne/day": ["metric ton per day"],
        "ton/day": ["short ton per day"],
        "SCFM": ["scfm", "standard cubic feet per minute"],
        "ACFM": ["acfm", "actual cubic feet per minute"],
        "Nm3/hr": ["Nm³/hr", "nm3/hr", "nm³/hr", "normal cubic meter per hour"],
        "m3/s": ["m³/s", "cubic meter per second"],
        "m3/min": ["m³/min", "cubic meter per minute"],
        "m3/hr": ["m³/hr", "cubic meter per hour"],
        "m3/day": ["m³/day", "cubic meter per day"],
        "ft3/s": ["ft³/s", "cubic foot per second"],
        "ft3/min": ["ft³/min", "cubic foot per minute", "cfm"],
        "ft3/hr": ["ft³/hr", "cubic foot per hour"],
        "L/s": ["l/s", "liter per second"],
        "L/min": ["l/min", "liter per minute"],
        "L/hr": ["l/hr", "liter per hour"],
        "L/day": ["l/day", "liter per day"],
        "gal/min": ["gpm", "gallon per minute"],
        "gal/hr": ["gallon per hour", "gph"],
        "gal/day": ["gallon per day", "gpd"],
        "imp_gal/min": ["imperial gallon per minute"],
        "imp_gal/hr": ["imperial gallon per hour"],
        "imp_gal/day": ["imperial gallon per day"],
        "bbl/day": ["barrel per day", "bpd"],
        "W": ["watt", "watts"],
        "kW": ["kilowatt", "kilowatts"],
        "MW": ["megawatt", "megawatts"],
        "GW": ["gigawatt", "gigawatts"],
        "hp": ["horsepower", "HP"],
        "metric_hp": ["metric horsepower", "ps"],
        "BTU/hr": ["btu/hr", "BTU/h", "btu/h"],
        "MMBTU/hr": ["mmbtu/hr", "MMBTU/h", "mmbtu/h"],
        "cal/s": ["calorie per second"],
        "kcal/hr": ["kilocalorie per hour"],
        "ft·lbf/s": ["ft-lbf/s", "foot pound per second"],
        "J": ["joule", "joules"],
        "kJ": ["kilojoule", "kilojoules"],
        "MJ": ["megajoule", "megajoules"],
        "GJ": ["gigajoule", "gigajoules"],
        "Wh": ["watt hour", "watt-hour"],
        "kWh": ["kilowatt hour", "kilowatt-hour"],
        "MWh": ["megawatt hour", "megawatt-hour"],
        "BTU": ["btu"],
        "cal": ["calorie", "calories"],
        "kcal": ["kilocalorie", "kilocalories"],
        "therm": ["therms"],
        "erg": ["ergs"],
        "eV": ["ev", "electron volt", "electron volts"],
        "kg/m3": ["kg/m³", "kilogram per cubic meter"],
        "kg/L": ["kg/l", "kilogram per liter"],
        "g/cm3": ["g/cm³", "gram per cubic centimeter", "specific gravity"],
        "g/L": ["g/l", "gram per liter"],
        "lb/ft3": ["lb/ft³", "pound per cubic foot"],
        "lb/gal": ["pound per gallon"],
        "Pa·s": ["pa.s", "pascal second", "pascal-second"],
        "mPa·s": ["mpa.s", "millipascal second", "millipascal-second"],
        "cP": ["cp", "centipoise"],
        "P": ["poise"],
        "lb/ft·s": ["lb/ft*s", "pound per foot second"],
        "m2/s": ["square meter per second"],
        "cSt": ["cst", "centistokes"],
        "St": ["st", "stokes"],
        "ft2/s": ["ft²/s", "square foot per second"],
        "W/m·K": ["w/mk", "watt per meter kelvin"],
        "BTU/(ft·hr·°F)": ["btu/ft·hr·f", "btu/(ft hr f)"],
        "cal/(cm·s·°C)": ["cal/(cm s C)", "cal/cm·s·°C"],
        "W/m2·K": ["w/m²k", "watt per square meter kelvin"],
        "BTU/(ft2·hr·°F)": ["btu/(ft² hr f)", "btu/ft²·hr·°F"],
        "J/kg·K": ["j/kgk", "joule per kilogram kelvin"],
        "kJ/kg·K": ["kj/kgk", "kilojoule per kilogram kelvin"],
        "BTU/lb·°F": ["btu/lb-f", "btu per pound fahrenheit"],
        "cal/g·°C": ["cal/g-c", "calorie per gram celsius"],
    },
)
UNIT_LABEL_BTU_LB: Final[str] = _t_str("UNIT_LABEL_BTU_LB", "BTU/lb")
UNIT_LABEL_KG_HR: Final[str] = _t_str("UNIT_LABEL_KG_HR", "kg/hr")
UNIT_LABEL_LB_HR: Final[str] = _t_str("UNIT_LABEL_LB_HR", "lb/hr")
UNIT_LABEL_MG_KG: Final[str] = _t_str("UNIT_LABEL_MG_KG", "mg/kg")
UNIT_LABEL_MJ_KG: Final[str] = _t_str("UNIT_LABEL_MJ_KG", "MJ/kg")
UNIT_LABEL_SCFM: Final[str] = _t_str("UNIT_LABEL_SCFM", "SCFM")
UNIT_LABEL_WT_PERCENT: Final[str] = _t_str("UNIT_LABEL_WT_PERCENT", "wt%")
US_BARREL_TO_CU_METER: Final[float] = _t("US_BARREL_TO_CU_METER", 0.158987294928)
US_FLUID_OUNCE_TO_CU_METER: Final[float] = _t(
    "US_FLUID_OUNCE_TO_CU_METER", 2.95735295625e-05
)
US_GALLON_TO_CU_METER: Final[float] = _t("US_GALLON_TO_CU_METER", 0.003785411784)
US_PINT_TO_CU_METER: Final[float] = _t("US_PINT_TO_CU_METER", 0.000473176473)
US_QUART_TO_CU_METER: Final[float] = _t("US_QUART_TO_CU_METER", 0.000946352946)
VALIDATION_RANGES: dict[str, tuple[float, float]] = _t_dict_tuple(
    "VALIDATION_RANGES",
    {
        "temperature_K": [0.0, 10000.0],
        "pressure_Pa": [0.0, 1000000000000.0],
        "mass_kg": [0.0, 1000000000000.0],
        "length_m": [0.0, 1000000000000.0],
        "energy_J": [-1000000000000000.0, 1000000000000000.0],
        "power_W": [0.0, 1000000000000000.0],
    },
)
WATT_HOUR_TO_JOULE: Final[float] = _t("WATT_HOUR_TO_JOULE", 3600.0)
WATT_PER_METER_KELVIN: Final[float] = _t("WATT_PER_METER_KELVIN", 1.0)
WATT_PER_SQ_METER_KELVIN: Final[float] = _t("WATT_PER_SQ_METER_KELVIN", 1.0)
WATT_TO_WATT: Final[float] = _t("WATT_TO_WATT", 1.0)
YARD_TO_METER: Final[float] = _t("YARD_TO_METER", 0.9144)
