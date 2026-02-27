#!/usr/bin/env python3
"""Unit Conversion Constants

NIST-standard conversion factors and physical constants for unit conversions.
All values are sourced from NIST Special Publication 811 (2008 Edition) and
CODATA 2018 recommended values.

References:
- NIST SP 811: Guide for the Use of the International System of Units (SI)
- CODATA 2018: Committee on Data for Science and Technology
"""

from typing import Final

# =============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS (CODATA 2018)
# =============================================================================

# Universal gas constant [J/(mol·K)]
R_UNIVERSAL: Final[float] = 8.314462618
# Universal gas constant [J/(kmol·K)] - for legacy engineering units
R_UNIVERSAL_KMOL: Final[float] = R_UNIVERSAL * 1000.0

# Standard gravity [m/s²]
STANDARD_GRAVITY: Final[float] = 9.80665

# Avogadro constant [1/mol]
AVOGADRO_NUMBER: Final[float] = 6.02214076e23

# Boltzmann constant [J/K]
BOLTZMANN_CONSTANT: Final[float] = 1.380649e-23

# =============================================================================
# STANDARD CONDITIONS
# =============================================================================

# Standard Temperature and Pressure (IUPAC - Since 1982)
# Reference: IUPAC. Compendium of Chemical Terminology, 2nd ed. (the "Gold Book").
# STP is defined as 273.15 K (0°C) and 10^5 Pa (1 bar).
STP_TEMPERATURE_K: Final[float] = 273.15  # 0°C
STP_PRESSURE_PA: Final[float] = 100000.0  # 1 bar

# Old Standard Temperature and Pressure (Pre-1982) / Physical Chemistry
# Often used in older data (e.g., JANAF tables often reference 1 atm)
STP_OLD_PRESSURE_PA: Final[float] = 101325.0  # 1 atm (Standard Atmosphere)

# Normal Temperature and Pressure (NTP - common industrial)
NTP_TEMPERATURE_K: Final[float] = 293.15  # 20°C
NTP_PRESSURE_PA: Final[float] = 101325.0  # 1 atm

# Standard Ambient Temperature and Pressure (SATP)
SATP_TEMPERATURE_K: Final[float] = 298.15  # 25°C
SATP_PRESSURE_PA: Final[float] = 100000.0  # 1 bar

# SCFM Standard Conditions (US Engineering)
# Note: SCFM is ambiguous and requires explicit definition.
# Common definitions include 60°F or 70°F and 1 atm (14.696 psia).
SCFM_60F_TEMPERATURE_K: Final[float] = 288.706  # 60°F
SCFM_70F_TEMPERATURE_K: Final[float] = 294.261  # 70°F
SCFM_PRESSURE_PA: Final[float] = 101325.0  # 1 atm

# Ideal gas molar volume at STP (IUPAC current) [m³/mol]
MOLAR_VOLUME_STP: Final[float] = 0.02271095  # R * 273.15 / 100000

# Ideal gas molar volume at STP (Old/Standard Atmosphere) [m³/mol]
MOLAR_VOLUME_STP_OLD: Final[float] = 0.022413969545  # R * 273.15 / 101325

# =============================================================================
# LENGTH CONVERSIONS (all to meters) - NIST Exact Values
# =============================================================================

METER_TO_METER: Final[float] = 1.0
CENTIMETER_TO_METER: Final[float] = 0.01
MILLIMETER_TO_METER: Final[float] = 0.001
KILOMETER_TO_METER: Final[float] = 1000.0
MICROMETER_TO_METER: Final[float] = 1.0e-6
NANOMETER_TO_METER: Final[float] = 1.0e-9
ANGSTROM_TO_METER: Final[float] = 1.0e-10

# US/Imperial units (exact by definition)
INCH_TO_METER: Final[float] = 0.0254
FOOT_TO_METER: Final[float] = 0.3048
YARD_TO_METER: Final[float] = 0.9144
MILE_TO_METER: Final[float] = 1609.344
MIL_TO_METER: Final[float] = 2.54e-5  # 1/1000 inch

# =============================================================================
# AREA CONVERSIONS (all to m²)
# =============================================================================

SQ_METER_TO_SQ_METER: Final[float] = 1.0
SQ_CENTIMETER_TO_SQ_METER: Final[float] = 1.0e-4
SQ_MILLIMETER_TO_SQ_METER: Final[float] = 1.0e-6
SQ_KILOMETER_TO_SQ_METER: Final[float] = 1.0e6

SQ_INCH_TO_SQ_METER: Final[float] = 6.4516e-4
SQ_FOOT_TO_SQ_METER: Final[float] = 0.09290304
SQ_YARD_TO_SQ_METER: Final[float] = 0.83612736
ACRE_TO_SQ_METER: Final[float] = 4046.8564224
HECTARE_TO_SQ_METER: Final[float] = 10000.0

# =============================================================================
# VOLUME CONVERSIONS (all to m³)
# =============================================================================

CU_METER_TO_CU_METER: Final[float] = 1.0
LITER_TO_CU_METER: Final[float] = 0.001
MILLILITER_TO_CU_METER: Final[float] = 1.0e-6
CU_CENTIMETER_TO_CU_METER: Final[float] = 1.0e-6
CU_MILLIMETER_TO_CU_METER: Final[float] = 1.0e-9

CU_FOOT_TO_CU_METER: Final[float] = 0.028316846592
CU_INCH_TO_CU_METER: Final[float] = 1.6387064e-5

# US liquid measures (exact by definition)
US_GALLON_TO_CU_METER: Final[float] = 0.003785411784
US_QUART_TO_CU_METER: Final[float] = 0.000946352946
US_PINT_TO_CU_METER: Final[float] = 0.000473176473
US_FLUID_OUNCE_TO_CU_METER: Final[float] = 2.95735295625e-5
US_BARREL_TO_CU_METER: Final[float] = 0.158987294928

# Imperial measures
IMPERIAL_GALLON_TO_CU_METER: Final[float] = 0.00454609

# =============================================================================
# MASS CONVERSIONS (all to kg)
# =============================================================================

KILOGRAM_TO_KILOGRAM: Final[float] = 1.0
GRAM_TO_KILOGRAM: Final[float] = 0.001
MILLIGRAM_TO_KILOGRAM: Final[float] = 1.0e-6
METRIC_TON_TO_KILOGRAM: Final[float] = 1000.0

# US/Imperial (exact by definition)
POUND_TO_KILOGRAM: Final[float] = 0.45359237
OUNCE_TO_KILOGRAM: Final[float] = 2.8349523125e-2  # [kg/oz] Source: NIST SP 811

# Derived Mass Constants
LB_TO_G: Final[float] = POUND_TO_KILOGRAM * 1000.0  # [g/lb]
SHORT_TON_TO_KILOGRAM: Final[float] = 907.18474
LONG_TON_TO_KILOGRAM: Final[float] = 1016.0469088
SLUG_TO_KILOGRAM: Final[float] = 14.59390294
GRAIN_TO_KILOGRAM: Final[float] = 6.479891e-5

# Derived mass conversions
KG_TO_LB: Final[float] = 1.0 / POUND_TO_KILOGRAM
LB_TO_KG: Final[float] = POUND_TO_KILOGRAM
TPD_TO_LB: Final[float] = (
    METRIC_TON_TO_KILOGRAM * KG_TO_LB
)  # 2204.6226 [lb/ton] 1 Metric Ton = 1000 kg

# =============================================================================
# TIME CONVERSIONS (all to seconds)
# =============================================================================

SECOND_TO_SECOND: Final[float] = 1.0
MINUTE_TO_SECOND: Final[float] = 60.0
HOUR_TO_SECOND: Final[float] = 3600.0
DAY_TO_SECOND: Final[float] = 86400.0
HOURS_PER_DAY: Final[int] = 24  # [hr/day] 24-hour civil day

# =============================================================================
# TEMPERATURE - Note: These require special handling as they're not linear
# =============================================================================

CELSIUS_OFFSET: Final[float] = 273.15
RANKINE_RATIO: Final[float] = 5.0 / 9.0

# =============================================================================
# PRESSURE CONVERSIONS (all to Pa)
# =============================================================================

PASCAL_TO_PASCAL: Final[float] = 1.0
KILOPASCAL_TO_PASCAL: Final[float] = 1000.0
MEGAPASCAL_TO_PASCAL: Final[float] = 1.0e6
GIGAPASCAL_TO_PASCAL: Final[float] = 1.0e9

BAR_TO_PASCAL: Final[float] = 100000.0
MILLIBAR_TO_PASCAL: Final[float] = 100.0
ATMOSPHERE_TO_PASCAL: Final[float] = 101325.0  # Exact by definition
ATM_TO_KPA: Final[float] = 101.325
BAR_TO_KPA: Final[float] = 100.0

# US/Imperial pressure (exact by definition)
PSI_TO_PASCAL: Final[float] = 6894.757293168
PSI_TO_KPA: Final[float] = 6.894757293168
TORR_TO_PASCAL: Final[float] = 133.322387415
MMHG_TO_PASCAL: Final[float] = 133.322387415

# Other pressure units
INCH_HG_TO_PASCAL: Final[float] = 3386.389
INCH_H2O_TO_PASCAL: Final[float] = 249.082
FOOT_H2O_TO_PASCAL: Final[float] = 2989.07
CM_H2O_TO_PASCAL: Final[float] = 98.0665

# =============================================================================
# ENERGY CONVERSIONS (all to Joules)
# =============================================================================

JOULE_TO_JOULE: Final[float] = 1.0
KILOJOULE_TO_JOULE: Final[float] = 1000.0
MEGAJOULE_TO_JOULE: Final[float] = 1.0e6
GIGAJOULE_TO_JOULE: Final[float] = 1.0e9

# Calorie (International Table calorie, exact by definition)
CALORIE_TO_JOULE: Final[float] = 4.184
KILOCALORIE_TO_JOULE: Final[float] = 4184.0

# BTU (International Table BTU, exact by definition)
BTU_TO_JOULE: Final[float] = 1055.05585262
THERM_TO_JOULE: Final[float] = 105505585.262

# Electrical
WATT_HOUR_TO_JOULE: Final[float] = 3600.0
KILOWATT_HOUR_TO_JOULE: Final[float] = 3.6e6
MEGAWATT_HOUR_TO_JOULE: Final[float] = 3.6e9

# Other
ERG_TO_JOULE: Final[float] = 1.0e-7
ELECTRON_VOLT_TO_JOULE: Final[float] = 1.602176634e-19

# =============================================================================
# SPECIFIC ENERGY CONVERSIONS (Energy per Mass)
# =============================================================================

# BTU/lb to J/kg and MJ/kg
BTU_PER_LB_TO_J_PER_KG: Final[float] = BTU_TO_JOULE / 0.45359237
BTU_PER_LB_TO_MJ_PER_KG: Final[float] = BTU_PER_LB_TO_J_PER_KG / 1.0e6


# =============================================================================
# POWER CONVERSIONS (all to Watts)
# =============================================================================

WATT_TO_WATT: Final[float] = 1.0
KILOWATT_TO_WATT: Final[float] = 1000.0
MEGAWATT_TO_WATT: Final[float] = 1.0e6
GIGAWATT_TO_WATT: Final[float] = 1.0e9

# BTU/hr (derived from BTU and hour)
BTU_PER_HOUR_TO_WATT: Final[float] = BTU_TO_JOULE / HOUR_TO_SECOND  # 0.29307107017222
MMBTU_PER_HOUR_TO_WATT: Final[float] = BTU_PER_HOUR_TO_WATT * 1.0e6


# Other power units
HORSEPOWER_TO_WATT: Final[float] = 745.69987158227022  # Mechanical HP (exact)
METRIC_HORSEPOWER_TO_WATT: Final[float] = 735.49875
CALORIE_PER_SECOND_TO_WATT: Final[float] = 4.184
KCAL_PER_HOUR_TO_WATT: Final[float] = 1.163
FOOT_POUND_PER_SECOND_TO_WATT: Final[float] = 1.3558179483314004

# =============================================================================
# FLOW RATE CONVERSIONS
# =============================================================================

# Mass flow (all to kg/s)
KG_PER_SECOND_TO_KG_PER_SECOND: Final[float] = 1.0
KG_PER_MINUTE_TO_KG_PER_SECOND: Final[float] = 1.0 / 60.0
KG_PER_HOUR_TO_KG_PER_SECOND: Final[float] = 1.0 / 3600.0
GRAM_PER_SECOND_TO_KG_PER_SECOND: Final[float] = 0.001
POUND_PER_SECOND_TO_KG_PER_SECOND: Final[float] = POUND_TO_KILOGRAM
POUND_PER_MINUTE_TO_KG_PER_SECOND: Final[float] = POUND_TO_KILOGRAM / 60.0
POUND_PER_HOUR_TO_KG_PER_SECOND: Final[float] = POUND_TO_KILOGRAM / 3600.0

# Volumetric flow for SCFM (Standard Cubic Feet per Minute)
# 1 ft³ = 0.028316846592 m³, at standard conditions
# Note: SCFM requires specifying which standard conditions are being used
SCFM_TO_CU_METER_PER_HOUR_AT_60F: Final[float] = (
    CU_FOOT_TO_CU_METER * 60.0
)  # 1.699010795

# =============================================================================
# DENSITY CONVERSIONS (all to kg/m³)
# =============================================================================

KG_PER_CU_METER_TO_KG_PER_CU_METER: Final[float] = 1.0
GRAM_PER_CU_CM_TO_KG_PER_CU_METER: Final[float] = 1000.0
GRAM_PER_LITER_TO_KG_PER_CU_METER: Final[float] = 1.0
POUND_PER_CU_FOOT_TO_KG_PER_CU_METER: Final[float] = 16.01846337396
POUND_PER_GALLON_TO_KG_PER_CU_METER: Final[float] = 119.8264273

# =============================================================================
# VISCOSITY CONVERSIONS
# =============================================================================

# Dynamic viscosity (all to Pa·s)
PASCAL_SECOND_TO_PASCAL_SECOND: Final[float] = 1.0
CENTIPOISE_TO_PASCAL_SECOND: Final[float] = 0.001
POISE_TO_PASCAL_SECOND: Final[float] = 0.1
POUND_PER_FOOT_SECOND_TO_PASCAL_SECOND: Final[float] = 1.4881639436

# Kinematic viscosity (all to m²/s)
SQ_METER_PER_SECOND_TO_SQ_METER_PER_SECOND: Final[float] = 1.0
CENTISTOKE_TO_SQ_METER_PER_SECOND: Final[float] = 1.0e-6
STOKE_TO_SQ_METER_PER_SECOND: Final[float] = 1.0e-4
SQ_FOOT_PER_SECOND_TO_SQ_METER_PER_SECOND: Final[float] = 0.09290304

# =============================================================================
# THERMAL PROPERTY CONVERSIONS
# =============================================================================

# Thermal conductivity (all to W/(m·K))
WATT_PER_METER_KELVIN: Final[float] = 1.0
BTU_PER_FOOT_HOUR_FAHRENHEIT_TO_W_PER_M_K: Final[float] = 1.7307346664
CAL_PER_CM_SECOND_CELSIUS_TO_W_PER_M_K: Final[float] = 418.4

# Heat transfer coefficient (all to W/(m²·K))
WATT_PER_SQ_METER_KELVIN: Final[float] = 1.0
BTU_PER_SQ_FOOT_HOUR_FAHRENHEIT_TO_W_PER_M2_K: Final[float] = 5.6782633411

# Specific heat (all to J/(kg·K))
JOULE_PER_KG_KELVIN: Final[float] = 1.0
BTU_PER_POUND_FAHRENHEIT_TO_J_PER_KG_K: Final[float] = 4186.8
CAL_PER_GRAM_CELSIUS_TO_J_PER_KG_K: Final[float] = 4186.8

# =============================================================================
# GAS PROPERTIES AT STP (0°C, 101.325 kPa)
# =============================================================================

# Molecular weights [kg/kmol]
MW_AIR: Final[float] = 28.9647
MW_NITROGEN: Final[float] = 28.0134
MW_OXYGEN: Final[float] = 31.9988
MW_HYDROGEN: Final[float] = 2.01588
MW_METHANE: Final[float] = 16.0425
MW_CARBON_MONOXIDE: Final[float] = 28.0101
MW_CARBON_DIOXIDE: Final[float] = 44.0095
MW_WATER_VAPOR: Final[float] = 18.01528
MW_AMMONIA: Final[float] = 17.0305
MW_HYDROGEN_SULFIDE: Final[float] = 34.0809

# Densities at STP [kg/m³]
DENSITY_STP_AIR: Final[float] = 1.2922
DENSITY_STP_NITROGEN: Final[float] = 1.2506
DENSITY_STP_OXYGEN: Final[float] = 1.4289
DENSITY_STP_HYDROGEN: Final[float] = 0.08988
DENSITY_STP_METHANE: Final[float] = 0.7168
DENSITY_STP_CO: Final[float] = 1.2500
DENSITY_STP_CO2: Final[float] = 1.9768
DENSITY_STP_WATER_VAPOR: Final[float] = (
    0.00485  # [kg/m³] Saturated water vapor at STP (0°C, 1 atm);
    # Source: Perry's Chemical Engineers' Handbook, 9th Ed., Table 2-95
)

# =============================================================================
# STANDARD FLUID PROPERTIES (WATER)
# =============================================================================

# Specific Heat Capacity [J/(kg·K)]
# Source: IAPWS-IF97 at standard conditions
CP_WATER_LIQUID: Final[float] = 4181.3  # liquid at 25°C, 1 atm
CP_WATER_VAPOR: Final[float] = 1858.9  # vapor at 100°C, 1 atm (ideal gas limit ~1860)
# Note: Average engineering value often used is 2.01 kJ/kgK (2010 J/kgK) for steam

# Latent Heat of Vaporization [J/kg] at 100°C, 1 atm
H_VAP_WATER: Final[float] = 2257000.0  # 2257 kJ/kg

# Density [kg/m³]
DENSITY_WATER_STD: Final[float] = 997.0  # Liquid at 25°C


# =============================================================================
# UNIT ALIASES AND CANONICAL NAMES
# =============================================================================

# Define canonical unit names and their aliases
UNIT_ALIASES: dict[str, list[str]] = {
    # Length
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
    # Area
    "m2": ["m^2", "square meter", "square metre"],
    "cm2": ["cm^2", "square centimeter", "square centimetre"],
    "mm2": ["square millimeter", "square millimetre"],
    "km2": ["square kilometer", "square kilometre"],
    "in2": ["square inch", "sq in"],
    "ft2": ["square foot", "sq ft"],
    "yd2": ["square yard", "sq yd"],
    "acre": ["acres"],
    "hectare": ["hectares"],
    # Volume
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
    # Mass
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
    # Time
    "s": ["sec", "second", "seconds"],
    "min": ["minute", "minutes"],
    "hr": ["hour", "hours", "h"],
    "day": ["days", "d"],
    # Temperature
    "K": ["kelvin", "k"],
    "C": ["celsius", "degC", "°C"],
    "F": ["fahrenheit", "degF", "°F"],
    "R": ["rankine", "degR", "°R"],
    # Pressure
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
    # Mass flow
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
    # Volumetric flow
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
    # Power
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
    # Energy
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
    # Density
    "kg/m3": ["kg/m³", "kilogram per cubic meter"],
    "kg/L": ["kg/l", "kilogram per liter"],
    "g/cm3": ["g/cm³", "gram per cubic centimeter", "specific gravity"],
    "g/L": ["g/l", "gram per liter"],
    "lb/ft3": ["lb/ft³", "pound per cubic foot"],
    "lb/gal": ["pound per gallon"],
    # Dynamic viscosity
    "Pa·s": ["pa.s", "pascal second", "pascal-second"],
    "mPa·s": ["mpa.s", "millipascal second", "millipascal-second"],
    "cP": ["cp", "centipoise"],
    "P": ["poise"],
    "lb/ft·s": ["lb/ft*s", "pound per foot second"],
    # Kinematic viscosity
    "m2/s": ["square meter per second"],
    "cSt": ["cst", "centistokes"],
    "St": ["st", "stokes"],
    "ft2/s": ["ft²/s", "square foot per second"],
    # Thermal conductivity
    "W/m·K": ["w/mk", "watt per meter kelvin"],
    "BTU/(ft·hr·°F)": ["btu/ft·hr·f", "btu/(ft hr f)"],
    "cal/(cm·s·°C)": ["cal/(cm s C)", "cal/cm·s·°C"],
    # Heat transfer coefficient
    "W/m2·K": ["w/m²k", "watt per square meter kelvin"],
    "BTU/(ft2·hr·°F)": ["btu/(ft² hr f)", "btu/ft²·hr·°F"],
    # Specific heat
    "J/kg·K": ["j/kgk", "joule per kilogram kelvin"],
    "kJ/kg·K": ["kj/kgk", "kilojoule per kilogram kelvin"],
    "BTU/lb·°F": ["btu/lb-f", "btu per pound fahrenheit"],
    "cal/g·°C": ["cal/g-c", "calorie per gram celsius"],
}

# =============================================================================
# VALIDATION RANGES
# =============================================================================

# Define physically meaningful ranges for validation
VALIDATION_RANGES = {
    "temperature_K": (0.0, 10000.0),  # Absolute zero to plasma temps
    "pressure_Pa": (0.0, 1.0e12),  # Vacuum to extreme high pressure
    "mass_kg": (0.0, 1.0e12),  # Non-negative
    "length_m": (0.0, 1.0e12),  # Non-negative
    "energy_J": (-1.0e15, 1.0e15),  # Can be negative (e.g., exothermic)
    "power_W": (0.0, 1.0e15),  # Non-negative
}

# =============================================================================
# NUMERICAL CONSTANTS
# =============================================================================

# =============================================================================
# NUMERICAL CONSTANTS
# =============================================================================

# Minimum basis for molar calculations to avoid scaling issues
MIN_BASIS_MOLES: Final[float] = 1.0


# =============================================================================
# UI DISPLAY LABELS
# =============================================================================

UNIT_LABEL_WT_PERCENT: Final[str] = "wt%"
UNIT_LABEL_MG_KG: Final[str] = "mg/kg"
UNIT_LABEL_BTU_LB: Final[str] = "BTU/lb"
UNIT_LABEL_MJ_KG: Final[str] = "MJ/kg"
UNIT_LABEL_LB_HR: Final[str] = "lb/hr"
UNIT_LABEL_KG_HR: Final[str] = "kg/hr"
UNIT_LABEL_SCFM: Final[str] = "SCFM"
