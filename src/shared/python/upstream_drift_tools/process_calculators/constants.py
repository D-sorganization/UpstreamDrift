# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Physical Constants and Conversion Factors for Process Calculators.

This module provides NIST-standard physical constants and conversion factors
needed by the standalone process calculators. Values are sourced from:
- NIST Special Publication 811 (2008 Edition)
- CODATA 2018 recommended values
- Perry's Chemical Engineers' Handbook, 9th ed.
"""

from typing import Final

from upstream_drift_tools.utils.unit_constants import (
    ATM_TO_KPA,
    ATMOSPHERE_TO_PASCAL,
    AVOGADRO_NUMBER,
    BAR_TO_PASCAL,
    BOLTZMANN_CONSTANT,
    BTU_TO_JOULE,
    CALORIE_TO_JOULE,
    CENTIPOISE_TO_PASCAL_SECOND,
    HOUR_TO_SECOND,
    INCH_H2O_TO_PASCAL,
    KG_PER_HOUR_TO_KG_PER_SECOND,
    KILOCALORIE_TO_JOULE,
    KILOJOULE_TO_JOULE,
    KILOPASCAL_TO_PASCAL,
    MEGAPASCAL_TO_PASCAL,
    MINUTE_TO_SECOND,
    MMHG_TO_PASCAL,
    MOLAR_VOLUME_STP_OLD,
    POISE_TO_PASCAL_SECOND,
    POUND_PER_HOUR_TO_KG_PER_SECOND,
    PSI_TO_PASCAL,
    R_UNIVERSAL,
    STANDARD_GRAVITY,
    TORR_TO_PASCAL,
)
from upstream_drift_tools.utils.unit_constants import (
    CELSIUS_OFFSET as CELSIUS_TO_KELVIN_OFFSET,
)
from upstream_drift_tools.utils.unit_constants import (
    DENSITY_WATER_STD as _DENSITY_WATER_STD,
)
from upstream_drift_tools.utils.unit_constants import (
    HOURS_PER_DAY as _HOURS_PER_DAY,
)
from upstream_drift_tools.utils.unit_constants import (
    KG_TO_LB as _KG_TO_LB,
)
from upstream_drift_tools.utils.unit_constants import (
    R_UNIVERSAL_KMOL as _R_UNIVERSAL_KMOL,
)
from upstream_drift_tools.utils.unit_constants import (
    STP_TEMPERATURE_K as _STP_TEMPERATURE_K,
)

# =============================================================================
# RE-EXPORTED ALIASES (preserve backwards-compatible names)
# =============================================================================

R_GAS_J_MOL_K: Final[float] = R_UNIVERSAL  # Alias
G: Final[float] = STANDARD_GRAVITY  # Alias
NA: Final[float] = AVOGADRO_NUMBER  # Alias
KB: Final[float] = BOLTZMANN_CONSTANT  # Alias

# Pressure aliases (constants.py used shorter names)
BAR_TO_PA: Final[float] = BAR_TO_PASCAL
KPA_TO_PA: Final[float] = KILOPASCAL_TO_PASCAL
MPA_TO_PA: Final[float] = MEGAPASCAL_TO_PASCAL
ATM_TO_PA: Final[float] = ATMOSPHERE_TO_PASCAL
PSI_TO_PA: Final[float] = PSI_TO_PASCAL
TORR_TO_PA: Final[float] = TORR_TO_PASCAL
MMHG_TO_PA: Final[float] = MMHG_TO_PASCAL
INH2O_TO_PA: Final[float] = INCH_H2O_TO_PASCAL

# Standard atmosphere aliases
ATM_PA: Final[float] = ATMOSPHERE_TO_PASCAL
ATM_KPA: Final[float] = ATM_TO_KPA

# Energy aliases
KJ_TO_J: Final[float] = KILOJOULE_TO_JOULE
CAL_TO_J: Final[float] = CALORIE_TO_JOULE
KCAL_TO_J: Final[float] = KILOCALORIE_TO_JOULE
BTU_TO_J: Final[float] = BTU_TO_JOULE

# Viscosity aliases
CP_TO_PA_S: Final[float] = CENTIPOISE_TO_PASCAL_SECOND
POISE_TO_PA_S: Final[float] = POISE_TO_PASCAL_SECOND

# Flow aliases
LB_HR_TO_KG_S: Final[float] = POUND_PER_HOUR_TO_KG_PER_SECOND
KG_HR_TO_KG_S: Final[float] = KG_PER_HOUR_TO_KG_PER_SECOND

# Time aliases
SECONDS_PER_HOUR: Final[float] = HOUR_TO_SECOND
SECONDS_PER_MINUTE: Final[float] = MINUTE_TO_SECOND

# Molar volume alias
MOLAR_VOLUME_STP_ATM: Final[float] = MOLAR_VOLUME_STP_OLD

# Re-exported from unit_constants (explicit annotations for mypy visibility)
DENSITY_WATER_STD: Final[float] = _DENSITY_WATER_STD
KG_TO_LB: Final[float] = _KG_TO_LB
R_UNIVERSAL_KMOL: Final[float] = _R_UNIVERSAL_KMOL
HOURS_PER_DAY: Final[int] = _HOURS_PER_DAY
STP_TEMPERATURE_K: Final[float] = _STP_TEMPERATURE_K

# =============================================================================
# STANDARD CONDITIONS (additional, not in unit_constants)
# =============================================================================

STP_PRESSURE_KPA: Final[float] = 100.0  # 1 bar

# Standard Ambient Temperature and Pressure (SATP)
SATP_TEMPERATURE_C: Final[float] = 25.0

# Reference conditions for thermodynamic calculations
T_REF_K: Final[float] = 298.15  # Standard reference temperature (25°C)
P_REF_PA: Final[float] = 101325.0  # Standard reference pressure (1 atm)
P_REF_KPA: Final[float] = 101.325

# =============================================================================
# STEFAN-BOLTZMANN CONSTANT
# =============================================================================

# Stefan-Boltzmann constant [W/(m²·K⁴)]
STEFAN_BOLTZMANN: Final[float] = 5.670374419e-8

# =============================================================================
# TEMPERATURE CONVERSIONS (additional)
# =============================================================================

FAHRENHEIT_TO_RANKINE_OFFSET: Final[float] = 459.67
RANKINE_PER_KELVIN: Final[float] = 1.8

# =============================================================================
# FLOW CONVERSIONS
# =============================================================================

# Volumetric flow
CFM_TO_M3_S: Final[float] = 0.000471947443
LPM_TO_M3_S: Final[float] = 0.0000166667

# =============================================================================
# NUMERICAL TOLERANCES
# =============================================================================

ATOL_ZERO: Final[float] = 1e-12  # Absolute tolerance for zero comparisons
RTOL_DEFAULT: Final[float] = 1e-6  # Default relative tolerance

# =============================================================================
# COMMON MOLECULAR WEIGHTS [kg/mol]
# =============================================================================

MW_H2: Final[float] = 0.00201588
MW_O2: Final[float] = 0.031998
MW_N2: Final[float] = 0.0280134
MW_CO: Final[float] = 0.02801
MW_CO2: Final[float] = 0.04401
MW_H2O: Final[float] = 0.01801528
MW_CH4: Final[float] = 0.01604
MW_NH3: Final[float] = 0.01703
MW_H2S: Final[float] = 0.03408
MW_HCL: Final[float] = 0.036461
MW_HF: Final[float] = 0.02001
MW_SO2: Final[float] = 0.064066
MW_AIR: Final[float] = 0.028964  # Dry air average

# Dictionary form for lookup
MOLECULAR_WEIGHTS: dict[str, float] = {
    "H2": MW_H2,
    "O2": MW_O2,
    "N2": MW_N2,
    "CO": MW_CO,
    "CO2": MW_CO2,
    "H2O": MW_H2O,
    "CH4": MW_CH4,
    "NH3": MW_NH3,
    "H2S": MW_H2S,
    "HCl": MW_HCL,
    "HF": MW_HF,
    "SO2": MW_SO2,
    "Air": MW_AIR,
}

# =============================================================================
# WATER/STEAM PROPERTIES
# =============================================================================

# Water properties at 25°C
WATER_DENSITY_25C: Final[float] = 997.05  # kg/m³
WATER_VISCOSITY_25C: Final[float] = 0.00089  # Pa·s

# Triple point
WATER_TRIPLE_POINT_T: Final[float] = 273.16  # K
WATER_TRIPLE_POINT_P: Final[float] = 611.657  # Pa

# Critical point
WATER_CRITICAL_T: Final[float] = 647.096  # K
WATER_CRITICAL_P: Final[float] = 22064000.0  # Pa
WATER_CRITICAL_RHO: Final[float] = 322.0  # kg/m³

# Heat of vaporization at 100°C [J/kg]
WATER_HEAT_VAP_100C: Final[float] = 2256400.0

# Antoine equation coefficients for water (pressure in mmHg, temperature in °C)
# Valid range: 1°C to 100°C
ANTOINE_WATER_A: Final[float] = 8.07131
ANTOINE_WATER_B: Final[float] = 1730.63
ANTOINE_WATER_C: Final[float] = 233.426


def get_molecular_weight(species: str) -> float:
    """Get molecular weight for a species [kg/mol].

    Args:
        species: Chemical species name or formula

    Returns:
        Molecular weight in kg/mol, or 0.029 (air) if not found
    """
    # Normalize common variations
    normalized = species.upper().replace(" ", "")
    lookup = {k.upper(): v for k, v in MOLECULAR_WEIGHTS.items()}
    return lookup.get(normalized, MW_AIR)


def celsius_to_kelvin(temp_c: float) -> float:
    """Convert temperature from Celsius to Kelvin."""
    return float(temp_c + CELSIUS_TO_KELVIN_OFFSET)


def kelvin_to_celsius(temp_k: float) -> float:
    """Convert temperature from Kelvin to Celsius."""
    return float(temp_k - CELSIUS_TO_KELVIN_OFFSET)


def fahrenheit_to_kelvin(temp_f: float) -> float:
    """Convert temperature from Fahrenheit to Kelvin."""
    return (temp_f + FAHRENHEIT_TO_RANKINE_OFFSET) / RANKINE_PER_KELVIN


def kelvin_to_fahrenheit(temp_k: float) -> float:
    """Convert temperature from Kelvin to Fahrenheit."""
    return temp_k * RANKINE_PER_KELVIN - FAHRENHEIT_TO_RANKINE_OFFSET


def bar_to_pa(pressure_bar: float) -> float:
    """Convert pressure from bar to Pascal."""
    return pressure_bar * BAR_TO_PA


def pa_to_bar(pressure_pa: float) -> float:
    """Convert pressure from Pascal to bar."""
    return pressure_pa / BAR_TO_PA


def psi_to_pa(pressure_psi: float) -> float:
    """Convert pressure from psi to Pascal."""
    return pressure_psi * PSI_TO_PA


def pa_to_psi(pressure_pa: float) -> float:
    """Convert pressure from Pascal to psi."""
    return pressure_pa / PSI_TO_PA


# =============================================================================
# ADDITIONAL WATER PROPERTIES FOR PROCESS CALCULATORS
# =============================================================================

# Liquid water heat capacity [J/(kg·K)] at 25°C
CP_WATER_LIQUID: Final[float] = 4182.0

# Heat of vaporization of water [J/kg] at 100°C
H_VAP_WATER: Final[float] = 2256400.0

# Mass conversion
# KG_TO_LB is imported from unit_constants above

# =============================================================================
# WATER VAPOR PRESSURE COEFFICIENTS (Modified Buck Equation)
# =============================================================================
# Source: Buck, A.L. (1981). "New Equations for Computing Vapor Pressure"

WATER_VAPOR_A: Final[float] = 0.61115  # [kPa]
WATER_VAPOR_B: Final[float] = 23.036  # [K]
WATER_VAPOR_C: Final[float] = 279.82  # [K]
WATER_VAPOR_D: Final[float] = 333.7  # [K]


# =============================================================================
# UNIT CONVERSION FACTORS
# =============================================================================

# Volumetric flow: m3/s to cubic feet per minute
M3_S_TO_CFM: Final[float] = 2118.88

# Length conversions
METERS_TO_INCHES: Final[float] = 39.3701
METERS_TO_FEET: Final[float] = 3.28084
HUNDRED_FEET_IN_METERS: Final[float] = 30.48  # 100 ft = 30.48 m

# Velocity conversion: ft/s to m/s
FT_S_TO_M_S: Final[float] = 0.3048

# Density conversion: kg/m3 to lb/ft3
KG_M3_TO_LB_FT3: Final[float] = 0.062428

# Molecular weight conversion: g/mol to kg/mol
G_MOL_TO_KG_MOL: Final[float] = 1000.0

# Pressure conversion: mmHg to Pa
MMHG_TO_PA_CONV: Final[float] = 133.322

# Power conversion: Watts per horsepower
WATTS_PER_HP: Final[float] = 745.7

# =============================================================================
# FLARE DESIGN CONSTANTS (API 521)
# =============================================================================
# Source: API Standard 521 (6th Edition, 2014)

# Maximum exit velocity for smokeless flare operation [m/s]
FLARE_MAX_EXIT_VELOCITY: Final[float] = 170.0

# Safe radiation intensity at ground level for personnel access [kW/m2]
FLARE_SAFE_RADIATION_INTENSITY: Final[float] = 1.6

# Typical flame emissivity for clean hydrocarbon flames [dimensionless]
FLARE_FLAME_EMISSIVITY: Final[float] = 0.3

# Minimum flare stack height [m]
FLARE_MIN_HEIGHT: Final[float] = 10.0

# Radiation zone thresholds [kW/m2]
RADIATION_LETHAL: Final[float] = 37.5
RADIATION_DAMAGE: Final[float] = 12.5
RADIATION_SAFE: Final[float] = 1.6
RADIATION_COMFORT: Final[float] = 0.5

# Combustion efficiency parameters
FLARE_BASE_EFFICIENCY: Final[float] = 0.98
FLARE_MIN_EFFICIENCY: Final[float] = 0.95
FLARE_MAX_EFFICIENCY: Final[float] = 0.999
FLARE_H2_EFFICIENCY_BOOST: Final[float] = 0.01
FLARE_CO_EFFICIENCY_PENALTY: Final[float] = 0.02
FLARE_H2S_EFFICIENCY_PENALTY: Final[float] = 0.01
FLARE_COLD_TEMP_PENALTY: Final[float] = 0.02
FLARE_HOT_TEMP_BOOST: Final[float] = 0.01
FLARE_H2_THRESHOLD: Final[float] = 0.5
FLARE_CO_THRESHOLD: Final[float] = 0.3
FLARE_H2S_THRESHOLD: Final[float] = 0.1
FLARE_COLD_TEMP_K: Final[float] = 300.0
FLARE_HOT_TEMP_K: Final[float] = 500.0

# =============================================================================
# PRESSURE DROP CONSTANTS
# =============================================================================
# Source: Perry's Chemical Engineers' Handbook, 9th Ed.; Crane TP-410

# Reynolds number regime boundaries
RE_LAMINAR_UPPER: Final[float] = 2300.0
RE_TURBULENT_LOWER: Final[float] = 4000.0

# Laminar flow friction factor constant (Hagen-Poiseuille)
LAMINAR_FRICTION_CONSTANT: Final[float] = 64.0

# Default friction factor for laminar flow at Re ~ 1000
FRICTION_FACTOR_DEFAULT_LAMINAR: Final[float] = 0.064

# Colebrook-White roughness coefficient (denominator term)
COLEBROOK_ROUGHNESS_COEFF: Final[float] = 3.7

# Swamee-Jain Reynolds number coefficient
SWAMEE_JAIN_COEFF: Final[float] = 5.74

# Churchill correlation B-term coefficient
CHURCHILL_B_COEFF: Final[float] = 37530.0

# API RP 14E erosional velocity C-factors
API_14E_C_CONTINUOUS: Final[float] = 100.0
API_14E_C_INTERMITTENT: Final[float] = 125.0

# =============================================================================
# SCRUBBER DESIGN CONSTANTS
# =============================================================================
# Source: Perry's 9th Ed.; Sutherland (1893); Eckert (1961)

# Syngas viscosity reference at 300 K [Pa-s]
SYNGAS_VISCOSITY_REF: Final[float] = 1.8e-5

# Sutherland's formula reference temperature [K]
SUTHERLAND_T_REF: Final[float] = 300.0

# Sutherland constant for air-like gases [K]
SUTHERLAND_CONSTANT_AIR: Final[float] = 110.4

# Air molecular weight [g/mol] (for scrubber MW correction)
MW_AIR_GMOL: Final[float] = 29.0

# Eckert correlation coefficients for pressure drop
ECKERT_ALPHA: Final[float] = 85.0  # Pa/m base coefficient
ECKERT_BETA: Final[float] = 1.1  # Exponent on capacity parameter
ECKERT_GAMMA: Final[float] = 3.5  # Liquid effect coefficient

# Maximum pressure drop per meter indicating flooding [Pa/m]
ECKERT_MAX_DP_PER_M: Final[float] = 2000.0

# HTU clamp range [m]
HTU_MIN: Final[float] = 0.1
HTU_MAX: Final[float] = 3.0

# NaOH solution density correlation (rho = intercept + slope * wt%)
NAOH_DENSITY_INTERCEPT: Final[float] = 1000.0  # kg/m3
NAOH_DENSITY_SLOPE: Final[float] = 10.8  # kg/m3 per wt%

# Syngas default heat capacity [J/(kg-K)]
SYNGAS_CP_DEFAULT: Final[float] = 1100.0

# Cooling water approach temperature [C]
COOLING_WATER_APPROACH_TEMP: Final[float] = 5.0

# Scrubber outlet gas temperature [C]
SCRUBBER_OUTLET_GAS_TEMP: Final[float] = 38.0

# =============================================================================
# WGS REACTOR CONSTANTS
# =============================================================================
# Source: NIST-JANAF Tables; Van't Hoff equation

# WGS reaction enthalpy [J/mol] (CO + H2O -> CO2 + H2)
WGS_DELTA_H: Final[float] = -41200.0

# WGS reaction entropy [J/(mol-K)]
WGS_DELTA_S: Final[float] = -42.1

# Standard state pressure [Pa] (1 bar)
STANDARD_STATE_PRESSURE_PA: Final[float] = 100000.0

# Typical GHSV for WGS reactors [1/hr]
WGS_TYPICAL_GHSV: Final[float] = 3000.0

# Catalyst volume fraction of reactor
WGS_CATALYST_VOLUME_FRACTION: Final[float] = 0.8

# Reactor length-to-diameter ratio
WGS_REACTOR_LD_RATIO: Final[float] = 3.0

# WGS heat of reaction [kJ/mol CO]
WGS_HEAT_KJ_PER_MOL: Final[float] = 41.2

# Conversion: kJ/hr to kW
KJ_HR_TO_KW: Final[float] = 3.6

# =============================================================================
# SYNGAS COMPRESSION CONSTANTS
# =============================================================================

# Intercooler outlet temperature [K] (40 C)
INTERCOOLER_OUTLET_TEMP_K: Final[float] = 313.15

# Temperature warning thresholds [K]
COMPRESSION_TEMP_WARNING_K: Final[float] = 473.15  # 200 C
COMPRESSION_TEMP_CRITICAL_K: Final[float] = 523.15  # 250 C

# High pressure threshold [bar]
COMPRESSION_HIGH_PRESSURE_BAR: Final[float] = 100.0

# High power threshold [HP]
COMPRESSION_HIGH_POWER_HP: Final[float] = 1000.0

# Minimum acceptable compression efficiency
COMPRESSION_MIN_EFFICIENCY: Final[float] = 0.7

# Default heat capacity ratio for diatomic gases
DEFAULT_GAMMA_DIATOMIC: Final[float] = 1.4

# Upper bound for heat capacity ratio (monatomic gases)
GAMMA_UPPER_BOUND: Final[float] = 1.7

# =============================================================================
# IAPWS-IF97 CONSTANTS
# =============================================================================
# Source: IAPWS-IF97 (International Association for Properties of Water and Steam)

# Critical temperature of water [K]
IAPWS_CRITICAL_TEMP: Final[float] = 647.096

# Critical pressure of water [Pa]
IAPWS_CRITICAL_PRESSURE: Final[float] = 22064000.0

# Triple point temperature of water [K]
IAPWS_TRIPLE_POINT_TEMP: Final[float] = 273.16

# IAPWS saturation pressure correlation coefficients
IAPWS_COEFFICIENTS: Final[list[float]] = [
    -7.85951783,
    1.84408259,
    -11.7866497,
    22.6807411,
    -15.9618719,
    1.80122502,
]

# =============================================================================
# MAGNUS EQUATION CONSTANTS
# =============================================================================
# Source: Alduchov & Eskridge (1996) improved Magnus formula

# Magnus equation: P_hPa = A * exp(B * T / (T + C))
MAGNUS_A: Final[float] = 6.1094  # hPa
MAGNUS_B: Final[float] = 17.625  # dimensionless
MAGNUS_C: Final[float] = 243.04  # C

# =============================================================================
# BUCK EQUATION CONSTANTS (Above Freezing)
# =============================================================================
# Source: Buck, A.L. (1981) "New Equations for Computing Vapor Pressure"

BUCK_ABOVE_FREEZING_A: Final[float] = 0.61121  # kPa
BUCK_ABOVE_FREEZING_B: Final[float] = 18.678  # dimensionless
BUCK_ABOVE_FREEZING_C: Final[float] = 234.5  # C
BUCK_ABOVE_FREEZING_D: Final[float] = 257.14  # C

# =============================================================================
# EXTENDED ANTOINE EQUATION CONSTANTS
# =============================================================================
# Source: Perry's Chemical Engineers' Handbook, 8th Ed.

# Water (high temperature range, 100-374 C)
ANTOINE_WATER_HIGH_A: Final[float] = 8.14019
ANTOINE_WATER_HIGH_B: Final[float] = 1810.94
ANTOINE_WATER_HIGH_C: Final[float] = 244.485

# Hydrogen Fluoride
ANTOINE_HF_A: Final[float] = 7.158
ANTOINE_HF_B: Final[float] = 1111.0
ANTOINE_HF_C: Final[float] = 235.0

# Hydrogen Chloride
ANTOINE_HCL_A: Final[float] = 7.960
ANTOINE_HCL_B: Final[float] = 1118.0
ANTOINE_HCL_C: Final[float] = 240.0

# Hydrogen Sulfide
ANTOINE_H2S_A: Final[float] = 6.987
ANTOINE_H2S_B: Final[float] = 884.0
ANTOINE_H2S_C: Final[float] = 240.0

# =============================================================================
# BAGHOUSE CALCULATOR CONSTANTS
# =============================================================================
# Source: NIST-JANAF Thermochemical Tables (Cp at ~500 K)

# Molar heat capacities at ~500 K [J/(mol-K)]
CP_H2_500K: Final[float] = 29.1
CP_CO_500K: Final[float] = 29.2
CP_CO2_500K: Final[float] = 41.3
CP_H2O_500K: Final[float] = 35.5
CP_N2_500K: Final[float] = 29.5
CP_CH4_500K: Final[float] = 44.5
CP_O2_500K: Final[float] = 30.1
CP_AR_500K: Final[float] = 20.8

# Default Cp fallback [J/(mol-K)]
CP_DEFAULT_FALLBACK: Final[float] = 30.0

# Molecular weights for baghouse ideal gas calcs [kg/mol]
MW_H2_KG: Final[float] = 0.002
MW_CO_KG: Final[float] = 0.028
MW_CO2_KG: Final[float] = 0.044
MW_H2O_KG: Final[float] = 0.018
MW_N2_KG: Final[float] = 0.028
MW_CH4_KG: Final[float] = 0.016
MW_O2_KG: Final[float] = 0.032
MW_AR_KG: Final[float] = 0.040

# Default molecular weight fallback [kg/mol]
MW_DEFAULT_KG: Final[float] = 0.028

# Default Cp mass fallback [J/(kg-K)]
CP_MASS_DEFAULT_FALLBACK: Final[float] = 1000.0

# =============================================================================
# SYNGAS WATER CALCULATOR CONSTANTS
# =============================================================================

# Water molecular weight [g/mol]
MW_WATER_GMOL: Final[float] = 18.015

# Typical syngas average MW [g/mol]
MW_SYNGAS_TYPICAL_GMOL: Final[float] = 15.0

# Gas constant for density calcs [J/(kmol-K)] (= R * 1000)
R_GAS_DENSITY: Final[float] = 8314.46

# Normal conditions for gas volume
NORMAL_PRESSURE_PA: Final[float] = 101325.0
NORMAL_TEMPERATURE_K: Final[float] = 273.15

# =============================================================================
# SIGNAL TOOLKIT CONSTANTS
# =============================================================================

# Default line frequency for periodic noise [Hz]
DEFAULT_LINE_FREQUENCY_HZ: Final[float] = 60.0

# Periodic noise harmonic amplitudes (fraction of fundamental)
PERIODIC_NOISE_2ND_HARMONIC: Final[float] = 0.3
PERIODIC_NOISE_3RD_HARMONIC: Final[float] = 0.1
