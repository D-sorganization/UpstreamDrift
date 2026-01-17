"""Physical and mathematical constants with citations.

All constants must include:
1. Value with appropriate precision
2. Units in square brackets
3. Source citation
"""

import math


class PhysicalConstant(float):
    """A float subclass that carries physical unit and provenance metadata."""

    def __new__(
        cls, value: float, unit: str, source: str, description: str = ""
    ) -> "PhysicalConstant":
        return float.__new__(cls, value)

    def __init__(
        self, value: float, unit: str, source: str, description: str = ""
    ) -> None:
        self.unit = unit
        self.source = source
        self.description = description

    def __repr__(self) -> str:
        return f"PhysicalConstant({float(self):.4g}, unit='{self.unit}')"


# Mathematical constants
PI: float = math.pi  # [dimensionless] Ratio of circumference to diameter
E: float = math.e  # [dimensionless] Euler's number
PI_HALF: float = math.pi / 2
PI_QUARTER: float = math.pi / 4

# Spatial algebra constants
SPATIAL_DIM: int = 6
SPATIAL_LIN_DIM: int = 3
SPATIAL_ANG_DIM: int = 3

# Physical constants - SI units
GRAVITY_M_S2 = PhysicalConstant(
    9.80665, "m/s^2", "NIST CODATA 2018", "Standard gravity"
)
SPEED_OF_LIGHT_M_S = PhysicalConstant(
    299792458, "m/s", "SI Definition", "Speed of light in vacuum"
)
AIR_DENSITY_SEA_LEVEL_KG_M3 = PhysicalConstant(
    1.225, "kg/m^3", "ISA Standard Atmosphere", "Air density at sea level, 15C"
)
ATMOSPHERIC_PRESSURE_PA = PhysicalConstant(
    101325, "Pa", "ISA Standard Atmosphere", "Standard atmospheric pressure"
)
UNIVERSAL_GAS_CONSTANT_J_MOL_K = PhysicalConstant(
    8.314462618, "J/(mol K)", "CODATA 2018", "Molar gas constant"
)

# Golf-specific constants
GOLF_BALL_MASS_KG = PhysicalConstant(
    0.04593, "kg", "USGA Rule 5-1", "Maximum golf ball mass (1.620 oz)"
)
GOLF_BALL_DIAMETER_M = PhysicalConstant(
    0.04267, "m", "USGA Rule 5-2", "Minimum golf ball diameter (1.680 in)"
)
GOLF_BALL_DRAG_COEFFICIENT = PhysicalConstant(
    0.25,
    "dimensionless",
    "Bearman & Harvey 1976",
    "Typical drag coefficient at Re=1.5e5",
)
GOLF_BALL_LIFT_COEFFICIENT = PhysicalConstant(
    0.15, "dimensionless", "Bearman & Harvey 1976", "Typical lift coefficient"
)

# Club specifications
DRIVER_LENGTH_MAX_M = PhysicalConstant(
    1.1684, "m", "USGA Rule 1-1c", "Max driver length (46 inches)"
)
DRIVER_LOFT_TYPICAL_DEG = PhysicalConstant(
    10.5, "degrees", "Modern Trade Average", "Typical driver loft"
)
IRON_LOFT_RANGE_DEG = (
    PhysicalConstant(
        18.0, "degrees", "Modern Trade Range", "Minimum typical iron loft"
    ),
    PhysicalConstant(
        64.0, "degrees", "Modern Trade Range", "Maximum typical iron loft"
    ),
)
IRON_7_LOFT_DEG = PhysicalConstant(
    34.0, "degrees", "Modern Trade Average", "Standard 7-iron loft"
)
PUTTER_LOFT_DEG = PhysicalConstant(3.0, "degrees", "Standard", "Standard putter loft")

# Course conditions
GREEN_SPEED_STIMP = PhysicalConstant(10.0, "ft", "USGA Stimpmeter", "Fast green speed")
ROUGH_HEIGHT_MM = PhysicalConstant(
    25.0, "mm", "Standard Maintenance", "Medium rough height"
)
BUNKER_DEPTH_MM = PhysicalConstant(
    100.0, "mm", "Standard Construction", "Standard bunker depth"
)

# Atmospheric conditions (Standard)
TEMPERATURE_C = PhysicalConstant(20.0, "C", "Standard", "Standard temperature")
PRESSURE_HPA = PhysicalConstant(
    1013.25, "hPa", "Standard", "Standard atmospheric pressure"
)
HUMIDITY_PERCENT = PhysicalConstant(50.0, "%", "Standard", "Standard relative humidity")

# Conversion factors (exact)
MPS_TO_KPH = PhysicalConstant(3.6, "(km/h)/(m/s)", "Exact", "m/s to km/h")
MPS_TO_MPH = PhysicalConstant(2.23694, "mph/(m/s)", "NIST", "m/s to mph")
DEG_TO_RAD = PhysicalConstant(
    math.pi / 180, "rad/deg", "Mathematical", "Degrees to radians"
)
RAD_TO_DEG = PhysicalConstant(
    180 / math.pi, "deg/rad", "Mathematical", "Radians to degrees"
)
KG_TO_LB = PhysicalConstant(2.20462262185, "lb/kg", "NIST", "Kilograms to pounds")
M_TO_FT = PhysicalConstant(3.28084, "ft/m", "NIST", "Meters to feet")
M_TO_YARD = PhysicalConstant(1.09361, "yd/m", "NIST", "Meters to yards")

# Material properties
GRAPHITE_DENSITY_KG_M3 = PhysicalConstant(
    1750, "kg/m^3", "Materials Handbook", "Typical golf shaft graphite"
)
STEEL_DENSITY_KG_M3 = PhysicalConstant(
    7850, "kg/m^3", "Materials Handbook", "Carbon steel"
)
TITANIUM_DENSITY_KG_M3 = PhysicalConstant(
    4506, "kg/m^3", "Materials Handbook", "Ti-6Al-4V alloy"
)
ALUMINUM_DENSITY_KG_M3 = PhysicalConstant(
    2700, "kg/m^3", "Materials Handbook", "6061-T6 aluminum"
)

# Aerodynamic coefficients
MAGNUS_COEFFICIENT = PhysicalConstant(
    0.25, "dimensionless", "Bearman & Harvey", "Magnus effect coefficient"
)
SPIN_DECAY_RATE_S = PhysicalConstant(0.05, "1/s", "Trackman Data", "Spin decay rate")
AIR_VISCOSITY_KG_M_S = PhysicalConstant(
    1.789e-5, "kg/(m s)", "ISO 2533", "Dynamic viscosity of air at 15C"
)
