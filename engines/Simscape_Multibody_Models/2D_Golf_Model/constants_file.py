"""Physical and mathematical constants with citations.

All constants must include:
1. Value with appropriate precision
2. Units in square brackets
3. Source citation
"""

import math

# Mathematical constants
PI: float = math.pi  # Ratio of circumference to diameter
E: float = math.e    # Euler's number

# Physical constants - SI units
GRAVITY_M_S2: float = 9.80665  # [m/s²] Standard gravity, ISO 80000-3:2006
SPEED_OF_LIGHT_M_S: float = 299792458  # [m/s] Exact by definition, SI
AIR_DENSITY_SEA_LEVEL_KG_M3: float = 1.225  # [kg/m³] ISA at sea level, 15°C

# Golf-specific constants
GOLF_BALL_MASS_KG: float = 0.04593  # [kg] USGA Rule 5-1 (1.620 oz max)
GOLF_BALL_DIAMETER_M: float = 0.04267  # [m] USGA Rule 5-2 (1.680 in min)
# Smooth ball at Re~150,000 per Bearman & Harvey 1976
GOLF_BALL_DRAG_COEFFICIENT: float = 0.25

# Club specifications
DRIVER_LOFT_DEG: float = 10.5  # [deg] Typical driver loft angle
IRON_7_LOFT_DEG: float = 34.0  # [deg] Standard 7-iron loft
PUTTER_LOFT_DEG: float = 3.0   # [deg] Standard putter loft

# Course conditions
GREEN_SPEED_STIMP: float = 10.0  # [ft] Fast green speed
ROUGH_HEIGHT_MM: float = 25.0    # [mm] Medium rough height
BUNKER_DEPTH_MM: float = 100.0   # [mm] Standard bunker depth

# Atmospheric conditions
TEMPERATURE_C: float = 20.0      # [°C] Standard temperature
PRESSURE_HPA: float = 1013.25    # [hPa] Standard atmospheric pressure
HUMIDITY_PERCENT: float = 50.0   # [%] Standard relative humidity


