"""Physical and mathematical constants with citations.

All constants must include:
1. Value with appropriate precision
2. Units in square brackets
3. Source citation

This module provides physical constants, conversion factors, and material
properties used in scientific computing applications.
"""

import math
from pathlib import Path

# Mathematical constants
PI: float = math.pi  # [dimensionless] Ratio of circumference to diameter
E: float = (
    2.718281828459045  # [dimensionless] Euler's number, base of natural logarithm
)
PI_HALF: float = math.pi / 2  # [dimensionless] π/2
PI_QUARTER: float = math.pi / 4  # [dimensionless] π/4

# Spatial algebra constants
SPATIAL_DIM: int = (
    6  # [dimensionless] Dimension of spatial vectors (3 linear + 3 angular)
)
SPATIAL_LIN_DIM: int = 3  # [dimensionless] Linear dimension (position/velocity)
SPATIAL_ANG_DIM: int = (
    3  # [dimensionless] Angular dimension (orientation/angular velocity)
)

# Physical constants - SI units
GRAVITY_M_S2: float = 9.80665  # [m/s²] Standard gravity, ISO 80000-3:2006
SPEED_OF_LIGHT_M_S: float = 299792458  # [m/s] Exact by definition, SI
AIR_DENSITY_SEA_LEVEL_KG_M3: float = 1.225  # [kg/m³] ISA at sea level, 15°C
ATMOSPHERIC_PRESSURE_PA: float = 101325  # [Pa] Standard atmospheric pressure, ISA
UNIVERSAL_GAS_CONSTANT_J_MOL_K: float = 8.314462618  # [J/(mol·K)] CODATA 2018

# Golf-specific constants
GOLF_BALL_MASS_KG: float = 0.04593  # [kg] USGA Rule 5-1 (1.620 oz max)
GOLF_BALL_DIAMETER_M: float = 0.04267  # [m] USGA Rule 5-2 (1.680 in min)
# Smooth ball at Re=150,000 per Bearman & Harvey 1976
GOLF_BALL_DRAG_COEFFICIENT: float = 0.25  # [dimensionless]
GOLF_BALL_LIFT_COEFFICIENT: float = (
    0.15  # [dimensionless] Typical for golf ball dimples
)

# Club specifications
DRIVER_LENGTH_MAX_M: float = 1.1684  # [m] USGA Rule 1-1c (46 inches)
DRIVER_LOFT_TYPICAL_DEG: float = 10.5  # [degrees] Modern driver average
IRON_LOFT_RANGE_DEG: tuple[float, float] = (18.0, 64.0)  # [degrees] 2-iron to lob wedge
IRON_7_LOFT_DEG: float = 34.0  # [degrees] Standard 7-iron loft
PUTTER_LOFT_DEG: float = 3.0  # [degrees] Standard putter loft

# Course conditions
GREEN_SPEED_STIMP: float = 10.0  # [ft] Fast green speed
ROUGH_HEIGHT_MM: float = 25.0  # [mm] Medium rough height
BUNKER_DEPTH_MM: float = 100.0  # [mm] Standard bunker depth

# Atmospheric conditions (Standard)
TEMPERATURE_C: float = 20.0  # [°C] Standard temperature
PRESSURE_HPA: float = 1013.25  # [hPa] Standard atmospheric pressure
HUMIDITY_PERCENT: float = 50.0  # [%] Standard relative humidity

# Conversion factors (exact)
MPS_TO_KPH: float = 3.6  # [m/s → km/h] = (1000m/1km)/(3600s/1h)
MPS_TO_MPH: float = 2.23694  # [m/s → mph] = 3600/1609.344
DEG_TO_RAD: float = 0.017453292519943295  # [degrees → radians] = π/180
RAD_TO_DEG: float = 57.29577951308232  # [radians → degrees] = 180/π
KG_TO_LB: float = 2.20462262185  # [kg → lb] Exact conversion
M_TO_FT: float = 3.28084  # [m → ft] Exact: 1m = 100cm/(2.54cm/in)/(12in/ft)
M_TO_YARD: float = 1.09361  # [m → yard] Exact: 1m = 100cm/(2.54cm/in)/(36in/yard)

# Material properties
GRAPHITE_DENSITY_KG_M3: float = 1750  # [kg/m³] Typical golf shaft graphite
STEEL_DENSITY_KG_M3: float = 7850  # [kg/m³] Carbon steel
TITANIUM_DENSITY_KG_M3: float = 4506  # [kg/m³] Ti-6Al-4V alloy
ALUMINUM_DENSITY_KG_M3: float = 2700  # [kg/m³] 6061-T6 aluminum

# Aerodynamic coefficients
MAGNUS_COEFFICIENT: float = (
    0.25  # [dimensionless] Typical for golf ball, per Bearman & Harvey
)
SPIN_DECAY_RATE_S: float = 0.05  # [1/s] Typical spin decay, per Trackman data
AIR_VISCOSITY_KG_M_S: float = 1.789e-5  # [kg/(m·s)] Dynamic viscosity at 15°C

# Numerical constants
EPSILON: float = 1e-15  # [dimensionless] Machine epsilon for double precision
MAX_ITERATIONS: int = 10000  # [dimensionless] Maximum iterations for numerical methods
CONVERGENCE_TOLERANCE: float = 1e-6  # [dimensionless] Default convergence tolerance

# Reproducibility
DEFAULT_RANDOM_SEED: int = 42  # [dimensionless] Answer to everything

# Project Config
# Relative paths from repository root
MUJOCO_LAUNCHER_SCRIPT: Path = Path(
    "engines/physics_engines/mujoco/python/mujoco_humanoid_golf/advanced_gui.py"
)
DRAKE_LAUNCHER_SCRIPT: Path = Path(
    "engines/physics_engines/drake/python/src/golf_gui.py"
)
PINOCCHIO_LAUNCHER_SCRIPT: Path = Path(
    "engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py"
)
URDF_GENERATOR_SCRIPT: Path = Path("tools/urdf_generator/main.py")
C3D_VIEWER_SCRIPT: Path = Path(
    "engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py"
)
GUI_LAUNCHER_SCRIPT: Path = Path("launchers/golf_launcher.py")
LOCAL_LAUNCHER_SCRIPT: Path = Path("launchers/golf_suite_launcher.py")
