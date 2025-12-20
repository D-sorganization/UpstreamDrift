"""Physical and mathematical constants for MuJoCo golf swing analysis.

This module defines standard constants used throughout the codebase
to avoid magic numbers and ensure consistency.
"""

import math

# Physical constants (NIST reference values)
GRAVITY_STANDARD_M_S2: float = 9.80665  # Standard gravity [m/s²], NIST reference

# Mathematical constants
PI: float = math.pi  # π constant [dimensionless]
PI_HALF: float = math.pi / 2  # π/2 constant [dimensionless]
PI_QUARTER: float = math.pi / 4  # π/4 constant [dimensionless]

# Spatial algebra constants
SPATIAL_DIM: int = 6  # Dimension of spatial vectors (3 linear + 3 angular)
SPATIAL_LIN_DIM: int = 3  # Linear dimension (position/velocity)
SPATIAL_ANG_DIM: int = 3  # Angular dimension (orientation/angular velocity)
