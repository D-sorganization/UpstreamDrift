"""
Physical and default constants for model generation.

This module provides centralized constants used throughout
the model_generation package.
"""

from __future__ import annotations

import math

# =============================================================================
# Physical Constants
# =============================================================================

# Standard gravity (m/s^2)
GRAVITY_M_S2: float = 9.80665

# Human tissue average density (kg/m^3)
# Approximately the density of muscle tissue
TISSUE_DENSITY_KG_M3: float = 1050.0

# Water density for reference (kg/m^3)
WATER_DENSITY_KG_M3: float = 1000.0

# Bone density (kg/m^3)
BONE_DENSITY_KG_M3: float = 1900.0

# Fat tissue density (kg/m^3)
FAT_DENSITY_KG_M3: float = 900.0

# =============================================================================
# Default Values
# =============================================================================

# Default density for mesh-based inertia calculation (kg/m^3)
DEFAULT_DENSITY_KG_M3: float = TISSUE_DENSITY_KG_M3

# Default inertia value when not specified (kg*m^2)
# Suitable for small to medium rigid bodies
DEFAULT_INERTIA_KG_M2: float = 0.1

# Default minimum mass for intermediate/virtual links (kg)
# Small but non-zero to avoid numerical issues
INTERMEDIATE_LINK_MASS: float = 0.001

# =============================================================================
# Joint Defaults
# =============================================================================

# Default joint damping coefficient (N*m*s/rad)
DEFAULT_JOINT_DAMPING: float = 0.5

# Default joint friction coefficient (N*m)
DEFAULT_JOINT_FRICTION: float = 0.0

# Default maximum joint effort (N*m)
DEFAULT_JOINT_EFFORT: float = 1000.0

# Default maximum joint velocity (rad/s)
DEFAULT_JOINT_VELOCITY: float = 10.0

# =============================================================================
# Angle Limits (radians)
# =============================================================================

# Full rotation range
FULL_ROTATION_RAD: float = 2.0 * math.pi

# Common joint limit presets
JOINT_LIMIT_SMALL: float = math.radians(30)  # ±30°
JOINT_LIMIT_MEDIUM: float = math.radians(60)  # ±60°
JOINT_LIMIT_LARGE: float = math.radians(90)  # ±90°
JOINT_LIMIT_FULL: float = math.pi  # ±180°

# =============================================================================
# Humanoid Proportions (from de Leva 1996)
# =============================================================================

# Total segments in standard humanoid model
HUMANOID_SEGMENT_COUNT: int = 22

# Default height for humanoid models (m)
DEFAULT_HEIGHT_M: float = 1.75

# Default mass for humanoid models (kg)
DEFAULT_MASS_KG: float = 75.0

# =============================================================================
# Mesh Processing
# =============================================================================

# Default mesh simplification ratio for collision geometry
COLLISION_MESH_SIMPLIFICATION: float = 0.3

# Minimum faces for simplified collision mesh
MIN_COLLISION_FACES: int = 50

# Maximum faces for detailed visual mesh
MAX_VISUAL_FACES: int = 10000

# =============================================================================
# Numerical Tolerances
# =============================================================================

# Tolerance for floating point comparisons
FLOAT_TOLERANCE: float = 1e-10

# Minimum mass to consider non-zero (kg)
MIN_MASS_KG: float = 1e-6

# Minimum inertia to consider non-zero (kg*m^2)
MIN_INERTIA_KG_M2: float = 1e-12

# =============================================================================
# URDF Generation
# =============================================================================

# Default URDF indent string
URDF_INDENT: str = "  "

# XML declaration for URDF files
URDF_XML_DECLARATION: str = '<?xml version="1.0"?>'

# Default robot name
DEFAULT_ROBOT_NAME: str = "humanoid"
