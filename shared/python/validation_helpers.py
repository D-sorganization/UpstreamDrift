"""Physical plausibility validation helpers for scientific computing.

This module provides utilities to validate physics inputs (positions, velocities,
accelerations) for physical plausibility before performing computations.

DESIGN PHILOSOPHY:
------------------
"Fail fast with meaningful errors" - Detect invalid inputs early before they
propagate through complex computations and produce misleading results.

VALIDATION LEVELS:
------------------
1. **Mathematical validity**: NaN, Inf checks
2. **Physical plausibility**: Magnitude bounds based on human biomechanics
3. **Numerical stability**: Warn on values approaching precision limits

USAGE:
------
```python
from shared.python.validation_helpers import validate_joint_state, ValidationLevel

# Strict validation (raises on any issue)
validate_joint_state(qpos, q

vel, qacc, level=ValidationLevel.STRICT)

# Permissive validation (warns but continues)
validate_joint_state(qpos, qvel, qacc, level=ValidationLevel.PERMISSIVE)
```

REFERENCES:
-----------
- Human biomechanics literature (Zatsiorsky, Winter)
- IEEE 754 floating point standards
- Assessment C-005: Input validation recommendations
"""

import warnings
from enum import Enum

import numpy as np


class ValidationLevel(Enum):
    """Validation strictness levels."""

    PERMISSIVE = "permissive"  # Warn only
    STANDARD = "standard"  # Raise on critical issues, warn on others
    STRICT = "strict"  # Raise on any issue


class PhysicsValidationError(ValueError):
    """Raised when physics inputs fail validation."""

    pass


# Physical plausibility bounds (based on human biomechanics)
# Source: Zatsiorsky "Kinetics of Human Motion" (2002), Winter "Biomechanics" (2009)

MAX_JOINT_VELOCITY_RAD_S = 50.0  # rad/s (extremely fast, e.g., baseball pitch)
"""Maximum plausible joint angular velocity [rad/s].

RATIONALE:
- Typical human joint velocities: 1-10 rad/s
- Elite athletes (e.g., pitching, golf swing): 20-40 rad/s
- 50 rad/s provides safety margin above max observed

SOURCE: Feltner & Dapena "Baseball pitching" (1989) - shoulder: ~40 rad/s
"""

MAX_JOINT_ACCELERATION_RAD_S2 = 1000.0  # rad/s²
"""Maximum plausible joint angular acceleration [rad/s²].

RATIONALE:
- Typical accelerations: 10-100 rad/s²
- Impact/collision events: 100-500 rad/s²
- 1000 rad/s² is extreme but physically possible

SOURCE: Winter "Biomechanics", Table 4.2
"""

MAX_JOINT_POSITION_RAD = 2 * np.pi  # rad (full rotation)
"""Maximum reasonable joint position magnitude [rad].

RATIONALE:
- Most human joints have limited ROM (0.5-2.5 rad)
- Some joints (shoulder, hip) can approach ±π
- 2π allows for continuous rotations in some models

NOTE: This is a "reasonability" check, not a hard anatomical limit.
"""

MAX_CARTESIAN_VELOCITY_M_S = 100.0  # m/s
"""Maximum plausible Cartesian velocity [m/s].

RATIONALE:
- Human body segments rarely exceed 30 m/s (golf club head)
- 100 m/s provides safety margin

SOURCE: Golf swing literature - club head: 50-60 m/s max
"""

from shared.python.constants import GRAVITY_M_S2

MAX_CARTESIAN_ACCELERATION_M_S2 = 10000.0  # m/s² (~1000g)
"""Maximum plausible Cartesian acceleration [m/s²].

RATIONALE:
- Gravity: {GRAVITY_M_S2} m/s²
- Typical accelerations: 10-100 m/s²
- Impact events: 100-1000 m/s² (10-100g)
- 10000 m/s² (1000g) is extreme but possible in collisions

SOURCE: Biomechanics literature on impacts
"""


def validate_finite(
    array: np.ndarray, name: str, level: ValidationLevel = ValidationLevel.STANDARD
) -> None:
    """Validate that array contains only finite values (no NaN or Inf).

    Args:
        array: NumPy array to validate
        name: Name of the array (for error messages)
        level: Validation strictness level

    Raises:
        PhysicsValidationError: If array contains NaN/Inf (STRICT or STANDARD)

    Warns:
        UserWarning: If array contains NaN/Inf (PERMISSIVE)
    """
    if not np.all(np.isfinite(array)):
        message = (
            f"{name} contains NaN or Inf values. "
            f"This indicates numerical instability or uninitialized data. "
            f"Non-finite values: {np.sum(~np.isfinite(array))}/{array.size}"
        )

        if level == ValidationLevel.PERMISSIVE:
            warnings.warn(message, category=UserWarning, stacklevel=3)
        else:
            raise PhysicsValidationError(message)


def validate_magnitude(
    array: np.ndarray,
    name: str,
    max_value: float,
    units: str,
    level: ValidationLevel = ValidationLevel.STANDARD,
) -> None:
    """Validate that array values don't exceed physical plausibility bounds.

    Args:
        array: NumPy array to validate
        name: Name of the array (for error messages)
        max_value: Maximum plausible value
        units: Units string (for error messages)
        level: Validation strictness level

    Raises:
        PhysicsValidationError: If values exceed bounds (STRICT)

    Warns:
        UserWarning: If values exceed bounds (STANDARD or PERMISSIVE)
    """
    max_observed = np.max(np.abs(array))

    if max_observed > max_value:
        message = (
            f"{name} contains implausibly large values: "
            f"max={max_observed:.2e} {units} (threshold: {max_value:.2e} {units}). "
            f"This may indicate unit errors (e.g., mm vs m) or simulation instability."
        )

        if level == ValidationLevel.STRICT:
            raise PhysicsValidationError(message)
        else:
            warnings.warn(message, category=UserWarning, stacklevel=3)


def validate_joint_state(
    qpos: np.ndarray,
    qvel: np.ndarray | None = None,
    qacc: np.ndarray | None = None,
    level: ValidationLevel = ValidationLevel.STANDARD,
) -> None:
    """Validate joint positions, velocities, and accelerations for plausibility.

    This function checks:
    1. Mathematical validity (no NaN/Inf)
    2. Physical plausibility (magnitude bounds)
    3. Dimensional consistency

    Args:
        qpos: Joint positions [nv] (rad for revolute, m for prismatic)
        qvel: Optional joint velocities [nv] (rad/s or m/s)
        qacc: Optional joint accelerations [nv] (rad/s² or m/s²)
        level: Validation strictness level

    Raises:
        PhysicsValidationError: If validation fails at the specified level

    Warns:
        UserWarning: For plausibility issues at lower strictness levels

    Examples:
        >>> import numpy as np
        >>> qpos = np.array([0.5, -0.3])  # Valid
        >>> qvel = np.array([1.0, 2.0])    # Valid
        >>> validate_joint_state(qpos, qvel)  # Passes
        >>>
        >>> qpos_bad = np.array([np.nan, 0.0])
        >>> validate_joint_state(qpos_bad)  # Raises PhysicsValidationError
    """
    # Check dimensional consistency
    nv = len(qpos)

    if qvel is not None and len(qvel) != nv:
        raise PhysicsValidationError(
            f"Dimension mismatch: qpos has {nv} DOF, qvel has {len(qvel)} DOF"
        )

    if qacc is not None and len(qacc) != nv:
        raise PhysicsValidationError(
            f"Dimension mismatch: qpos has {nv} DOF, qacc has {len(qacc)} DOF"
        )

    # Validate finite values
    validate_finite(qpos, "qpos", level)

    if qvel is not None:
        validate_finite(qvel, "qvel", level)

    if qacc is not None:
        validate_finite(qacc, "qacc", level)

    # Validate magnitudes (assuming revolute joints - most common)
    validate_magnitude(qpos, "qpos", MAX_JOINT_POSITION_RAD, "rad", level)

    if qvel is not None:
        validate_magnitude(qvel, "qvel", MAX_JOINT_VELOCITY_RAD_S, "rad/s", level)

    if qacc is not None:
        validate_magnitude(qacc, "qacc", MAX_JOINT_ACCELERATION_RAD_S2, "rad/s²", level)


def validate_cartesian_state(
    position: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    acceleration: np.ndarray | None = None,
    level: ValidationLevel = ValidationLevel.STANDARD,
) -> None:
    """Validate Cartesian positions, velocities, and accelerations.

    Args:
        position: Optional Cartesian position [3] or [6] (m, or m+rad for pose)
        velocity: Optional Cartesian velocity [3] or [6] (m/s, or m/s+rad/s)
        acceleration: Optional Cartesian acceleration [3] or [6] (m/s² or mixed)
        level: Validation strictness level

    Raises:
        PhysicsValidationError: If validation fails

    Warns:
        UserWarning: For plausibility issues
    """
    if position is not None:
        validate_finite(position, "Cartesian position", level)

    if velocity is not None:
        validate_finite(velocity, "Cartesian velocity", level)
        # Check only translational part (first 3 elements)
        validate_magnitude(
            velocity[:3], "Cartesian velocity", MAX_CARTESIAN_VELOCITY_M_S, "m/s", level
        )

    if acceleration is not None:
        validate_finite(acceleration, "Cartesian acceleration", level)
        validate_magnitude(
            acceleration[:3],
            "Cartesian acceleration",
            MAX_CARTESIAN_ACCELERATION_M_S2,
            "m/s²",
            level,
        )


def validate_model_parameters(
    body_masses: np.ndarray,
    level: ValidationLevel = ValidationLevel.STANDARD,
) -> None:
    """Validate model parameters (masses, inertias) for physical plausibility.

    Args:
        body_masses: Body masses [nbody] (kg)
        level: Validation strictness level

    Raises:
        PhysicsValidationError: If masses are non-positive or implausible

    Examples:
        >>> masses = np.array([70.0, 5.0, 2.0])  # Reasonable
        >>> validate_model_parameters(masses)  # Passes
    """
    validate_finite(body_masses, "body_masses", level)

    # Masses must be strictly positive
    if np.any(body_masses <= 0):
        min_mass = np.min(body_masses)
        raise PhysicsValidationError(
            f"Body masses must be positive. Found min mass: {min_mass:.2e} kg"
        )

    # Check total mass plausibility (should be reasonable for humanoid)
    total_mass = np.sum(body_masses)

    # Skip total mass check if only a few bodies (might be partial model)
    if len(body_masses) > 10:
        # Human body mass typically 40-200 kg
        if total_mass < 20 or total_mass > 500:
            warnings.warn(
                f"Total model mass ({total_mass:.1f} kg) is outside typical "
                f"human range (40-200 kg). Verify model parameters and units.",
                category=UserWarning,
                stacklevel=2,
            )


# Export public API
__all__ = [
    "ValidationLevel",
    "PhysicsValidationError",
    "validate_finite",
    "validate_magnitude",
    "validate_joint_state",
    "validate_cartesian_state",
    "validate_model_parameters",
    # Constants (for reference)
    "MAX_JOINT_VELOCITY_RAD_S",
    "MAX_JOINT_ACCELERATION_RAD_S2",
    "MAX_CARTESIAN_VELOCITY_M_S",
    "MAX_CARTESIAN_ACCELERATION_M_S2",
]
