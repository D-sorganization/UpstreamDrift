"""Input validation utilities for physics engine parameters.

Assessment A Finding F-005 / API Safety Implementation

This module provides validators to prevent physically impossible parameters
from being set in physics engines, catching errors at API boundaries before
they propagate into simulation.

Physical validity checks:
- Mass > 0 (no negative or zero mass)
- Inertia matrix positive definite (physically realizable)
- Timestep > 0 (no zero or negative timesteps)
- Joint limits: q_min < q_max
- Friction coefficients >= 0
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

from ..core.error_utils import ValidationError

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class PhysicalValidationError(ValidationError):
    """Raised when a physical parameter fails validation.

    This class extends the base ValidationError for physics-specific validation.
    It can be initialized with either a simple message string (for backward
    compatibility) or with structured field/value/constraint parameters.
    """

    def __init__(
        self,
        message_or_field: str,
        value: Any = None,
        physical_constraint: str | None = None,
    ) -> None:
        # Support both old-style (single message) and new-style (field, value, constraint)
        if value is None and physical_constraint is None:
            # Old-style: message_or_field is the full error message
            super().__init__(
                field="physical_parameter",
                value=None,
                reason=message_or_field,
            )
        else:
            # New-style: structured parameters
            super().__init__(
                field=message_or_field,
                value=value,
                reason=physical_constraint or "Physical constraint violated",
            )
        self.physical_constraint = physical_constraint


def validate_mass(mass: float, param_name: str = "mass") -> None:
    """Validate mass is positive.

    Args:
        mass: Mass value [kg]
        param_name: Parameter name for error messages

    Raises:
        PhysicalValidationError: If mass <= 0

    Example:
        >>> validate_mass(1.5)  # OK
        >>> validate_mass(-1.0)  # Raises PhysicalValidationError
    """
    if mass <= 0:
        raise PhysicalValidationError(
            f"Invalid {param_name}: {mass:.6f} kg\\n"
            f"  Physical requirement: mass > 0\\n"
            f"  Negative mass violates Newton's laws.\\n"
            f"  Zero mass creates singularities in M(q)."
        )


def validate_timestep(dt: float) -> None:
    """Validate timestep is positive and reasonable.

    Args:
        dt: Timestep [s]

    Raises:
        PhysicalValidationError: If dt <= 0 or dt > 1.0 (suspiciously large)

    Example:
        >>> validate_timestep(0.001)  # OK
        >>> validate_timestep(0.0)    # Raises PhysicalValidationError
        >>> validate_timestep(2.0)    # Raises PhysicalValidationError (too large)
    """
    if dt <= 0:
        raise PhysicalValidationError(
            f"Invalid timestep: {dt:.6e} s\\n"
            f"  Physical requirement: dt > 0\\n"
            f"  Zero or negative timestep is nonsensical."
        )

    if dt > 1.0:
        logger.warning(
            f"⚠️ Suspiciously large timestep: {dt:.3f} s\\n"
            f"  Typical biomechanics timesteps: 0.0001 - 0.01 s\\n"
            f"  Large timesteps cause integration instability.\\n"
            f"  Recommendation: Use dt < 0.01 s for accuracy"
        )


def validate_inertia_matrix(
    inertia_matrix: np.ndarray, param_name: str = "inertia"
) -> None:
    """Validate inertia matrix is symmetric positive definite.

    Physical requirement: Inertia tensor must be SPD (symmetric positive definite).
    This ensures kinetic energy KE = 0.5·ω^T·I·ω > 0 for all ω ≠ 0.

    Args:
        inertia_matrix: Inertia matrix (3×3) [kg·m²]
        param_name: Parameter name for error messages

    Raises:
        PhysicalValidationError: If matrix is not SPD

    Example:
        >>> inertia = np.diag([1.0, 2.0, 3.0])  # Valid diagonal inertia
        >>> validate_inertia_matrix(inertia)  # OK
        >>> I_bad = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])  # Negative eigenvalue
        >>> validate_inertia_matrix(I_bad)  # Raises PhysicalValidationError
    """
    if inertia_matrix.shape != (3, 3):
        raise PhysicalValidationError(
            f"Invalid {param_name} shape: {inertia_matrix.shape}\\n"
            f"  Expected: (3, 3) for 3D inertia tensor"
        )

    # Check symmetry
    if not np.allclose(inertia_matrix, inertia_matrix.T, atol=1e-10):
        asymmetry = np.max(np.abs(inertia_matrix - inertia_matrix.T))
        raise PhysicalValidationError(
            f"Inertia matrix not symmetric:\\n"
            f"  Max asymmetry: {asymmetry:.2e}\\n"
            f"  Physical inertia tensors must be symmetric."
        )

    # Check positive definiteness via eigenvalues
    eigenvalues = np.linalg.eigvalsh(
        inertia_matrix
    )  # Hermitian eigenvalues (faster for symmetric)

    if np.any(eigenvalues <= 0):
        min_eig = eigenvalues.min()
        raise PhysicalValidationError(
            f"Inertia matrix not positive definite:\\n"
            f"  Smallest eigenvalue: {min_eig:.2e}\\n"
            f"  Eigenvalues: {eigenvalues}\\n"
            f"  Physical requirement: All eigenvalues > 0\\n"
            f"  Non-PD inertia → negative kinetic energy (impossible)"
        )


def validate_joint_limits(
    q_min: np.ndarray, q_max: np.ndarray, param_name: str = "joint_limits"
) -> None:
    """Validate joint limits are consistent (q_min < q_max).

    Args:
        q_min: Lower joint limits [rad or m]
        q_max: Upper joint limits [rad or m]
        param_name: Parameter name for error messages

    Raises:
        PhysicalValidationError: If any q_min[i] >= q_max[i]

    Example:
        >>> q_min = np.array([0, -π])
        >>> q_max = np.array([π, π])
        >>> validate_joint_limits(q_min, q_max)  # OK
        >>> validate_joint_limits(q_max, q_min)  # Raises (reversed)
    """
    if q_min.shape != q_max.shape:
        raise PhysicalValidationError(
            f"Invalid {param_name}: shape mismatch\\n"
            f"  q_min shape: {q_min.shape}\\n"
            f"  q_max shape: {q_max.shape}"
        )

    violations = q_min >= q_max
    if np.any(violations):
        indices = np.where(violations)[0]
        raise PhysicalValidationError(
            f"Invalid {param_name}: q_min >= q_max at indices {indices}:\\n"
            f"  q_min[{indices}] = {q_min[indices]}\\n"
            f"  q_max[{indices}] = {q_max[indices]}\\n"
            f"  Physical requirement: q_min < q_max for all joints"
        )


def validate_friction_coefficient(mu: float, param_name: str = "friction") -> None:
    """Validate friction coefficient is non-negative.

    Args:
        mu: Friction coefficient [dimensionless]
        param_name: Parameter name for error messages

    Raises:
        PhysicalValidationError: If mu < 0

    Example:
        >>> validate_friction_coefficient(0.5)  # OK
        >>> validate_friction_coefficient(-0.1)  # Raises
    """
    if mu < 0:
        raise PhysicalValidationError(
            f"Invalid {param_name}: {mu:.6f}\\n"
            f"  Physical requirement: friction coefficient >= 0\\n"
            f"  Negative friction would add energy to system."
        )


def validate_physical_bounds(func: F) -> F:
    """Decorator to validate physical parameters at API boundaries.

    Automatically validates common physics parameters based on naming conventions:
    - 'mass' or '*_mass': Must be > 0
    - 'dt' or 'timestep': Must be > 0
    - 'inertia' or '*_inertia': Must be symmetric positive definite (3×3)
    - 'q_min', 'q_max': Must satisfy q_min < q_max
    - 'friction' or '*_friction': Must be >= 0

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with automatic validation

    Example:
        >>> @validate_physical_bounds
        ... def set_mass(self, body_name: str, mass: float) -> None:
        ...     self._masses[body_name] = mass
        >>> engine.set_mass("link1", 1.5)  # OK
        >>> engine.set_mass("link1", -1.0)  # Raises PhysicalValidationError
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Validate physical parameters before calling the wrapped function."""
        import inspect

        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        def _validate_param(param_name: str, param_value: Any) -> None:
            """Apply all physical validation rules to a single named parameter."""
            assert param_name is not None, "param_name must be provided"
            assert param_name is not None, "param_name must be provided"
            if param_name in ("self", "cls"):
                return

            # Mass validation
            if "mass" in param_name.lower() and isinstance(param_value, int | float):
                validate_mass(float(param_value), param_name)

            # Timestep validation
            if param_name in ("dt", "timestep") and isinstance(
                param_value, int | float
            ):
                validate_timestep(float(param_value))

            # Inertia validation
            if (
                "inertia" in param_name.lower()
                and isinstance(param_value, np.ndarray)
                and param_value.shape == (3, 3)
            ):
                validate_inertia_matrix(param_value, param_name)

            # Friction validation
            if "friction" in param_name.lower() and isinstance(
                param_value, int | float
            ):
                validate_friction_coefficient(float(param_value), param_name)

        # Validate parameters by name — normal and **kwargs parameters
        for param_name, param_value in bound_args.arguments.items():
            # Detect VAR_KEYWORD parameters (i.e. **kwargs in the function signature)
            # For these, param_value is a dict — iterate its contents too.
            param_obj = sig.parameters.get(param_name)
            if (
                param_obj is not None
                and param_obj.kind == inspect.Parameter.VAR_KEYWORD
                and isinstance(param_value, dict)
            ):
                for kw_name, kw_value in param_value.items():
                    _validate_param(kw_name, kw_value)
            else:
                _validate_param(param_name, param_value)

        # Joint limits validation (needs both q_min and q_max together)
        all_params: dict[str, Any] = {}
        for param_name, param_value in bound_args.arguments.items():
            param_obj = sig.parameters.get(param_name)
            if (
                param_obj is not None
                and param_obj.kind == inspect.Parameter.VAR_KEYWORD
                and isinstance(param_value, dict)
            ):
                all_params.update(param_value)
            else:
                all_params[param_name] = param_value

        if "q_min" in all_params and "q_max" in all_params:
            q_min, q_max = all_params["q_min"], all_params["q_max"]
            if isinstance(q_min, np.ndarray) and isinstance(q_max, np.ndarray):
                validate_joint_limits(q_min, q_max)

        return func(*args, **kwargs)

    return wrapper  # type: ignore


# Example usage for documentation
if __name__ == "__main__":
    import doctest

    doctest.testmod()
