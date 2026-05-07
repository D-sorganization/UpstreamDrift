"""Drake motion-matching package.

Implements the cross-engine parity contracts (cross-engine §2.2-§2.6)
for Drake. See ``DRAKE_PARITY_SPEC.md`` for the architecture overview.
"""

from __future__ import annotations

from .compute_cost_drake import compute_cost_drake, drake_simout_to_shared
from .fit_swing import fit_swing_drake, polynomial_parameter_bounds
from .fit_swing_autodiff import (
    DEFAULT_COEFFICIENT_BOUNDS,
    FitOptions,
    FitResult,
    default_theta_bounds,
    fit_swing_drake_autodiff,
)

# Note: ``fit_swing.FitOptions`` / ``FitResult`` shadow the autodiff variants
# above when imported directly from ``.fit_swing``; the top-level package
# re-exports the autodiff dataclasses for backwards compatibility.
from .humanoid_urdf import (
    CANONICAL_URDF,
    SHARED_DIMENSIONS_YAML,
    build_humanoid_urdf,
    load_humanoid_dimensions,
    load_humanoid_into_plant,
)
from .simulate import (
    COEFFS_PER_JOINT,
    SimOptions,
    SimOut,
    evaluate_torque_polynomial,
    simulate_with_coefficients,
)

__all__ = [
    "CANONICAL_URDF",
    "COEFFS_PER_JOINT",
    "DEFAULT_COEFFICIENT_BOUNDS",
    "FitOptions",
    "FitResult",
    "SHARED_DIMENSIONS_YAML",
    "SimOptions",
    "SimOut",
    "build_humanoid_urdf",
    "compute_cost_drake",
    "default_theta_bounds",
    "drake_simout_to_shared",
    "evaluate_torque_polynomial",
    "fit_swing_drake",
    "fit_swing_drake_autodiff",
    "load_humanoid_dimensions",
    "load_humanoid_into_plant",
    "polynomial_parameter_bounds",
    "simulate_with_coefficients",
]
