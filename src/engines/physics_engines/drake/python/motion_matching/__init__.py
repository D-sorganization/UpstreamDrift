"""Drake motion-matching parity package.

This package implements the Drake-side of the cross-engine motion-matching
contract defined in
``src/engines/CROSS_ENGINE_PARITY_SPEC.md`` and elaborated in
``src/engines/physics_engines/drake/DRAKE_PARITY_SPEC.md``.

Issue #4108 (DRAKE-1) lands the URDF generator + loader; downstream issues
(DRAKE-2..7) consume them.
"""

from __future__ import annotations

from .fit_swing_autodiff import (
    DEFAULT_COEFFICIENT_BOUNDS,
    FitOptions,
    FitResult,
    default_theta_bounds,
    fit_swing_drake_autodiff,
)
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
    "default_theta_bounds",
    "evaluate_torque_polynomial",
    "fit_swing_drake_autodiff",
    "load_humanoid_dimensions",
    "load_humanoid_into_plant",
    "simulate_with_coefficients",
]
