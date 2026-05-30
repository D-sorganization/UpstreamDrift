"""Drake motion-matching package.

Implements the cross-engine parity contracts (cross-engine §2.2-§2.6)
for Drake. See ``DRAKE_PARITY_SPEC.md`` for the architecture overview.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .humanoid_urdf import (
    CANONICAL_URDF,
    SHARED_DIMENSIONS_YAML,
    build_humanoid_urdf,
    load_humanoid_dimensions,
    load_humanoid_into_plant,
)

if TYPE_CHECKING:
    from .compute_cost_drake import compute_cost_drake, drake_simout_to_shared
    from .fit_swing import fit_swing_drake, polynomial_parameter_bounds
    from .fit_swing_autodiff import (
        DEFAULT_COEFFICIENT_BOUNDS,
        FitOptions,
        FitResult,
        default_theta_bounds,
        fit_swing_drake_autodiff,
    )
    from .provider import DrakeFitSwingProvider
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
    "DrakeFitSwingProvider",
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

_LAZY_EXPORTS = {
    "compute_cost_drake": ".compute_cost_drake",
    "drake_simout_to_shared": ".compute_cost_drake",
    "fit_swing_drake": ".fit_swing",
    "polynomial_parameter_bounds": ".fit_swing",
    "DEFAULT_COEFFICIENT_BOUNDS": ".fit_swing_autodiff",
    "FitOptions": ".fit_swing_autodiff",
    "FitResult": ".fit_swing_autodiff",
    "default_theta_bounds": ".fit_swing_autodiff",
    "fit_swing_drake_autodiff": ".fit_swing_autodiff",
    "DrakeFitSwingProvider": ".provider",
    "COEFFS_PER_JOINT": ".simulate",
    "SimOptions": ".simulate",
    "SimOut": ".simulate",
    "evaluate_torque_polynomial": ".simulate",
    "simulate_with_coefficients": ".simulate",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(_LAZY_EXPORTS.keys()) + [
        "CANONICAL_URDF",
        "SHARED_DIMENSIONS_YAML",
        "build_humanoid_urdf",
        "load_humanoid_dimensions",
        "load_humanoid_into_plant",
    ]


# ---------------------------------------------------------------------------
# Auto-register the Drake provider at import time (issue #4516).
# ---------------------------------------------------------------------------
# The cross-engine matcher (issue #4513) discovers engines by importing
# ``src.engines.physics_engines.<engine>.python.motion_matching`` and
# expecting the module to have called ``register_provider`` as a side
# effect. We do that here. The registry call is idempotent so repeat
# imports are safe.
try:
    from src.shared.python.motion_matching.provider_registry import (
        register_provider as _register_provider,
    )
except ImportError:  # pragma: no cover - defensive
    # Registry module is optional during partial-rollout periods (the
    # canonical surface is part of #4514). If it is absent we still
    # expose ``DrakeFitSwingProvider`` for direct construction.
    pass
else:
    from .provider import DrakeFitSwingProvider

    _register_provider(DrakeFitSwingProvider())
