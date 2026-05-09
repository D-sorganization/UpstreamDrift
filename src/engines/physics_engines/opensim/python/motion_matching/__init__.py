"""OpenSim motion-matching package.

Houses the canonical engine-agnostic forward-sim wrapper
``simulate_with_coefficients`` (issue #4120) plus its supporting modules.

Public surface (lazily imported to keep ``import opensim`` optional):

* ``simulate_with_coefficients(theta, options, initial_pose) -> SimOut``
* ``SimOptions`` / ``SimOut`` frozen dataclasses
* ``PolynomialTorqueController`` (only when ``opensim`` is installed)
* ``evaluate_polynomial_torque`` (pure-numpy, always importable)
* ``forward_kinematics``: extract canonical landmarks (grip, clubhead, ...)
  from a SimTK state given the OpenSim model.
* ``fit_swing_opensim``: scipy.optimize.minimize(SLSQP) driver that fits
  polynomial torque coefficients to a measured ClubTarget (issue #4128).
* ``fit_swing_opensim_multistart``: deterministic multistart orchestrator
  for the single-start OpenSim fit driver (issue #4297).
* ``synthesize_target_from_coefficients``: TDD oracle (issue #4124).
* ``coord_map``: pure-Python OpenSim<->Simscape coordinate mapping helpers
  (no SWIG ``opensim`` import required).
"""

from __future__ import annotations

from src.engines.physics_engines.opensim.python.motion_matching.coord_map import (
    OPENSIM_COORD_ORDER,
    OPENSIM_NEUTRAL_POSE,
    OPENSIM_SIGN_CONVENTION,
    OPENSIM_TO_SIMSCAPE,
    SIMSCAPE_COORD_ORDER,
    frame_y_up_to_z_up,
    frame_z_up_to_y_up,
    from_simscape,
    quat_canonical_to_eigen,
    quat_eigen_to_canonical,
    to_simscape,
)
from src.engines.physics_engines.opensim.python.motion_matching.fit_multistart import (
    AllStartsFailedError,
    MultistartOptions,
    fit_swing_opensim_multistart,
    generate_multistart_seeds,
)
from src.engines.physics_engines.opensim.python.motion_matching.fit_swing import (
    FitOptions,
    FitResult,
    fit_swing_opensim,
)
from src.engines.physics_engines.opensim.python.motion_matching.forward_kinematics import (
    extract_clubhead_pose,
    extract_full_pose,
    extract_grip_pose,
)
from src.engines.physics_engines.opensim.python.motion_matching.simulate import (
    COEFFS_PER_JOINT,
    POLY_DEGREE,
    SimOptions,
    SimOut,
    evaluate_polynomial_torque,
    simulate_with_coefficients,
)
from src.engines.physics_engines.opensim.python.motion_matching.provider import (
    OpenSimFitSwingProvider,
)
from src.engines.physics_engines.opensim.python.motion_matching.synthesize import (
    SynthOptions,
    synthesize_target_from_coefficients,
)

__all__ = [
    "COEFFS_PER_JOINT",
    "AllStartsFailedError",
    "FitOptions",
    "FitResult",
    "MultistartOptions",
    "OPENSIM_COORD_ORDER",
    "OPENSIM_NEUTRAL_POSE",
    "OPENSIM_SIGN_CONVENTION",
    "OPENSIM_TO_SIMSCAPE",
    "POLY_DEGREE",
    "SIMSCAPE_COORD_ORDER",
    "OpenSimFitSwingProvider",
    "SimOptions",
    "SimOut",
    "SynthOptions",
    "evaluate_polynomial_torque",
    "extract_clubhead_pose",
    "extract_full_pose",
    "extract_grip_pose",
    "fit_swing_opensim",
    "fit_swing_opensim_multistart",
    "frame_y_up_to_z_up",
    "frame_z_up_to_y_up",
    "from_simscape",
    "generate_multistart_seeds",
    "quat_canonical_to_eigen",
    "quat_eigen_to_canonical",
    "simulate_with_coefficients",
    "synthesize_target_from_coefficients",
    "to_simscape",
    "viz",
]


# ---------------------------------------------------------------------------
# Auto-register the OpenSim provider at import time (issue #4708).
# ---------------------------------------------------------------------------
# The cross-engine matcher (issue #4513) discovers engines by importing
# ``src.engines.physics_engines.<engine>.python.motion_matching`` and
# expecting the module to have called ``register_provider`` as a side
# effect. We register against both the canonical Protocol-based registry
# (``provider``) and the engine-agnostic surface (``provider_registry``)
# so that downstream code using either lookup path sees the OpenSim
# entry. Each call is idempotent on repeat imports.
#
# The try/except blocks ensure that callers without an OpenSim wheel
# (or with partial-rollout shared modules) can still ``import`` this
# package without crashing -- the registration silently no-ops if the
# canonical registry surface is unavailable.
try:
    from src.shared.python.motion_matching.provider import (
        register_provider as _register_canonical,
    )
except ImportError:  # pragma: no cover - defensive
    pass
else:
    # ``register_provider`` is idempotent for repeat same-class registrations,
    # so any exception here (e.g. an ``engine_name`` collision with a different
    # provider class) reflects a real registration bug and must surface rather
    # than be silently swallowed (issue #4743).
    _register_canonical(OpenSimFitSwingProvider())

try:
    from src.shared.python.motion_matching.provider_registry import (
        register_provider as _register_legacy,
    )
except ImportError:  # pragma: no cover - defensive
    pass
else:
    _register_legacy(OpenSimFitSwingProvider())
