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
* ``synthesize_target_from_coefficients``: TDD oracle (issue #4124).
"""

from __future__ import annotations

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
from src.engines.physics_engines.opensim.python.motion_matching.synthesize import (
    SynthOptions,
    synthesize_target_from_coefficients,
)

__all__ = [
    "COEFFS_PER_JOINT",
    "FitOptions",
    "FitResult",
    "POLY_DEGREE",
    "SimOptions",
    "SimOut",
    "SynthOptions",
    "evaluate_polynomial_torque",
    "extract_clubhead_pose",
    "extract_full_pose",
    "extract_grip_pose",
    "fit_swing_opensim",
    "simulate_with_coefficients",
    "synthesize_target_from_coefficients",
    "viz",
]
