"""OpenSim motion-matching package.

Houses the canonical engine-agnostic forward-sim wrapper
``simulate_with_coefficients`` (issue #4120) plus its supporting modules.

Public surface (lazily imported to keep ``import opensim`` optional):

* ``simulate_with_coefficients(theta, options, initial_pose) -> SimOut``
* ``SimOptions`` / ``SimOut`` frozen dataclasses
* ``PolynomialTorqueController`` (only when ``opensim`` is installed)
* ``evaluate_polynomial_torque`` (pure-numpy, always importable)
"""

from __future__ import annotations

from src.engines.physics_engines.opensim.python.motion_matching.simulate import (
    COEFFS_PER_JOINT,
    POLY_DEGREE,
    SimOptions,
    SimOut,
    evaluate_polynomial_torque,
    simulate_with_coefficients,
)

__all__ = [
    "POLY_DEGREE",
    "COEFFS_PER_JOINT",
    "SimOptions",
    "SimOut",
    "evaluate_polynomial_torque",
    "simulate_with_coefficients",
    "viz",
]
