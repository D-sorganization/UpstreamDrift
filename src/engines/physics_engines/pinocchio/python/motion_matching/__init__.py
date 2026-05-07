"""Pinocchio motion-matching forward simulation + LM fit utilities.

Implements the canonical engine-agnostic ``simulate_with_coefficients``
forward-sim wrapper (issue #4118) and the ``fit_swing_pinocchio`` LM
optimiser with analytical Jacobians (issue #4132 — the killer feature).

Public API:
    simulate_with_coefficients -- RK4 + ABA forward simulator.
    SimOptions                 -- frozen dataclass of integrator options.
    SimOut                     -- frozen dataclass of trajectory + diagnostics.
    evaluate_polynomial_torque -- pure-numpy polynomial torque evaluation.
    fit_swing_pinocchio        -- LM + analytical-Jacobian swing fit.
    FitOptions, FitResult      -- canonical fit configuration / result.
    polynomial_basis,
    polynomial_torque_chain_rule,
    rotmat_to_quat_wxyz        -- pure-numpy gradient helpers (testable
                                  without pinocchio).
    POLY_DEGREE                -- canonical polynomial degree (6).
    COEFFS_PER_JOINT           -- canonical coeffs-per-joint count (7).
"""

from __future__ import annotations

from .fit_swing import (
    FitOptions,
    FitResult,
    fit_swing_pinocchio,
    polynomial_basis,
    polynomial_torque_chain_rule,
    rotmat_to_quat_wxyz,
)
from .simulate import (
    COEFFS_PER_JOINT,
    POLY_DEGREE,
    SimOptions,
    SimOut,
    evaluate_polynomial_torque,
    simulate_with_coefficients,
)

__all__ = [
    "COEFFS_PER_JOINT",
    "FitOptions",
    "FitResult",
    "POLY_DEGREE",
    "SimOptions",
    "SimOut",
    "evaluate_polynomial_torque",
    "fit_swing_pinocchio",
    "polynomial_basis",
    "polynomial_torque_chain_rule",
    "rotmat_to_quat_wxyz",
    "simulate_with_coefficients",
]
