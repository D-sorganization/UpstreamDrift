"""Pinocchio motion-matching forward simulation, LM fit, target adapters, and
leaderboard.

Public API:
    simulate_with_coefficients -- RK4 + ABA forward simulator (issue #4118).
    SimOptions                 -- frozen dataclass of integrator options.
    SimOut                     -- frozen dataclass of trajectory + diagnostics.
    evaluate_polynomial_torque -- pure-numpy polynomial torque evaluation.
    fit_swing_pinocchio        -- LM + analytical-Jacobian swing fit (issue #4132).
    FitOptions, FitResult      -- canonical fit configuration / result.
    polynomial_basis,
    polynomial_torque_chain_rule,
    rotmat_to_quat_wxyz        -- pure-numpy gradient helpers (testable
                                  without pinocchio).
    POLY_DEGREE                -- canonical polynomial degree (6).
    COEFFS_PER_JOINT           -- canonical coeffs-per-joint count (7).
    load_robneal_target        -- ClubTarget adapter for Rob Neal *.mat (issue #4127).
    write_leaderboard_entry    -- leaderboard JSON writer (issue #4133).
    ClubTargetLike             -- protocol-style record for leaderboard inputs.
    synthesize_target_from_coefficients -- SimOut -> ClubTarget oracle (issue #4121).
    SynthesizeOptions          -- options for the synthesize oracle.
    run_recovery_sweep         -- K random-theta recovery harness.
    RecoveryHarnessOptions     -- knobs for the recovery sweep.
    RecoverySummary            -- aggregate recovery statistics.
    RecoveryTrial              -- per-sample recovery diagnostics.
    sample_random_theta        -- helper to draw a random theta vector.

Engine-local Rob Neal ``.mat`` adapter (:func:`load_robneal_target`)
intentionally lives here. Issue #4095 (PARITY-LOADERS) is completed — the
shared ``shared/python/motion_matching/loaders/`` package but only promoted
the C3D / Excel / synthetic loaders; the Rob Neal ``club_swing_dataset``
adapter was not part of that scope. A future refactor may promote it
upstream and have this module re-export it for backward compatibility.
"""

from __future__ import annotations

from ._types import ClubTargetLike
from .club_target_adapter import load_robneal_target
from .fit_swing import (
    FitOptions,
    FitResult,
    fit_swing_pinocchio,
    polynomial_basis,
    polynomial_torque_chain_rule,
    rotmat_to_quat_wxyz,
)
from .leaderboard_writer import write_leaderboard_entry
from .provider import PinocchioFitSwingProvider
from .recovery_harness import (
    RecoveryHarnessOptions,
    RecoverySummary,
    RecoveryTrial,
    run_recovery_sweep,
    sample_random_theta,
)
from .simulate import (
    COEFFS_PER_JOINT,
    POLY_DEGREE,
    SimOptions,
    SimOut,
    evaluate_polynomial_torque,
    simulate_with_coefficients,
)
from .synthesize import (
    SynthesizeOptions,
    synthesize_target_from_coefficients,
)

__all__ = [
    "COEFFS_PER_JOINT",
    "ClubTargetLike",
    "FitOptions",
    "FitResult",
    "POLY_DEGREE",
    "PinocchioFitSwingProvider",
    "RecoveryHarnessOptions",
    "RecoverySummary",
    "RecoveryTrial",
    "SimOptions",
    "SimOut",
    "SynthesizeOptions",
    "evaluate_polynomial_torque",
    "fit_swing_pinocchio",
    "load_robneal_target",
    "polynomial_basis",
    "polynomial_torque_chain_rule",
    "rotmat_to_quat_wxyz",
    "run_recovery_sweep",
    "sample_random_theta",
    "simulate_with_coefficients",
    "synthesize_target_from_coefficients",
    "write_leaderboard_entry",
]
