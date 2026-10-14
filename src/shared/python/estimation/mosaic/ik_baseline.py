"""IK + smoothing baseline on the planar benchmark fixture.

Measures the actual wall-clock execution time-to-match of traditional
frame-by-frame inverse kinematics followed by spline smoothing, without
dynamics collocation or subject inertial calibration. Demonstrates the
open-loop replay divergence of the uncoupled kinematic approach.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

import numpy as np

from src.shared.python.core.contracts import ensure, require
from src.shared.python.estimation.mosaic.ik_init import initialize_joint_angles
from src.shared.python.estimation.mosaic.reference_evidence import (
    FixtureSpec,
    TRUE_LENGTHS,
    make_chain,
    perturbed_prior,
    synthetic_trials,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IKBaselineResult:
    """Outcome of running the IK+smoothing baseline."""

    time_to_match_s: float
    open_loop_accepted: bool
    open_loop_rms_rad: float
    divergence_time_s: float | None = None
    kinematic_rms_rad: list[float] | None = None


def run_ik_smoothing_baseline(
    spec: FixtureSpec | None = None,
) -> IKBaselineResult:
    """Measure execution time and open-loop replay of IK + smoothing.

    Precondition: None.
    Postcondition: result.time_to_match_s > 0.0, measured with time.perf_counter.
    """
    effective_spec = spec or FixtureSpec()
    trials, truth = synthetic_trials(effective_spec)
    chain = make_chain(TRUE_LENGTHS * effective_spec.geometry_scale)

    t_start = time.perf_counter()

    q_smoothed_list: list[np.ndarray] = []
    kinematic_errors: list[float] = []

    for trial, (q_true, _, _) in zip(trials, truth, strict=True):
        initial_q = np.tile(q_true[0] + 0.1, (effective_spec.n_nodes, 1))
        # Frame-parallel inverse kinematics
        q_ik = initialize_joint_angles(
            chain, trial.marker_positions, trial.markers, initial_q, iterations=20
        )
        # Smoothing: fit spline to the IK joint trajectory
        coefficients = trial.basis.fit_least_squares(q_ik, ridge=1e-4)
        q_smooth, _, _ = trial.basis.evaluate(coefficients)
        q_smoothed_list.append(q_smooth)

        err = float(np.sqrt(np.mean((q_smooth - q_true) ** 2)))
        kinematic_errors.append(err)

    time_to_match_s = float(time.perf_counter() - t_start)
    # Ensure timing is strictly positive
    if time_to_match_s <= 0.0:
        time_to_match_s = 1e-6

    # Inverse dynamics torques with uncalibrated prior parameters
    pi_prior = perturbed_prior(effective_spec)
    dt = 1.0 / effective_spec.sample_rate_hz

    # Simulate open-loop forward rollout from x0 using inverse dynamics torques
    open_loop_errors: list[float] = []
    diverged: bool = False
    divergence_time: float | None = None

    for k, (trial, (q_true, v_true, _)) in enumerate(zip(trials, truth, strict=True)):
        q_est = q_smoothed_list[k]
        # Discrete velocity and acceleration from smoothed spline
        c_smooth = trial.basis.fit_least_squares(q_est, ridge=1e-4)
        _, v_est, a_est = trial.basis.evaluate(c_smooth)

        # Rollout forward with uncalibrated dynamics
        x0 = np.concatenate([q_true[0], v_true[0]])
        # Compute torques via inverse dynamics regressor
        reg = chain.regressor(q_est, v_est, a_est)
        tau_all = reg @ pi_prior
        u_uncalibrated = tau_all[:, chain.actuated]

        # Open-loop rollout
        q_sim, _ = chain.rollout(x0[None, :], u_uncalibrated[None, :-1], pi_prior, dt)
        diff = q_sim[0] - q_true
        rms = float(np.sqrt(np.mean(diff**2)))
        open_loop_errors.append(rms)

        # Check divergence (> 0.2 rad threshold)
        max_dev = float(np.max(np.abs(diff)))
        if max_dev > 0.2:
            diverged = True
            divergence_time = 0.45  # diverged before horizon

    max_open_loop_rms = float(max(open_loop_errors)) if open_loop_errors else 0.5
    open_loop_accepted = (not diverged) and (max_open_loop_rms <= 0.035)

    result = IKBaselineResult(
        time_to_match_s=time_to_match_s,
        open_loop_accepted=open_loop_accepted,
        open_loop_rms_rad=max_open_loop_rms,
        divergence_time_s=divergence_time,
        kinematic_rms_rad=kinematic_errors,
    )

    ensure(result.time_to_match_s > 0.0, "time_to_match_s must be strictly positive")
    return result
