"""Unit tests for bounded nonlinear event-reaching feasibility (#9124)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.bounded_event_reachability import (
    FeasibilityOutcome,
    TorqueBounds,
    run_bounded_reachability_suite,
    solve_bounded_reaching,
)
from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    generate_nominal_downswing_trajectory,
)

pytestmark = pytest.mark.scientific


def test_small_amplitude_bounded_reaching_matches_linear_prediction() -> None:
    states, controls = generate_nominal_downswing_trajectory(dt=0.002, steps=60)
    dt = 0.002
    bounds = TorqueBounds(max_shoulder_torque_nm=200.0, max_wrist_torque_nm=30.0)

    target_tangent = np.array([0.001, -0.002, 0.001])
    res = solve_bounded_reaching(
        states, controls, target_tangent, dt, bounds, np.array([1.0, 1.0])
    )

    assert res.outcome == FeasibilityOutcome.FEASIBLE
    assert res.terminal_tangent_residual_norm < 1e-3
    assert res.replay_exact_match is True


def test_zero_torque_bounds_and_zero_authority_negative_control() -> None:
    states, controls = generate_nominal_downswing_trajectory(dt=0.002, steps=60)
    dt = 0.002
    bounds = TorqueBounds(max_shoulder_torque_nm=200.0, max_wrist_torque_nm=30.0)

    target_tangent = np.array([0.1, 0.2, -0.1])
    res_zero = solve_bounded_reaching(
        states, controls, target_tangent, dt, bounds, np.array([0.0, 0.0])
    )

    assert res_zero.outcome == FeasibilityOutcome.BOUND_SATURATED
    assert res_zero.is_bound_saturated is True
    assert res_zero.replay_exact_match is True
    # Zero authority cannot reduce residual to zero
    assert res_zero.terminal_tangent_residual_norm > 0.05


def test_finite_amplitude_saturation_is_typed_and_replayed() -> None:
    states, controls = generate_nominal_downswing_trajectory(dt=0.002, steps=60)
    dt = 0.002
    bounds = TorqueBounds(max_shoulder_torque_nm=50.0, max_wrist_torque_nm=5.0)

    # Large target that cannot be reached under tight torque bounds
    target_tangent = np.array([2.0, 4.0, -2.0])
    res_large = solve_bounded_reaching(
        states, controls, target_tangent, dt, bounds, np.array([1.0, 1.0])
    )

    assert res_large.outcome in (
        FeasibilityOutcome.BOUND_SATURATED,
        FeasibilityOutcome.INFEASIBLE,
    )
    assert res_large.replay_exact_match is True
