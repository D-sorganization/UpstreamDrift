from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.rotating_base_two_hand import (
    RotatingBaseConfig,
    RotatingBaseParams,
    RotatingBaseState,
    TorsoTwoHandControl,
    constraint_jacobian,
    constraint_vector,
    initial_state,
    rollout,
    solve_constrained_dynamics,
)

pytestmark = pytest.mark.unit


def test_contracts_reject_nonphysical_inputs() -> None:
    with pytest.raises(ValueError, match="torso_inertia"):
        replace(RotatingBaseParams.publication_default(), torso_inertia_kg_m2=0.0)
    with pytest.raises(ValueError, match="step_s"):
        RotatingBaseConfig(duration_s=0.1, step_s=0.0)
    with pytest.raises(ValueError, match="finite"):
        TorsoTwoHandControl(torso_nm=np.nan)


def test_initial_state_closes_separated_grip_constraints() -> None:
    params = RotatingBaseParams.publication_default()
    state = initial_state(params, torso_rate_rad_s=4.0, club_rate_rad_s=2.0)

    assert np.linalg.norm(constraint_vector(state.q, params)) < 1e-10
    assert np.linalg.norm(constraint_jacobian(state.q, params) @ state.qdot) < 1e-10
    assert np.linalg.matrix_rank(constraint_jacobian(state.q, params)) == 4


def test_constrained_solve_closes_acceleration_and_action_reaction() -> None:
    params = RotatingBaseParams.publication_default()
    state = initial_state(params, torso_rate_rad_s=3.0, club_rate_rad_s=1.5)
    solution = solve_constrained_dynamics(
        state, TorsoTwoHandControl(torso_nm=35.0), params
    )

    assert solution.constraint_rank == 4
    assert solution.kkt_residual_norm < params.kkt_tolerance
    assert solution.acceleration_constraint_residual_norm < 1e-8
    assert np.allclose(
        solution.force_on_hands_n + solution.force_on_club_n,
        0.0,
        atol=1e-12,
    )


def test_short_rollout_preserves_constraints_and_energy_ledger() -> None:
    params = RotatingBaseParams.publication_default()
    initial = initial_state(params, torso_rate_rad_s=2.5, club_rate_rad_s=0.5)

    def law(_time_s: float, _state: RotatingBaseState) -> TorsoTwoHandControl:
        return TorsoTwoHandControl(torso_nm=25.0, lead_arm_nm=8.0, trail_arm_nm=8.0)

    trace = rollout(
        initial,
        law,
        params,
        RotatingBaseConfig(duration_s=0.04, step_s=0.0005),
    )

    assert np.max(trace.position_constraint_norm_m) < 1e-8
    assert np.max(trace.velocity_constraint_norm_m_s) < 1e-7
    assert np.max(np.abs(trace.contact_power_identity_residual_w)) < 1e-8
    assert abs(trace.work_energy_closure_j) < 2e-2
    assert np.all(np.isfinite(trace.clubhead_speed_m_s))


def test_coincident_grips_remove_force_generated_couple() -> None:
    base = RotatingBaseParams.publication_default()
    params = replace(base, lead_grip_offset_m=0.0, trail_grip_offset_m=0.0)
    state = initial_state(params, torso_rate_rad_s=3.0, club_rate_rad_s=1.0)
    solution = solve_constrained_dynamics(state, TorsoTwoHandControl(), params)

    assert abs(solution.force_generated_couple_nm) < 1e-12
