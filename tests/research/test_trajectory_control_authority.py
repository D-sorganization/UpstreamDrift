"""Unit tests for trajectory-varying event-conditioned control authority (#9123)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    compute_trajectory_authority,
    continuous_dynamics,
    discrete_control_jacobian,
    discrete_rk4_step,
    discrete_state_jacobian,
    generate_nominal_downswing_trajectory,
    orthonormal_tangent_basis,
    transverse_event_projector,
)

pytestmark = pytest.mark.scientific


def test_continuous_and_discrete_step_dimensions() -> None:
    state = np.array([1.0, 0.5, 2.0, -1.0])
    control = np.array([50.0, -5.0])
    dt = 0.002

    f = continuous_dynamics(state, control)
    assert f.shape == (4,)
    assert np.all(np.isfinite(f))

    next_state = discrete_rk4_step(state, control, dt)
    assert next_state.shape == (4,)
    assert np.all(np.isfinite(next_state))


def test_jacobians_structure_and_finiteness() -> None:
    state = np.array([1.5, 0.8, 1.0, -0.5])
    control = np.array([40.0, 10.0])
    dt = 0.002

    A = discrete_state_jacobian(state, control, dt)
    B = discrete_control_jacobian(state, control, dt)

    assert A.shape == (4, 4)
    assert B.shape == (4, 2)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))
    # A must be close to I + dt * df/dx
    assert np.allclose(np.diag(A), 1.0, atol=0.1)


def test_transverse_event_projector_properties() -> None:
    state = np.array([0.0, 0.2, 5.0, 2.0])
    control = np.array([10.0, 5.0])
    n = np.array([1.0, 0.0, 0.0, 0.0])

    P, is_transverse, inner = transverse_event_projector(
        state, control, guard_gradient=n
    )
    assert is_transverse is True
    assert inner == pytest.approx(5.0)

    # Idempotence: P^2 = P
    assert np.allclose(P @ P, P, atol=1e-12)

    # Tangent null direction: P * f = 0
    f = continuous_dynamics(state, control)
    assert np.allclose(P @ f, 0.0, atol=1e-12)


def test_near_grazing_guard_fails_closed() -> None:
    state = np.array([0.0, 0.2, 1e-6, 0.0])  # near zero velocity along guard normal
    control = np.array([0.0, 0.0])
    n = np.array([1.0, 0.0, 0.0, 0.0])

    P, is_transverse, inner = transverse_event_projector(
        state, control, guard_gradient=n, transverse_tolerance=1e-3
    )
    assert is_transverse is False
    assert np.allclose(P, np.eye(4))


def test_orthonormal_tangent_basis() -> None:
    n = np.array([1.0, 0.0, 0.0, 0.0])
    Q = orthonormal_tangent_basis(n)
    assert Q.shape == (4, 3)
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-12)
    assert np.allclose(n @ Q, 0.0, atol=1e-12)


def test_trajectory_authority_additivity_and_killswitches() -> None:
    states, controls = generate_nominal_downswing_trajectory(dt=0.002, steps=60)
    result = compute_trajectory_authority(states, controls, dt=0.002)

    # Zero input gives zero Gramian and rank 0
    assert result.full_rank_zero == 0
    assert result.tangent_rank_zero == 0
    assert np.allclose(result.full_gramian_zero, 0.0)

    # Single-channel additivity: W_both = W_shoulder + W_wrist
    assert result.additivity_residual_norm < 1e-10

    # Full and tangent rank
    assert result.full_rank_both == 4
    assert result.tangent_rank_both == 3

    # Pulse response agreement
    assert result.pulse_agreement_relative_error < 0.05
