"""Unit tests for the ODE reference backend.

The ODE backend is the analytical *safety net*: it must reproduce the existing
closed-form double-pendulum dynamics exactly. These tests verify

* the advertised capability flags,
* the ``reset`` / ``get_state`` round-trip,
* the rollout contract (``horizon + 1`` finite samples, ``(., 2)`` shapes),
* ``mass_matrix`` agreeing with :class:`DoublePendulumDynamics` to machine
  precision,
* ``forward_dynamics`` with no control equalling ``solve(M, -bias)``, and
* energy conservation of a passive, undamped, gravity-free rollout (the
  property a symplectic-ish reference integrator must approximately satisfy).

All RNG is seeded; no optional dependencies are touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumDynamics,
)
from src.shared.python.simulation_backends import GolfModelParams, SimState
from src.shared.python.simulation_backends.ode_backend import ODEBackend

pytestmark = pytest.mark.unit

_RNG = np.random.default_rng(0)


def _params() -> GolfModelParams:
    """Return the canonical default golf model parameters."""
    return GolfModelParams.default()


def _conservative_params() -> GolfModelParams:
    """Return a gravity-free, undamped variant for energy-conservation tests."""
    return GolfModelParams.default().model_copy(
        update={
            "gravity_enabled": False,
            "damping_shoulder": 0.0,
            "damping_wrist": 0.0,
        }
    )


def test_capabilities_flags() -> None:
    """Capabilities advertise a CPU, non-batched, dynamics-providing backend."""
    caps = ODEBackend(_params()).capabilities
    assert caps.name == "ode"
    assert caps.device == "cpu"
    assert caps.supports_batched is False
    assert caps.is_differentiable is False
    assert caps.provides_dynamics is True


def test_constructor_rejects_non_positive_dt() -> None:
    """A non-positive default ``dt`` is a precondition violation."""
    with pytest.raises(ValueError):
        ODEBackend(_params(), dt=0.0)


def test_reset_get_state_round_trip() -> None:
    """``reset`` then ``get_state`` returns the seeded positions/velocities."""
    backend = ODEBackend(_params())
    q = np.array([0.3, -0.7])
    v = np.array([1.1, -2.2])
    backend.reset(SimState(q=q, v=v, time=0.5))
    state = backend.get_state()
    np.testing.assert_allclose(state.q, q)
    np.testing.assert_allclose(state.v, v)
    assert state.time == pytest.approx(0.5)


def test_reset_none_zeros_state() -> None:
    """``reset(None)`` zeros positions, velocities and the clock."""
    backend = ODEBackend(_params())
    backend.reset(SimState(q=[1.0, 2.0], v=[3.0, 4.0], time=9.0))
    backend.reset(None)
    state = backend.get_state()
    np.testing.assert_allclose(state.q, np.zeros(2))
    np.testing.assert_allclose(state.v, np.zeros(2))
    assert state.time == pytest.approx(0.0)


def test_rollout_passive_trace_shape_and_finiteness() -> None:
    """A passive rollout yields ``horizon + 1`` finite samples of width 2."""
    backend = ODEBackend(_params())
    backend.reset(SimState(q=[0.2, 0.1], v=[0.0, 0.0]))
    horizon = 50
    dt = 0.01
    trace = backend.rollout(None, horizon=horizon, dt=dt)

    assert trace.backend == "ode"
    assert trace.num_steps == horizon + 1
    assert trace.q.shape == (horizon + 1, 2)
    assert trace.v.shape == (horizon + 1, 2)
    assert trace.u is None
    np.testing.assert_allclose(trace.t, np.arange(horizon + 1) * dt, rtol=0, atol=1e-12)
    assert np.all(np.isfinite(trace.q))
    assert np.all(np.isfinite(trace.v))


def test_rollout_initial_sample_matches_current_state() -> None:
    """The first trace sample is the pre-rollout state at ``t == 0``."""
    backend = ODEBackend(_params())
    backend.reset(SimState(q=[0.4, -0.2], v=[0.5, -0.5]))
    trace = backend.rollout(None, horizon=10, dt=0.01)
    np.testing.assert_allclose(trace.q[0], [0.4, -0.2])
    np.testing.assert_allclose(trace.v[0], [0.5, -0.5])
    assert trace.t[0] == pytest.approx(0.0)


def test_rollout_with_controls_records_time_aligned_history() -> None:
    """The control history is stored time-aligned: ``(horizon + 1, 2)`` rows.

    ``u[k]`` is the control applied during step ``k`` and the final row is
    zero-padded (no step departs the terminal sample), matching the MuJoCo
    backend layout for row-for-row cross-validation.
    """
    backend = ODEBackend(_params())
    horizon = 8
    controls = _RNG.standard_normal((horizon, 2))
    trace = backend.rollout(controls, horizon=horizon, dt=0.01)
    assert trace.u is not None
    assert trace.u.shape == (horizon + 1, 2)
    np.testing.assert_allclose(trace.u[:horizon], controls)
    np.testing.assert_allclose(trace.u[horizon], np.zeros(2))


@pytest.mark.parametrize(
    ("controls", "horizon"),
    [
        (np.zeros((3, 2)), 5),  # wrong number of rows
        (np.zeros((5, 3)), 5),  # wrong control width
    ],
)
def test_rollout_rejects_mismatched_controls(
    controls: np.ndarray, horizon: int
) -> None:
    """Controls whose shape disagrees with ``(horizon, 2)`` are rejected."""
    backend = ODEBackend(_params())
    with pytest.raises(ValueError):
        backend.rollout(controls, horizon=horizon, dt=0.01)


@pytest.mark.parametrize(("horizon", "dt"), [(0, 0.01), (5, 0.0), (5, -0.1)])
def test_rollout_rejects_bad_horizon_or_dt(horizon: int, dt: float) -> None:
    """Non-positive horizon or step size violates the rollout precondition."""
    backend = ODEBackend(_params())
    with pytest.raises(ValueError):
        backend.rollout(None, horizon=horizon, dt=dt)


def test_mass_matrix_matches_analytical_dynamics() -> None:
    """``mass_matrix`` reproduces the analytical 2x2 matrix to 1e-12."""
    params = _params()
    backend = ODEBackend(params)
    dyn = DoublePendulumDynamics(params.to_double_pendulum_parameters())
    for theta2 in (-1.3, -0.2, 0.0, 0.6, 1.4):
        q = np.array([0.0, theta2])
        expected = np.array(dyn.mass_matrix(theta2), dtype=float)
        np.testing.assert_allclose(backend.mass_matrix(q), expected, atol=1e-12)


def test_mass_matrix_is_symmetric() -> None:
    """The inertia matrix is symmetric (postcondition of a valid mass matrix)."""
    backend = ODEBackend(_params())
    m = backend.mass_matrix(np.array([0.0, 0.7]))
    np.testing.assert_allclose(m, m.T, atol=1e-12)


def test_bias_forces_match_analytical_sum() -> None:
    """``bias_forces`` equals coriolis + gravity + damping from the dynamics."""
    params = _params()
    backend = ODEBackend(params)
    dyn = DoublePendulumDynamics(params.to_double_pendulum_parameters())
    q = np.array([0.3, -0.4])
    v = np.array([1.2, -0.8])
    c1, c2 = dyn.coriolis_vector(q[1], v[0], v[1])
    g1, g2 = dyn.gravity_vector(q[0], q[1])
    d1, d2 = dyn.damping_vector(v[0], v[1])
    expected = np.array([c1 + g1 + d1, c2 + g2 + d2])
    np.testing.assert_allclose(backend.bias_forces(q, v), expected, atol=1e-12)


def test_forward_dynamics_passive_equals_solve_minus_bias() -> None:
    """With ``u=None`` the acceleration is ``solve(M, -bias)``."""
    backend = ODEBackend(_params())
    q = np.array([0.25, -0.6])
    v = np.array([0.9, -1.1])
    mass = backend.mass_matrix(q)
    bias = backend.bias_forces(q, v)
    expected = np.linalg.solve(mass, -bias)
    np.testing.assert_allclose(
        backend.forward_dynamics(q, v, None), expected, atol=1e-12
    )


def test_forward_dynamics_control_enters_linearly() -> None:
    """Applying torque ``u`` shifts the acceleration by ``solve(M, u)``."""
    backend = ODEBackend(_params())
    q = np.array([0.1, 0.4])
    v = np.array([-0.3, 0.7])
    u = np.array([2.5, -1.5])
    mass = backend.mass_matrix(q)
    passive = backend.forward_dynamics(q, v, None)
    expected = passive + np.linalg.solve(mass, u)
    np.testing.assert_allclose(backend.forward_dynamics(q, v, u), expected, atol=1e-12)


def test_set_control_requires_two_entries() -> None:
    """A control vector shorter than two entries is rejected."""
    backend = ODEBackend(_params())
    with pytest.raises(ValueError):
        backend.set_control(np.array([1.0]))


def test_set_control_is_copied() -> None:
    """``set_control`` stores a private copy (mutating the input is inert)."""
    backend = ODEBackend(_params())
    u = np.array([1.0, 2.0])
    backend.set_control(u)
    u[0] = 99.0
    # Drive one step then confirm the stored torque was the original, not 99.
    q = np.array([0.0, 0.0])
    v = np.array([0.0, 0.0])
    mass = backend.mass_matrix(q)
    expected = np.linalg.solve(mass, np.array([1.0, 2.0]) - backend.bias_forces(q, v))
    np.testing.assert_allclose(
        backend.forward_dynamics(q, v, np.array([1.0, 2.0])), expected, atol=1e-12
    )


def test_passive_rollout_conserves_energy() -> None:
    """A gravity-free, undamped passive rollout conserves kinetic energy.

    Kinetic energy ``0.5 v^T M(q) v`` must stay ~constant over a 200-step
    passive rollout started from a nonzero velocity (max relative drift < 1e-2).
    """
    params = _conservative_params()
    backend = ODEBackend(params)
    backend.reset(SimState(q=[0.5, -0.3], v=[1.5, -1.0]))
    horizon = 200
    trace = backend.rollout(None, horizon=horizon, dt=0.005)

    energies = np.array(
        [
            0.5 * v @ backend.mass_matrix(q) @ v
            for q, v in zip(trace.q, trace.v, strict=True)
        ]
    )
    e0 = energies[0]
    assert e0 > 0.0
    max_rel_drift = float(np.max(np.abs(energies - e0)) / e0)
    assert max_rel_drift < 1e-2
