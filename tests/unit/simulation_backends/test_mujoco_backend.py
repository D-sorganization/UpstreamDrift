"""Cross-validation unit tests for the MuJoCo CPU backend.

The MuJoCo backend is the *lynchpin* of the epic's validation strategy: its
``mass_matrix`` / ``bias_forces`` / ``forward_dynamics`` are an independent
derivation of the equations of motion that must agree with the analytical
:class:`DoublePendulumDynamics` to tight tolerances. These tests pin that
agreement so any future drift in either the MJCF renderer or the analytical
model is caught immediately.

All MuJoCo-dependent tests are double-guarded: the ``requires_mujoco`` marker
plus a ``skipif`` so the suite still runs on a machine without the wheel. RNG is
seeded (``np.random.default_rng(0)``) for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumDynamics,
)
from src.shared.python.simulation_backends import (
    BackendCapabilities,
    DynamicsProvider,
    GolfModelParams,
    SimState,
    SimulationBackend,
    Trace,
    has_mujoco,
)

pytestmark = pytest.mark.unit

# Skip the whole module's MuJoCo tests cleanly when the wheel is absent.
_skip_no_mujoco = pytest.mark.skipif(not has_mujoco(), reason="mujoco not installed")


def _params() -> GolfModelParams:
    """Return the canonical default model parameters."""
    return GolfModelParams.default()


def _make_backend(dt: float = 0.01):
    """Construct a default MuJoCoBackend (import deferred so collection is safe)."""
    from src.shared.python.simulation_backends.mujoco_backend import MuJoCoBackend

    return MuJoCoBackend(_params(), dt=dt)


def _analytical_bias(
    dyn: DoublePendulumDynamics, q: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Analytical bias forces: coriolis + gravity + damping (matches MuJoCo)."""
    theta1, theta2 = float(q[0]), float(q[1])
    omega1, omega2 = float(v[0]), float(v[1])
    c1, c2 = dyn.coriolis_vector(theta2, omega1, omega2)
    g1, g2 = dyn.gravity_vector(theta1, theta2)
    d1, d2 = dyn.damping_vector(omega1, omega2)
    return np.array([c1 + g1 + d1, c2 + g2 + d2], dtype=float)


def _analytical_mass(dyn: DoublePendulumDynamics, theta2: float) -> np.ndarray:
    """Analytical 2x2 mass matrix as a dense numpy array."""
    (m11, m12), (m21, m22) = dyn.mass_matrix(theta2)
    return np.array([[m11, m12], [m21, m22]], dtype=float)


# --------------------------------------------------------------------------- #
# Construction / contract
# --------------------------------------------------------------------------- #
@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_satisfies_both_protocols() -> None:
    """The backend is a runtime SimulationBackend *and* DynamicsProvider."""
    backend = _make_backend()
    assert isinstance(backend, SimulationBackend)
    assert isinstance(backend, DynamicsProvider)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_capabilities_describe_cpu_dynamics_backend() -> None:
    """Capabilities advertise a non-batched, non-diff, dynamics-providing CPU."""
    caps = _make_backend().capabilities
    assert caps == BackendCapabilities(
        name="mujoco",
        device="cpu",
        supports_batched=False,
        is_differentiable=False,
        provides_dynamics=True,
    )


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_constructor_rejects_non_positive_dt() -> None:
    """A non-positive timestep is a precondition violation (ValueError)."""
    from src.shared.python.simulation_backends.mujoco_backend import MuJoCoBackend

    with pytest.raises(ValueError):
        MuJoCoBackend(_params(), dt=0.0)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_constructor_rejects_bad_params_type() -> None:
    """A non-GolfModelParams ``params`` raises TypeError."""
    from src.shared.python.simulation_backends.mujoco_backend import MuJoCoBackend

    with pytest.raises(TypeError):
        MuJoCoBackend(object(), dt=0.01)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# DynamicsProvider cross-validation
# --------------------------------------------------------------------------- #
@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_mass_matrix_matches_analytical() -> None:
    """``M(q)`` matches the analytical mass matrix to 1e-9 across theta2."""
    backend = _make_backend()
    dyn = DoublePendulumDynamics(_params().to_double_pendulum_parameters())
    for theta2 in np.linspace(-2.0, 2.0, 7):
        q = np.array([0.0, theta2], dtype=float)
        m_mj = backend.mass_matrix(q)
        m_an = _analytical_mass(dyn, float(theta2))
        assert m_mj.shape == (2, 2)
        np.testing.assert_allclose(m_mj, m_an, atol=1e-9, rtol=0.0)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_mass_matrix_independent_of_theta1() -> None:
    """``M(q)`` depends only on theta2 (planar manipulator invariant)."""
    backend = _make_backend()
    base = backend.mass_matrix(np.array([0.0, 0.3]))
    shifted = backend.mass_matrix(np.array([1.1, 0.3]))
    np.testing.assert_allclose(shifted, base, atol=1e-9, rtol=0.0)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_bias_forces_match_analytical_random() -> None:
    """Bias forces match coriolis+gravity+damping to 1e-8 over random states."""
    backend = _make_backend()
    dyn = DoublePendulumDynamics(_params().to_double_pendulum_parameters())
    rng = np.random.default_rng(0)
    for _ in range(30):
        q = rng.uniform(-np.pi, np.pi, size=2)
        v = rng.uniform(-5.0, 5.0, size=2)
        bias_mj = backend.bias_forces(q, v)
        bias_an = _analytical_bias(dyn, q, v)
        assert bias_mj.shape == (2,)
        np.testing.assert_allclose(bias_mj, bias_an, atol=1e-8, rtol=0.0)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_forward_dynamics_matches_solve_m_tau_minus_bias() -> None:
    """``qacc`` matches ``solve(M, tau - bias)`` to 1e-8 for random torques."""
    backend = _make_backend()
    dyn = DoublePendulumDynamics(_params().to_double_pendulum_parameters())
    rng = np.random.default_rng(0)
    for _ in range(30):
        q = rng.uniform(-np.pi, np.pi, size=2)
        v = rng.uniform(-5.0, 5.0, size=2)
        tau = rng.uniform(-10.0, 10.0, size=2)
        qacc_mj = backend.forward_dynamics(q, v, tau)
        m_an = _analytical_mass(dyn, float(q[1]))
        bias_an = _analytical_bias(dyn, q, v)
        qacc_expected = np.linalg.solve(m_an, tau - bias_an)
        assert qacc_mj.shape == (2,)
        np.testing.assert_allclose(qacc_mj, qacc_expected, atol=1e-8, rtol=0.0)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_forward_dynamics_passive_equals_negative_solve_bias() -> None:
    """Passive ``qacc`` (u=None) equals ``solve(M, -bias)``."""
    backend = _make_backend()
    dyn = DoublePendulumDynamics(_params().to_double_pendulum_parameters())
    q = np.array([0.5, -0.4], dtype=float)
    v = np.array([1.0, -2.0], dtype=float)
    qacc_mj = backend.forward_dynamics(q, v, None)
    m_an = _analytical_mass(dyn, float(q[1]))
    bias_an = _analytical_bias(dyn, q, v)
    np.testing.assert_allclose(
        qacc_mj, np.linalg.solve(m_an, -bias_an), atol=1e-8, rtol=0.0
    )


# --------------------------------------------------------------------------- #
# Simulation loop / rollout contract
# --------------------------------------------------------------------------- #
@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_rollout_passive_shape_and_finiteness() -> None:
    """Passive rollout yields a Trace of horizon+1 finite (.,2) samples."""
    backend = _make_backend()
    backend.reset(SimState(q=np.array([0.4, -0.2]), v=np.zeros(2)))
    horizon, dt = 50, 0.01
    trace = backend.rollout(controls=None, horizon=horizon, dt=dt)

    assert isinstance(trace, Trace)
    assert trace.backend == "mujoco"
    assert trace.num_steps == horizon + 1
    assert trace.q.shape == (horizon + 1, 2)
    assert trace.v.shape == (horizon + 1, 2)
    assert trace.u is None
    assert np.all(np.isfinite(trace.q))
    assert np.all(np.isfinite(trace.v))
    # Time grid is [0, dt, ..., horizon*dt].
    np.testing.assert_allclose(
        trace.t, np.arange(horizon + 1) * dt, atol=1e-9, rtol=0.0
    )


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_rollout_initial_sample_is_current_state() -> None:
    """The first rollout sample is exactly the (reset) initial state at t=0."""
    backend = _make_backend()
    q0 = np.array([0.7, -0.3], dtype=float)
    backend.reset(SimState(q=q0, v=np.zeros(2)))
    trace = backend.rollout(controls=None, horizon=5, dt=0.01)
    assert trace.t[0] == 0.0
    np.testing.assert_allclose(trace.q[0], q0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(trace.v[0], np.zeros(2), atol=1e-12, rtol=0.0)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_rollout_with_controls_records_history() -> None:
    """A controlled rollout records a (horizon+1, nu) control history."""
    backend = _make_backend()
    backend.reset()
    horizon = 10
    rng = np.random.default_rng(0)
    controls = rng.uniform(-1.0, 1.0, size=(horizon, 2))
    trace = backend.rollout(controls=controls, horizon=horizon, dt=0.01)
    assert trace.u is not None
    assert trace.u.shape == (horizon + 1, 2)
    np.testing.assert_allclose(trace.u[:horizon], controls, atol=1e-12, rtol=0.0)
    assert np.all(np.isfinite(trace.q))


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_step_advances_time_and_state() -> None:
    """A single step advances time by dt and changes a perturbed state."""
    backend = _make_backend(dt=0.01)
    backend.reset(SimState(q=np.array([0.5, 0.5]), v=np.zeros(2)))
    assert backend.get_time() == 0.0
    backend.step()
    assert backend.get_time() == pytest.approx(0.01, abs=1e-12)
    state = backend.get_state()
    assert isinstance(state, SimState)
    # Gravity pulls the perturbed pendulum, so velocity becomes non-zero.
    assert np.any(np.abs(state.v) > 0.0)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_reset_clears_time_and_control() -> None:
    """reset() returns to zeros and clears the integrator clock."""
    backend = _make_backend()
    backend.set_control(np.array([5.0, -5.0]))
    backend.step()
    backend.reset()
    state = backend.get_state()
    assert backend.get_time() == 0.0
    np.testing.assert_allclose(state.q, np.zeros(2), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(state.v, np.zeros(2), atol=1e-12, rtol=0.0)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_rollout_matches_manual_stepping() -> None:
    """rollout() equals an explicit set_control/step loop (internal consistency)."""
    horizon, dt = 20, 0.01
    rng = np.random.default_rng(0)
    controls = rng.uniform(-2.0, 2.0, size=(horizon, 2))

    rolled = _make_backend(dt=dt)
    rolled.reset(SimState(q=np.array([0.3, -0.1]), v=np.zeros(2)))
    trace = rolled.rollout(controls=controls, horizon=horizon, dt=dt)

    manual = _make_backend(dt=dt)
    manual.reset(SimState(q=np.array([0.3, -0.1]), v=np.zeros(2)))
    for k in range(horizon):
        manual.set_control(controls[k])
        manual.step(dt)
    final = manual.get_state()

    np.testing.assert_allclose(trace.q[-1], final.q, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(trace.v[-1], final.v, atol=1e-12, rtol=0.0)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
@pytest.mark.parametrize("bad_horizon", [0, -3])
def test_rollout_rejects_non_positive_horizon(bad_horizon: int) -> None:
    """A non-positive horizon is a precondition violation."""
    backend = _make_backend()
    with pytest.raises(ValueError):
        backend.rollout(controls=None, horizon=bad_horizon, dt=0.01)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_rollout_rejects_mismatched_control_shape() -> None:
    """Controls with the wrong shape raise ValueError."""
    backend = _make_backend()
    bad = np.zeros((4, 2))  # horizon mismatch (expects 10 rows)
    with pytest.raises(ValueError):
        backend.rollout(controls=bad, horizon=10, dt=0.01)
