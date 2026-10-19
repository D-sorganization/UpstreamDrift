"""Tests for MOSAIC analytic dynamics derivatives (Singh-Russell-Wensing)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pytest

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.mosaic.analytic_derivatives import (
    ForwardDynamicsDerivatives,
    InverseDynamicsDerivatives,
    _finite_difference_inverse_dynamics_derivatives,
    compare_runtime,
    forward_dynamics_derivatives,
    inverse_dynamics_derivatives,
)
from src.shared.python.estimation.mosaic.inertial import PlanarParameterization
from src.shared.python.estimation.mosaic.planar_chain import PlanarChain

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures and Helpers
# ---------------------------------------------------------------------------


def _double_pendulum() -> tuple[PlanarChain, np.ndarray, PlanarParameterization]:
    chain = PlanarChain(
        link_lengths=np.array([0.8, 0.5]),
        actuated=np.array([True, True]),
    )
    param = PlanarParameterization()
    # Physical theta coordinates for double pendulum
    theta = np.array(
        [
            np.log(1.2),
            0.3,
            0.0,
            np.log(0.08),
            np.log(0.6),
            0.15,
            0.0,
            np.log(0.03),
        ],
        dtype=np.float64,
    )
    pi = param.from_unconstrained(theta)
    return chain, pi, param


def _triple_pendulum() -> tuple[PlanarChain, np.ndarray, PlanarParameterization]:
    chain = PlanarChain(
        link_lengths=np.array([0.9, 0.6, 0.4]),
        actuated=np.array([False, True, True]),
    )
    param = PlanarParameterization()
    theta = np.array(
        [
            np.log(1.3),
            0.35,
            0.0,
            np.log(0.1),
            np.log(0.7),
            0.2,
            0.0,
            np.log(0.05),
            np.log(0.4),
            0.1,
            0.0,
            np.log(0.02),
        ],
        dtype=np.float64,
    )
    pi = param.from_unconstrained(theta)
    return chain, pi, param


# ---------------------------------------------------------------------------
# Inverse Dynamics Derivatives Tests
# ---------------------------------------------------------------------------


def test_inverse_dynamics_derivatives_conformance_with_finite_differences() -> None:
    chain, pi, _ = _double_pendulum()
    rng = np.random.default_rng(42)
    n_nodes = 5
    q = rng.uniform(-1.5, 1.5, size=(n_nodes, 2))
    v = rng.uniform(-2.0, 2.0, size=(n_nodes, 2))
    a = rng.uniform(-5.0, 5.0, size=(n_nodes, 2))

    derivs: InverseDynamicsDerivatives = inverse_dynamics_derivatives(
        chain, q, v, a, pi
    )
    assert derivs.d_tau_dq.shape == (n_nodes, 2, 2)
    assert derivs.d_tau_dv.shape == (n_nodes, 2, 2)
    assert derivs.d_tau_da.shape == (n_nodes, 2, 2)
    assert derivs.d_tau_dpi.shape == (n_nodes, 2, 8)

    # Verify d_tau_da equals M(q, pi)
    mass = chain.mass_matrix(q, pi)
    np.testing.assert_allclose(derivs.d_tau_da, mass, atol=1e-12)

    # Verify d_tau_dpi equals Y(q, v, a)
    Y = chain.regressor(q, v, a)
    np.testing.assert_allclose(derivs.d_tau_dpi, Y, atol=1e-12)

    # Verify d_tau_dq against central finite differences
    eps = 1e-7
    for j in range(2):
        dq = np.zeros_like(q)
        dq[:, j] = eps
        tau_pos = chain.regressor(q + dq, v, a) @ pi
        tau_neg = chain.regressor(q - dq, v, a) @ pi
        fd_q = (tau_pos - tau_neg) / (2.0 * eps)
        np.testing.assert_allclose(derivs.d_tau_dq[..., j], fd_q, rtol=1e-5, atol=1e-6)

    # Verify d_tau_dv against central finite differences
    for j in range(2):
        dv = np.zeros_like(v)
        dv[:, j] = eps
        tau_pos = chain.regressor(q, v + dv, a) @ pi
        tau_neg = chain.regressor(q, v - dv, a) @ pi
        fd_v = (tau_pos - tau_neg) / (2.0 * eps)
        np.testing.assert_allclose(derivs.d_tau_dv[..., j], fd_v, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Forward Dynamics Derivatives Tests
# ---------------------------------------------------------------------------


def test_forward_dynamics_derivatives_conformance_with_finite_differences() -> None:
    chain, pi, _ = _double_pendulum()
    rng = np.random.default_rng(99)
    n_nodes = 4
    q = rng.uniform(-1.0, 1.0, size=(n_nodes, 2))
    v = rng.uniform(-1.5, 1.5, size=(n_nodes, 2))
    u = rng.uniform(-2.0, 2.0, size=(n_nodes, 2))

    fd_derivs: ForwardDynamicsDerivatives = forward_dynamics_derivatives(
        chain, q, v, u, pi
    )
    assert fd_derivs.d_qddot_dq.shape == (n_nodes, 2, 2)
    assert fd_derivs.d_qddot_dv.shape == (n_nodes, 2, 2)
    assert fd_derivs.d_qddot_du.shape == (n_nodes, 2, 2)
    assert fd_derivs.d_qddot_dpi.shape == (n_nodes, 2, 8)

    eps = 1e-7

    # Check d_qddot_dq against finite differences
    for j in range(2):
        dq = np.zeros_like(q)
        dq[:, j] = eps
        qddot_pos = chain.forward_dynamics(q + dq, v, u, pi)
        qddot_neg = chain.forward_dynamics(q - dq, v, u, pi)
        fd_q = (qddot_pos - qddot_neg) / (2.0 * eps)
        np.testing.assert_allclose(
            fd_derivs.d_qddot_dq[..., j], fd_q, rtol=1e-5, atol=1e-6
        )

    # Check d_qddot_dv against finite differences
    for j in range(2):
        dv = np.zeros_like(v)
        dv[:, j] = eps
        qddot_pos = chain.forward_dynamics(q, v + dv, u, pi)
        qddot_neg = chain.forward_dynamics(q, v - dv, u, pi)
        fd_v = (qddot_pos - qddot_neg) / (2.0 * eps)
        np.testing.assert_allclose(
            fd_derivs.d_qddot_dv[..., j], fd_v, rtol=1e-5, atol=1e-6
        )

    # Check d_qddot_du against finite differences
    for j in range(2):
        du = np.zeros_like(u)
        du[:, j] = eps
        qddot_pos = chain.forward_dynamics(q, v, u + du, pi)
        qddot_neg = chain.forward_dynamics(q, v, u - du, pi)
        fd_u = (qddot_pos - qddot_neg) / (2.0 * eps)
        np.testing.assert_allclose(
            fd_derivs.d_qddot_du[..., j], fd_u, rtol=1e-5, atol=1e-6
        )

    # Check d_qddot_dpi against finite differences
    for k in range(8):
        dpi = np.zeros(8)
        dpi[k] = eps
        qddot_pos = chain.forward_dynamics(q, v, u, pi + dpi)
        qddot_neg = chain.forward_dynamics(q, v, u, pi - dpi)
        fd_pi = (qddot_pos - qddot_neg) / (2.0 * eps)
        np.testing.assert_allclose(
            fd_derivs.d_qddot_dpi[..., k], fd_pi, rtol=1e-5, atol=1e-6
        )


def test_forward_dynamics_inertial_manifold_derivatives_conformance() -> None:
    chain, pi, param = _double_pendulum()
    theta0 = np.array(
        [
            np.log(1.2),
            0.3,
            0.0,
            np.log(0.08),
            np.log(0.6),
            0.15,
            0.0,
            np.log(0.03),
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(7)
    q = rng.uniform(-1.0, 1.0, size=(3, 2))
    v = rng.uniform(-1.0, 1.0, size=(3, 2))
    u = rng.uniform(-1.0, 1.0, size=(3, 2))

    fd_derivs = forward_dynamics_derivatives(
        chain, q, v, u, pi, parameterization=param, theta=theta0
    )
    assert fd_derivs.d_qddot_dtheta is not None
    assert fd_derivs.d_qddot_dtheta.shape == (3, 2, 8)

    # Compare d_qddot_dtheta with finite differences on theta
    eps = 1e-7
    for k in range(8):
        dth = np.zeros(8)
        dth[k] = eps
        pi_pos = param.from_unconstrained(theta0 + dth)
        pi_neg = param.from_unconstrained(theta0 - dth)
        qddot_pos = chain.forward_dynamics(q, v, u, pi_pos)
        qddot_neg = chain.forward_dynamics(q, v, u, pi_neg)
        fd_theta = (qddot_pos - qddot_neg) / (2.0 * eps)
        np.testing.assert_allclose(
            fd_derivs.d_qddot_dtheta[..., k], fd_theta, rtol=1e-5, atol=1e-6
        )


def test_multi_link_chain_scaling() -> None:
    """Validate analytic derivatives on a 3-link chain with unactuated base."""
    chain, pi, param = _triple_pendulum()
    rng = np.random.default_rng(101)
    q = rng.uniform(-0.8, 0.8, size=(4, 3))
    v = rng.uniform(-1.2, 1.2, size=(4, 3))
    # Actuated has 2 inputs
    u = rng.uniform(-1.5, 1.5, size=(4, 2))

    inv_derivs = inverse_dynamics_derivatives(chain, q, v, np.zeros_like(q), pi)
    assert inv_derivs.d_tau_dq.shape == (4, 3, 3)
    assert inv_derivs.d_tau_dv.shape == (4, 3, 3)

    fwd_derivs = forward_dynamics_derivatives(chain, q, v, u, pi)
    assert fwd_derivs.d_qddot_dq.shape == (4, 3, 3)
    assert fwd_derivs.d_qddot_du.shape == (4, 3, 2)
    assert fwd_derivs.d_qddot_dpi.shape == (4, 3, 12)

    # Spot-check finite difference agreement
    eps = 1e-7
    for j in range(3):
        dq = np.zeros_like(q)
        dq[:, j] = eps
        qddot_pos = chain.forward_dynamics(q + dq, v, u, pi)
        qddot_neg = chain.forward_dynamics(q - dq, v, u, pi)
        fd_q = (qddot_pos - qddot_neg) / (2.0 * eps)
        np.testing.assert_allclose(
            fwd_derivs.d_qddot_dq[..., j], fd_q, rtol=1e-5, atol=1e-6
        )


def _runtime_states(
    n_dof: int, batch: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    q = rng.uniform(-1.0, 1.0, size=(batch, n_dof))
    v = rng.uniform(-1.0, 1.0, size=(batch, n_dof))
    a = rng.uniform(-2.0, 2.0, size=(batch, n_dof))
    return q, v, a


class _ScriptedClock:
    """Deterministic stand-in for ``time.perf_counter`` returning scripted ticks."""

    def __init__(self, ticks: list[float]) -> None:
        self._ticks = list(ticks)

    def __call__(self) -> float:
        return self._ticks.pop(0)

    @property
    def remaining(self) -> int:
        return len(self._ticks)


def test_runtime_comparison_reported() -> None:
    """The comparison report is well-formed; timing comes from an injected clock.

    Wall-clock speed ratios are nondeterministic on shared CI runners (#13390),
    so the unit lane checks structure and arithmetic only. The genuine
    analytic-is-faster claim lives in
    ``tests/benchmarks/test_mosaic_analytic_derivatives_benchmark.py``.
    """
    chain, pi, _ = _triple_pendulum()
    q, v, a = _runtime_states(chain.n_links)
    n_repeats = 4
    # analytic: start 10.0, stop 11.0 -> 1.0 s total; FD: 20.0 -> 24.0 = 4.0 s
    clock = _ScriptedClock([10.0, 11.0, 20.0, 24.0])

    report = compare_runtime(chain, q, v, a, pi, n_repeats=n_repeats, clock=clock)

    assert clock.remaining == 0
    assert report.analytic_time_s == pytest.approx(1.0 / n_repeats)
    assert report.finite_difference_time_s == pytest.approx(4.0 / n_repeats)
    assert report.speedup == pytest.approx(4.0)
    assert report.max_relative_difference < 1e-4


def test_runtime_comparison_defaults_to_wall_clock() -> None:
    """Without an injected clock the report carries positive measured times."""
    chain, pi, _ = _double_pendulum()
    q, v, a = _runtime_states(chain.n_links, batch=2)

    report = compare_runtime(chain, q, v, a, pi, n_repeats=1)

    assert report.analytic_time_s > 0.0
    assert report.finite_difference_time_s > 0.0
    assert report.speedup > 0.0


@dataclass(frozen=True)
class _CountingPlanarChain(PlanarChain):
    """PlanarChain that counts regressor evaluations (the dominant cost)."""

    calls: Counter[str] = field(default_factory=Counter, compare=False, repr=False)

    def regressor(self, *args: object, **kwargs: object) -> np.ndarray:
        self.calls["regressor"] += 1
        return super().regressor(*args, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("link_lengths", [[0.8, 0.5], [0.9, 0.6, 0.4]])
def test_analytic_derivatives_need_fewer_dynamics_evaluations(
    link_lengths: list[float],
) -> None:
    """Deterministic proxy for the speedup claim: analytic does less model work.

    Analytic: one regressor for d_tau/dpi plus n for the mass matrix (n + 1).
    Central finite differences: 4n perturbed regressors (q and v, +/-), n for
    the mass matrix and one for d_tau/dpi (5n + 1).
    """
    n_dof = len(link_lengths)
    _, pi, _ = _triple_pendulum() if n_dof == 3 else _double_pendulum()
    chain = _CountingPlanarChain(
        link_lengths=np.array(link_lengths),
        actuated=np.ones(n_dof, dtype=bool),
    )
    q, v, a = _runtime_states(n_dof, batch=3)

    inverse_dynamics_derivatives(chain, q, v, a, pi)
    analytic_calls = chain.calls["regressor"]
    chain.calls.clear()
    _finite_difference_inverse_dynamics_derivatives(chain, q, v, a, pi)
    fd_calls = chain.calls["regressor"]

    assert analytic_calls == n_dof + 1
    assert fd_calls == 5 * n_dof + 1
    assert analytic_calls < fd_calls


def test_fallback_to_finite_differences_when_engine_lacks_analytic() -> None:
    """A model without analytic derivatives falls back gracefully to finite differences."""
    chain, pi, _ = _double_pendulum()

    class MockNumericOnlyModel:
        def __init__(self, inner: PlanarChain):
            self._inner = inner
            self.n_links = inner.n_links
            self.input_matrix = inner.input_matrix

        def regressor(self, q, v, a, gravity=None):
            return self._inner.regressor(q, v, a, gravity=gravity)

        def mass_matrix(self, q, pi):
            return self._inner.mass_matrix(q, pi)

        def forward_dynamics(self, q, v, u, pi):
            return self._inner.forward_dynamics(q, v, u, pi)

    mock_model = MockNumericOnlyModel(chain)
    q = np.array([[0.2, -0.3], [0.5, 0.1]])
    v = np.array([[0.1, 0.4], [-0.2, 0.2]])
    a = np.array([[0.0, -0.1], [0.3, 0.0]])

    derivs = inverse_dynamics_derivatives(mock_model, q, v, a, pi)
    assert derivs.d_tau_dq.shape == (2, 2, 2)
    assert derivs.d_tau_dv.shape == (2, 2, 2)


def test_preconditions() -> None:
    chain, pi, _ = _double_pendulum()
    q = np.zeros((3, 2))
    v = np.zeros((3, 2))
    a = np.zeros((2, 2))  # mismatched batch size

    with pytest.raises(ContractViolationError):
        inverse_dynamics_derivatives(chain, q, v, a, pi)

    with pytest.raises(ContractViolationError):
        forward_dynamics_derivatives(
            chain, q, v, np.zeros((3, 3)), pi
        )  # wrong input dim
