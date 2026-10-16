"""Shared models for the NSX-90 covariance-propagation tests.

The pendulum chain turns the real NSX-84 tangent linearization (reaction
re-solved, geometric preload included) into the explicit-Euler discrete
``(A, B)`` of ``loop_nullspace.tex`` ``sec:var-cov``; the noise data are the
declared hypotheses stored in the NSX-80 golden.
"""

from __future__ import annotations

import functools

import numpy as np

from src.shared.python.estimation.mosaic.local_policy import LinearizedDynamics
from src.shared.python.loop_nullspace import tangent_linearization as tl
from tests.unit.loop_nullspace._tangent_fixtures import (
    Pendulum,
    frac,
    golden,
    tangent_field,
)

EUCLID2 = tl.EuclideanManifold(2)


@functools.cache
def _noise_cached() -> dict[str, np.ndarray | float]:
    return _build_noise()


def noise() -> dict[str, np.ndarray | float]:
    """Declared NSX-80 noise data as floats (copies; safe to mutate)."""
    return {
        k: np.copy(v) if isinstance(v, np.ndarray) else v
        for k, v in _noise_cached().items()
    }


def _build_noise() -> dict[str, np.ndarray | float]:
    """Declared NSX-80 noise data as floats."""
    decl = golden()["model"]["noise_declared"]
    return {
        "dt": frac(decl["dt"]),
        "c": np.array([frac(x) for x in decl["c_outcome"]]),
        "sigma0": np.array([frac(x) for x in decl["sigma0"]]),
        "k": np.array([frac(x) for x in decl["k"]]),
        "p0": np.diag([frac(x) for x in decl["P0_diag"]]),
    }


def state_space_at(z: float, *, kind: str = "assembled") -> tl.TangentStateSpace:
    """Real ``(F_t, G_t)`` of the pendulum for the feed-forward ``u(z)``."""
    pend = Pendulum()
    q, v = pend.state()
    cls = (
        tl.AssembledKKTDerivatives
        if kind == "assembled"
        else tl.FiniteDifferenceDerivatives
    )
    prov = cls(pend, EUCLID2, basis_field=tangent_field)
    derivs = prov.constrained_acceleration_derivatives(q, v, pend.load(z))
    return tl.tangent_state_space(
        derivs, q=q, v=v, manifold=EUCLID2, basis_field=tangent_field
    )


def one_step(f_mat: np.ndarray, g_mat: np.ndarray, dt: float) -> LinearizedDynamics:
    """Explicit-Euler single step ``A = I + dt F``, ``B = dt G``."""
    a = np.eye(f_mat.shape[0]) + dt * f_mat
    return LinearizedDynamics(a[None], (dt * g_mat)[None])


def pendulum_dynamics(z: float, dt: float, n_steps: int = 1) -> LinearizedDynamics:
    """Constant-``(A, B)`` chain from the real linearization at allocation ``z``."""
    ss = state_space_at(z)
    one = one_step(ss.F, ss.G, dt)
    return LinearizedDynamics(
        np.repeat(one.a, n_steps, axis=0), np.repeat(one.b, n_steps, axis=0)
    )


def sdn_sigma(u_z: np.ndarray, n_steps: int = 1) -> np.ndarray:
    """Per-step ``diag(sigma0^2 + k^2 u^2)`` (golden M1 law), ``(T, 2, 2)``."""
    nz = noise()
    var = nz["sigma0"] ** 2 + nz["k"] ** 2 * np.asarray(u_z) ** 2
    return np.repeat(np.diag(var)[None], n_steps, axis=0)


def additive_sigma(n_steps: int = 1) -> np.ndarray:
    nz = noise()
    return np.repeat(np.diag(nz["sigma0"] ** 2)[None], n_steps, axis=0)


def pendulum_nonlinear_step(
    x: np.ndarray, force: np.ndarray, u_ff: np.ndarray, dt: float
) -> np.ndarray:
    """Explicit-Euler nonlinear pendulum step, state ``(theta, omega)``.

    ``alpha = (t(theta) . (u_ff + force) - m g sin theta) / (m L)`` is the exact
    reduced-coordinate acceleration the NSX-80 golden is checked against, so the
    Jacobian of this step at the nominal point is the discrete ``(A, B)``.
    """
    m, length, g = Pendulum.m, Pendulum.length, Pendulum.g
    th, om = x
    t_hat = np.array([np.cos(th), np.sin(th)])
    alpha = (t_hat @ (u_ff + force) - m * g * np.sin(th)) / (m * length)
    return np.array([th + dt * om, om + dt * alpha])


def nominal_angle() -> float:
    return float(np.arcsin(0.6))
