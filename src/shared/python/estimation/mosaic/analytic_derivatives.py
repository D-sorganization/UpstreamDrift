"""Analytic dynamics derivatives (Singh-Russell-Wensing) on the consistent inertial manifold (MOSAIC-08).

Implements:
- InverseDynamicsDerivatives: analytic Jacobians d_tau/dq, d_tau/dv, d_tau/da = M, d_tau/dpi = Y.
- ForwardDynamicsDerivatives: analytic Jacobians d_qddot/dq, d_qddot/dv, d_qddot/du, d_qddot/dpi,
  and d_qddot/dtheta on the consistent manifold chart.
- Fallback to high-order finite differences if a physics engine lacks analytic derivatives.
- Runtime profiling and verification against numerical differentiation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require
from src.shared.python.estimation.mosaic.inertial import (
    PLANAR_PARAMETERS_PER_BODY,
    InertialParameterization,
)
from src.shared.python.estimation.mosaic.planar_chain import PlanarChain

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _cross2(x: FloatArray, y: FloatArray) -> FloatArray:
    """Scalar z-component of planar cross product, batched on last axis."""
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InverseDynamicsDerivatives:
    """Analytic Jacobians of inverse dynamics tau(q, v, a, pi) = Y(q, v, a) pi."""

    d_tau_dq: FloatArray
    d_tau_dv: FloatArray
    d_tau_da: FloatArray
    d_tau_dpi: FloatArray

    def __post_init__(self) -> None:
        require(self.d_tau_dq.ndim == 3, "d_tau_dq must be (N, n, n)")
        require(self.d_tau_dv.shape == self.d_tau_dq.shape, "d_tau_dv shape mismatch")
        require(self.d_tau_da.shape == self.d_tau_dq.shape, "d_tau_da shape mismatch")
        require(
            self.d_tau_dpi.ndim == 3
            and self.d_tau_dpi.shape[:2] == self.d_tau_dq.shape[:2],
            "d_tau_dpi must be (N, n, n_pi)",
        )


@dataclass(frozen=True)
class ForwardDynamicsDerivatives:
    """Analytic Jacobians of forward dynamics q_ddot(q, v, u, pi) = M^{-1}(B u - h)."""

    q_ddot: FloatArray
    d_qddot_dq: FloatArray
    d_qddot_dv: FloatArray
    d_qddot_du: FloatArray
    d_qddot_dpi: FloatArray
    d_qddot_dtheta: FloatArray | None = None

    def __post_init__(self) -> None:
        require(self.q_ddot.ndim == 2, "q_ddot must be (N, n)")
        n_nodes, n_dof = self.q_ddot.shape
        require(
            self.d_qddot_dq.shape == (n_nodes, n_dof, n_dof),
            "d_qddot_dq must be (N, n, n)",
        )
        require(
            self.d_qddot_dv.shape == (n_nodes, n_dof, n_dof),
            "d_qddot_dv must be (N, n, n)",
        )
        require(
            self.d_qddot_du.ndim == 3 and self.d_qddot_du.shape[:2] == (n_nodes, n_dof),
            "d_qddot_du must be (N, n, m)",
        )
        require(
            self.d_qddot_dpi.ndim == 3
            and self.d_qddot_dpi.shape[:2] == (n_nodes, n_dof),
            "d_qddot_dpi must be (N, n, n_pi)",
        )
        if self.d_qddot_dtheta is not None:
            require(
                self.d_qddot_dtheta.ndim == 3
                and self.d_qddot_dtheta.shape[:2] == (n_nodes, n_dof),
                "d_qddot_dtheta must be (N, n, n_theta)",
            )


@dataclass(frozen=True)
class RuntimeComparison:
    """Benchmark comparing analytic derivatives against finite differences."""

    analytic_time_s: float
    finite_difference_time_s: float
    speedup: float
    max_relative_difference: float


# ---------------------------------------------------------------------------
# Analytic planar inverse dynamics derivatives (Singh-Russell-Wensing planar)
# ---------------------------------------------------------------------------


def _compute_planar_link_kinematics(
    chain: PlanarChain,
    q: FloatArray,
    v: FloatArray,
    a: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    batch = q.shape[0]
    phi = np.cumsum(q, axis=1)
    omega = np.cumsum(v, axis=1)
    alpha = np.cumsum(a, axis=1)
    e = np.stack([np.cos(phi), np.sin(phi)], axis=-1)
    normal = np.stack([-np.sin(phi), np.cos(phi)], axis=-1)
    lengths = chain.link_lengths[None, :, None]
    step = lengths * e
    origin = np.concatenate([np.zeros((batch, 1, 2)), np.cumsum(step, axis=1)], axis=1)
    acc_step = lengths * (alpha[..., None] * normal - (omega**2)[..., None] * e)
    origin_acc = np.concatenate(
        [np.zeros((batch, 1, 2)), np.cumsum(acc_step, axis=1)], axis=1
    )
    net_acc = origin_acc[:, :-1] - chain.gravity
    acc_hx = alpha[..., None] * normal - (omega**2)[..., None] * e
    acc_hy = -alpha[..., None] * e - (omega**2)[..., None] * normal
    return e, normal, origin, net_acc, acc_hx, acc_hy, omega, alpha


def _compute_planar_regressor_state_derivatives(
    chain: PlanarChain,
    q: FloatArray,
    v: FloatArray,
    a: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute d_Y_dq and d_Y_dv for planar serial chain."""
    batch = q.shape[0]
    n = chain.n_links
    lengths = chain.link_lengths[None, :, None]
    e, normal, origin, net_acc, acc_hx, acc_hy, omega, alpha = (
        _compute_planar_link_kinematics(chain, q, v, a)
    )

    # d_Y_dq and d_Y_dv
    d_Y_dq = np.zeros((batch, n, PLANAR_PARAMETERS_PER_BODY * n, n), dtype=np.float64)
    d_Y_dv = np.zeros((batch, n, PLANAR_PARAMETERS_PER_BODY * n, n), dtype=np.float64)

    for k in range(n):
        # Derivatives w.r.t q_k
        d_origin_k = np.zeros((batch, n + 1, 2), dtype=np.float64)
        d_origin_acc_qk = np.zeros((batch, n + 1, 2), dtype=np.float64)
        for m in range(k, n):
            d_origin_k[:, m + 1] = d_origin_k[:, m] + lengths[:, m] * normal[:, m]
            d_origin_acc_qk[:, m + 1] = d_origin_acc_qk[:, m] + lengths[:, m] * (
                -alpha[:, m, None] * e[:, m]
                - (omega[:, m] ** 2)[:, None] * normal[:, m]
            )

        # Derivatives w.r.t v_k
        d_origin_acc_vk = np.zeros((batch, n + 1, 2), dtype=np.float64)
        for m in range(k, n):
            d_origin_acc_vk[:, m + 1] = d_origin_acc_vk[:, m] + lengths[:, m] * (
                -2.0 * omega[:, m, None] * e[:, m]
            )

        for i in range(n):
            col = PLANAR_PARAMETERS_PER_BODY * i
            d_net_acc_qk = d_origin_acc_qk[:, i]
            d_net_acc_vk = d_origin_acc_vk[:, i]

            if k <= i:
                d_e_i_qk = normal[:, i]
                d_n_i_qk = -e[:, i]
                d_acc_hx_qk = acc_hy[:, i]
                d_acc_hy_qk = -acc_hx[:, i]
                d_acc_hx_vk = -2.0 * omega[:, i, None] * e[:, i]
                d_acc_hy_vk = -2.0 * omega[:, i, None] * normal[:, i]
            else:
                d_e_i_qk = np.zeros_like(normal[:, i])
                d_n_i_qk = np.zeros_like(e[:, i])
                d_acc_hx_qk = np.zeros_like(acc_hy[:, i])
                d_acc_hy_qk = np.zeros_like(acc_hx[:, i])
                d_acc_hx_vk = np.zeros_like(acc_hx[:, i])
                d_acc_hy_vk = np.zeros_like(acc_hy[:, i])

            for j in range(i + 1):
                lever = origin[:, i] - origin[:, j]
                d_lever_qk = d_origin_k[:, i] - d_origin_k[:, j]

                # q_k derivatives
                d_Y_dq[:, j, col, k] = _cross2(d_lever_qk, net_acc[:, i]) + _cross2(
                    lever, d_net_acc_qk
                )
                d_Y_dq[:, j, col + 1, k] = (
                    _cross2(d_e_i_qk, net_acc[:, i])
                    + _cross2(e[:, i], d_net_acc_qk)
                    + _cross2(d_lever_qk, acc_hx[:, i])
                    + _cross2(lever, d_acc_hx_qk)
                )
                d_Y_dq[:, j, col + 2, k] = (
                    _cross2(d_n_i_qk, net_acc[:, i])
                    + _cross2(normal[:, i], d_net_acc_qk)
                    + _cross2(d_lever_qk, acc_hy[:, i])
                    + _cross2(lever, d_acc_hy_qk)
                )

                # v_k derivatives
                d_Y_dv[:, j, col, k] = _cross2(lever, d_net_acc_vk)
                d_Y_dv[:, j, col + 1, k] = _cross2(e[:, i], d_net_acc_vk) + _cross2(
                    lever, d_acc_hx_vk
                )
                d_Y_dv[:, j, col + 2, k] = _cross2(
                    normal[:, i], d_net_acc_vk
                ) + _cross2(lever, d_acc_hy_vk)

    return d_Y_dq, d_Y_dv


def _planar_chain_inverse_dynamics_derivatives(
    chain: PlanarChain,
    q: FloatArray,
    v: FloatArray,
    a: FloatArray,
    pi: FloatArray,
) -> InverseDynamicsDerivatives:
    """Analytic inverse dynamics derivatives for planar serial chains."""
    batch = q.shape[0]
    n = chain.n_links
    require(q.shape == (batch, n), "q must be (N, n)", q.shape)
    require(v.shape == (batch, n), "v must be (N, n)", v.shape)
    require(a.shape == (batch, n), "a must be (N, n)", a.shape)
    require(
        pi.shape == (n * PLANAR_PARAMETERS_PER_BODY,), "pi shape mismatch", pi.shape
    )

    d_Y_dq, d_Y_dv = _compute_planar_regressor_state_derivatives(chain, q, v, a)

    # d_tau/dq = d_Y_dq @ pi: (N, n, n)
    d_tau_dq = np.einsum("njck,c->njk", d_Y_dq, pi)
    # d_tau/dv = d_Y_dv @ pi: (N, n, n)
    d_tau_dv = np.einsum("njck,c->njk", d_Y_dv, pi)
    # d_tau/da = M(q, pi): (N, n, n)
    d_tau_da = chain.mass_matrix(q, pi)
    # d_tau/dpi = Y(q, v, a): (N, n, n_pi)
    d_tau_dpi = chain.regressor(q, v, a)

    return InverseDynamicsDerivatives(
        d_tau_dq=d_tau_dq,
        d_tau_dv=d_tau_dv,
        d_tau_da=d_tau_da,
        d_tau_dpi=d_tau_dpi,
    )


# ---------------------------------------------------------------------------
# Numerical fallback for engines lacking analytic derivatives
# ---------------------------------------------------------------------------


def _finite_difference_inverse_dynamics_derivatives(
    model: Any,
    q: FloatArray,
    v: FloatArray,
    a: FloatArray,
    pi: FloatArray,
    step: float = 1e-7,
) -> InverseDynamicsDerivatives:
    """High-order central finite difference fallback."""
    batch, n_dof = q.shape
    d_tau_dq = np.zeros((batch, n_dof, n_dof), dtype=np.float64)
    d_tau_dv = np.zeros((batch, n_dof, n_dof), dtype=np.float64)

    for j in range(n_dof):
        dq = np.zeros_like(q)
        dq[:, j] = step
        tau_pos = model.regressor(q + dq, v, a) @ pi
        tau_neg = model.regressor(q - dq, v, a) @ pi
        d_tau_dq[..., j] = (tau_pos - tau_neg) / (2.0 * step)

        dv = np.zeros_like(v)
        dv[:, j] = step
        tau_pos_v = model.regressor(q, v + dv, a) @ pi
        tau_neg_v = model.regressor(q, v - dv, a) @ pi
        d_tau_dv[..., j] = (tau_pos_v - tau_neg_v) / (2.0 * step)

    if hasattr(model, "mass_matrix"):
        d_tau_da = model.mass_matrix(q, pi)
    else:
        d_tau_da = np.zeros((batch, n_dof, n_dof), dtype=np.float64)
        for j in range(n_dof):
            da = np.zeros_like(a)
            da[:, j] = step
            t_p = model.regressor(q, v, a + da) @ pi
            t_m = model.regressor(q, v, a - da) @ pi
            d_tau_da[..., j] = (t_p - t_m) / (2.0 * step)

    d_tau_dpi = model.regressor(q, v, a)
    return InverseDynamicsDerivatives(
        d_tau_dq=d_tau_dq,
        d_tau_dv=d_tau_dv,
        d_tau_da=d_tau_da,
        d_tau_dpi=d_tau_dpi,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inverse_dynamics_derivatives(
    model: Any,
    q: FloatArray,
    v: FloatArray,
    a: FloatArray,
    pi: FloatArray,
) -> InverseDynamicsDerivatives:
    """Compute analytic derivatives of inverse dynamics tau(q, v, a, pi).

    Dispatches to analytic routines when supported (e.g. PlanarChain or models exposing
    analytic derivatives) and falls back to central finite differences otherwise.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    pi_arr = np.asarray(pi, dtype=np.float64)

    require(
        q_arr.shape == v_arr.shape and q_arr.shape == a_arr.shape,
        "q, v, a must share the same shape (N, n)",
        (q_arr.shape, v_arr.shape, a_arr.shape),
    )

    if hasattr(model, "analytic_inverse_dynamics_derivatives"):
        return model.analytic_inverse_dynamics_derivatives(q_arr, v_arr, a_arr, pi_arr)
    if isinstance(model, PlanarChain):
        return _planar_chain_inverse_dynamics_derivatives(
            model, q_arr, v_arr, a_arr, pi_arr
        )
    return _finite_difference_inverse_dynamics_derivatives(
        model, q_arr, v_arr, a_arr, pi_arr
    )


def forward_dynamics_derivatives(
    model: Any,
    q: FloatArray,
    v: FloatArray,
    u: FloatArray,
    pi: FloatArray,
    parameterization: InertialParameterization | None = None,
    theta: FloatArray | None = None,
) -> ForwardDynamicsDerivatives:
    """Compute analytic derivatives of forward dynamics q_ddot(q, v, u, pi) = M^{-1}(B u - h).

    Uses the exact relations:
        d_qddot/dq = -M^{-1} [dh/dq + (dM/dq) q_ddot] = -M^{-1} [d_tau/dq]_{a=q_ddot}
        d_qddot/dv = -M^{-1} dh/dv = -M^{-1} [d_tau/dv]_{a=q_ddot}
        d_qddot/du = M^{-1} B
        d_qddot/dpi = -M^{-1} Y(q, v, q_ddot)
        d_qddot/dtheta = (d_qddot/dpi) (dpi/dtheta)
    """
    q_arr = np.asarray(q, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    u_arr = np.asarray(u, dtype=np.float64)
    pi_arr = np.asarray(pi, dtype=np.float64)

    batch, n_dof = q_arr.shape
    require(v_arr.shape == q_arr.shape, "v shape must match q", v_arr.shape)
    require(u_arr.shape[0] == batch, "u batch size must match q", u_arr.shape)

    # Forward dynamics acceleration
    q_ddot = model.forward_dynamics(q_arr, v_arr, u_arr, pi_arr)  # (N, n)
    mass = model.mass_matrix(q_arr, pi_arr)  # (N, n, n)

    # Inverse dynamics derivatives evaluated at a = q_ddot
    inv_derivs = inverse_dynamics_derivatives(model, q_arr, v_arr, q_ddot, pi_arr)

    # d_qddot / dq = -M^{-1} d_tau/dq
    d_qddot_dq = -np.linalg.solve(mass, inv_derivs.d_tau_dq)
    # d_qddot / dv = -M^{-1} d_tau/dv
    d_qddot_dv = -np.linalg.solve(mass, inv_derivs.d_tau_dv)

    # d_qddot / du = M^{-1} B
    input_mat = getattr(model, "input_matrix", np.eye(n_dof, dtype=np.float64))
    n_inputs = input_mat.shape[1]
    require(
        u_arr.shape[1] == n_inputs,
        "u column count must match model inputs",
        (u_arr.shape[1], n_inputs),
    )
    b_batched = np.broadcast_to(input_mat, (batch, n_dof, n_inputs))
    d_qddot_du = np.linalg.solve(mass, b_batched)

    # d_qddot / dpi = -M^{-1} Y(q, v, q_ddot)
    Y_qddot = model.regressor(q_arr, v_arr, q_ddot)
    d_qddot_dpi = -np.linalg.solve(mass, Y_qddot)

    # Manifold chart coordinates theta
    d_qddot_dtheta: FloatArray | None = None
    if parameterization is not None and theta is not None:
        theta_arr = np.asarray(theta, dtype=np.float64)
        dpi_dtheta = parameterization.jacobian(theta_arr)  # (n_pi, n_theta)
        d_qddot_dtheta = d_qddot_dpi @ dpi_dtheta

    return ForwardDynamicsDerivatives(
        q_ddot=q_ddot,
        d_qddot_dq=d_qddot_dq,
        d_qddot_dv=d_qddot_dv,
        d_qddot_du=d_qddot_du,
        d_qddot_dpi=d_qddot_dpi,
        d_qddot_dtheta=d_qddot_dtheta,
    )


def compare_runtime(
    model: Any,
    q: FloatArray,
    v: FloatArray,
    a: FloatArray,
    pi: FloatArray,
    n_repeats: int = 20,
) -> RuntimeComparison:
    """Benchmark analytic inverse dynamics derivatives against finite differences."""
    require(n_repeats >= 1, "n_repeats must be >= 1", n_repeats)
    q_arr = np.asarray(q, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    pi_arr = np.asarray(pi, dtype=np.float64)

    # Analytic timing
    t0 = time.perf_counter()
    analytic_res = None
    for _ in range(n_repeats):
        analytic_res = _planar_chain_inverse_dynamics_derivatives(
            model, q_arr, v_arr, a_arr, pi_arr
        )
    t_analytic = (time.perf_counter() - t0) / n_repeats

    # Finite difference timing
    t0 = time.perf_counter()
    fd_res = None
    for _ in range(n_repeats):
        fd_res = _finite_difference_inverse_dynamics_derivatives(
            model, q_arr, v_arr, a_arr, pi_arr
        )
    t_fd = (time.perf_counter() - t0) / n_repeats

    assert analytic_res is not None and fd_res is not None
    denom = np.maximum(1e-12, np.abs(analytic_res.d_tau_dq) + np.abs(fd_res.d_tau_dq))
    rel_diff = np.max(np.abs(analytic_res.d_tau_dq - fd_res.d_tau_dq) / denom)

    speedup = t_fd / max(1e-12, t_analytic)
    return RuntimeComparison(
        analytic_time_s=t_analytic,
        finite_difference_time_s=t_fd,
        speedup=speedup,
        max_relative_difference=float(rel_diff),
    )
