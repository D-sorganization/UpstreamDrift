"""Trajectory-varying event-conditioned control authority qualification (#9123).

Computes the discrete time-varying variational map z[k+1] = A[k] z[k] + B[k] v[k]
along the registered analytical double-pendulum downswing and conditions terminal
authority onto the transverse delivery event surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumDynamics,
    DoublePendulumParameters,
    DoublePendulumState,
)

FloatArray = npt.NDArray[np.float64]

INFERENCE_BOUNDARY = (
    "This qualification establishes only trajectory-varying linear variational "
    "authority and event-tangent conditioning for the declared analytical double pendulum. "
    "It cannot establish bounded nonlinear feasibility, human strength limits, "
    "muscle recruitment, passive biological torque, controller ranking, or coaching advice."
)


@dataclass(frozen=True, slots=True)
class TrajectoryAuthorityResult:
    """Reachability Gramian diagnostics for full state and event tangent space."""

    step_count: int
    dt_s: float
    nominal_event_time_s: float
    nominal_event_velocity: tuple[float, float, float, float]
    transverse_inner_product: float
    is_transverse: bool
    full_gramian_both: FloatArray
    full_gramian_shoulder: FloatArray
    full_gramian_wrist: FloatArray
    full_gramian_zero: FloatArray
    tangent_gramian_both: FloatArray
    tangent_gramian_shoulder: FloatArray
    tangent_gramian_wrist: FloatArray
    tangent_gramian_zero: FloatArray
    frozen_gramian_both: FloatArray
    full_rank_both: int
    full_rank_shoulder: int
    full_rank_wrist: int
    full_rank_zero: int
    tangent_rank_both: int
    tangent_rank_shoulder: int
    tangent_rank_wrist: int
    tangent_rank_zero: int
    additivity_residual_norm: float
    pulse_agreement_relative_error: float
    inference_boundary: str = INFERENCE_BOUNDARY


def continuous_dynamics(
    state: FloatArray,
    control: FloatArray,
    params: DoublePendulumParameters | None = None,
) -> FloatArray:
    """Continuous dynamics f(z, v) for the analytical double pendulum.

    State: [theta1, theta2, omega1, omega2] (angles and velocities).
    Control: [tau1, tau2] (shoulder, wrist torques).
    """
    p = params or DoublePendulumParameters.default()
    dyn = DoublePendulumDynamics(parameters=p)
    theta1, theta2, omega1, omega2 = state
    dp_state = DoublePendulumState(
        theta1=float(theta1),
        theta2=float(theta2),
        omega1=float(omega1),
        omega2=float(omega2),
    )
    drift, g_mat = dyn.control_affine(dp_state)
    u = np.array([float(control[0]), float(control[1])], dtype=np.float64)
    return np.array(drift, dtype=np.float64) + np.array(g_mat, dtype=np.float64) @ u


def discrete_rk4_step(
    state: FloatArray,
    control: FloatArray,
    dt: float,
    params: DoublePendulumParameters | None = None,
) -> FloatArray:
    """Exact discrete RK4 step for analytical double pendulum."""
    k1 = continuous_dynamics(state, control, params)
    k2 = continuous_dynamics(state + 0.5 * dt * k1, control, params)
    k3 = continuous_dynamics(state + 0.5 * dt * k2, control, params)
    k4 = continuous_dynamics(state + dt * k3, control, params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def discrete_state_jacobian(
    state: FloatArray,
    control: FloatArray,
    dt: float,
    params: DoublePendulumParameters | None = None,
    step: float = 1e-6,
) -> FloatArray:
    """Central-difference Jacobian A[k] = dF/dz of the discrete RK4 step."""
    dim = len(state)
    A = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        dx = np.zeros(dim, dtype=np.float64)
        dx[i] = step
        upper = discrete_rk4_step(state + dx, control, dt, params)
        lower = discrete_rk4_step(state - dx, control, dt, params)
        A[:, i] = (upper - lower) / (2.0 * step)
    return A


def discrete_control_jacobian(
    state: FloatArray,
    control: FloatArray,
    dt: float,
    params: DoublePendulumParameters | None = None,
    step: float = 1e-5,
) -> FloatArray:
    """Central-difference Jacobian B[k] = dF/dv of the discrete RK4 step."""
    state_dim = len(state)
    ctrl_dim = len(control)
    B = np.zeros((state_dim, ctrl_dim), dtype=np.float64)
    for j in range(ctrl_dim):
        du = np.zeros(ctrl_dim, dtype=np.float64)
        du[j] = step
        upper = discrete_rk4_step(state, control + du, dt, params)
        lower = discrete_rk4_step(state, control - du, dt, params)
        B[:, j] = (upper - lower) / (2.0 * step)
    return B


def transverse_event_projector(
    event_state: FloatArray,
    event_control: FloatArray,
    guard_gradient: FloatArray | None = None,
    params: DoublePendulumParameters | None = None,
    transverse_tolerance: float = 1e-3,
) -> tuple[FloatArray, bool, float]:
    """Build transverse event-state projector P = I - (f n^T)/(n^T f).

    For delivery event h(z) = theta1 = 0, n = [1, 0, 0, 0]^T.
    """
    n = (
        guard_gradient
        if guard_gradient is not None
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    f = continuous_dynamics(event_state, event_control, params)
    inner = float(np.dot(n, f))
    is_transverse = abs(inner) >= transverse_tolerance

    if not is_transverse:
        # Fails closed on near-grazing
        return np.eye(len(event_state), dtype=np.float64), False, inner

    P = np.eye(len(event_state), dtype=np.float64) - np.outer(f, n) / inner
    return P, True, inner


def orthonormal_tangent_basis(
    guard_gradient: FloatArray | None = None,
) -> FloatArray:
    """Return an orthonormal basis Q in R^(4x3) for the subspace orthogonal to n."""
    n = (
        guard_gradient
        if guard_gradient is not None
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    n_unit = n / np.linalg.norm(n)
    # Q consists of standard orthonormal basis vectors orthogonal to n_unit
    # For n = [1, 0, 0, 0], Q is [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    basis: list[FloatArray] = []
    eye = np.eye(len(n), dtype=np.float64)
    for i in range(len(n)):
        v = eye[:, i] - np.dot(eye[:, i], n_unit) * n_unit
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            v_unit = v / norm
            # Orthogonalize against existing basis
            for b in basis:
                v_unit -= np.dot(v_unit, b) * b
            norm_post = np.linalg.norm(v_unit)
            if norm_post > 1e-8:
                basis.append(v_unit / norm_post)
    return np.column_stack(basis[:3])


def matrix_rank(A: FloatArray, tol: float = 1e-9) -> int:
    """Compute numerical SVD rank."""
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > tol))


def compute_trajectory_authority(
    nominal_states: FloatArray,
    nominal_controls: FloatArray,
    dt: float,
    params: DoublePendulumParameters | None = None,
    guard_gradient: FloatArray | None = None,
) -> TrajectoryAuthorityResult:
    """Evaluate trajectory-varying reachability and event-conditioned Gramians."""
    K = len(nominal_controls)
    dim_x = nominal_states.shape[1]
    dim_u = nominal_controls.shape[1]

    # Compute Jacobians A[k] and B[k]
    A_list = [
        discrete_state_jacobian(nominal_states[k], nominal_controls[k], dt, params)
        for k in range(K)
    ]
    B_list = [
        discrete_control_jacobian(nominal_states[k], nominal_controls[k], dt, params)
        for k in range(K)
    ]

    # Compute propagated transition matrices Phi(K, k+1)
    # Phi(K, K) = I
    Phi_from_k1 = [np.eye(dim_x, dtype=np.float64) for _ in range(K + 1)]
    for k in range(K - 1, -1, -1):
        Phi_from_k1[k] = Phi_from_k1[k + 1] @ A_list[k]

    # Input sensitivities S[k] = Phi(K, k+1) * B[k]
    S_list = [Phi_from_k1[k + 1] @ B_list[k] for k in range(K)]

    # Compute Gramians for channel masks
    masks = {
        "both": np.eye(dim_u, dtype=np.float64),
        "shoulder": np.diag([1.0, 0.0]),
        "wrist": np.diag([0.0, 1.0]),
        "zero": np.zeros((dim_u, dim_u), dtype=np.float64),
    }

    full_gramians = {}
    for key, mask in masks.items():
        W = np.zeros((dim_x, dim_x), dtype=np.float64)
        for k in range(K):
            S_m = S_list[k] @ mask
            W += S_m @ S_m.T
        full_gramians[key] = W

    # Event projector at final state
    event_state = nominal_states[-1]
    event_control = nominal_controls[-1]
    P, is_transverse, inner = transverse_event_projector(
        event_state, event_control, guard_gradient, params
    )
    f_event = continuous_dynamics(event_state, event_control, params)

    tangent_gramians = {key: P @ full_gramians[key] @ P.T for key in masks}

    # Frozen-local countermodel: use state/control at midpoint
    mid = K // 2
    A0 = A_list[mid]
    B0 = B_list[mid]
    W_frozen = np.zeros((dim_x, dim_x), dtype=np.float64)
    A0_pow = np.eye(dim_x, dtype=np.float64)
    for _ in range(K):
        term = A0_pow @ B0
        W_frozen += term @ term.T
        A0_pow = A0_pow @ A0

    # Additivity check
    add_residual = float(
        np.linalg.norm(
            full_gramians["both"] - (full_gramians["shoulder"] + full_gramians["wrist"])
        )
    )

    # Finite-difference pulse validation at midpoint
    pulse_channel = 0
    pulse_magnitude = 1.0  # 1 N*m pulse for 1 step
    du = np.zeros(dim_u, dtype=np.float64)
    du[pulse_channel] = pulse_magnitude

    # Propagate linear prediction: delta_z = S[mid] * du
    linear_pred = S_list[mid] @ du

    # Non-linear direct propagation with pulse
    z_pert = nominal_states[mid].copy()
    z_pert = discrete_rk4_step(z_pert, nominal_controls[mid] + du, dt, params)
    for k in range(mid + 1, K):
        z_pert = discrete_rk4_step(z_pert, nominal_controls[k], dt, params)
    actual_delta = z_pert - nominal_states[-1]

    pulse_err = float(
        np.linalg.norm(actual_delta - linear_pred)
        / (np.linalg.norm(linear_pred) + 1e-12)
    )

    Q = orthonormal_tangent_basis(guard_gradient)
    tangent_ranks = {key: matrix_rank(Q.T @ tangent_gramians[key] @ Q) for key in masks}

    return TrajectoryAuthorityResult(
        step_count=K,
        dt_s=dt,
        nominal_event_time_s=float(K * dt),
        nominal_event_velocity=(
            float(f_event[0]),
            float(f_event[1]),
            float(f_event[2]),
            float(f_event[3]),
        ),
        transverse_inner_product=inner,
        is_transverse=is_transverse,
        full_gramian_both=full_gramians["both"],
        full_gramian_shoulder=full_gramians["shoulder"],
        full_gramian_wrist=full_gramians["wrist"],
        full_gramian_zero=full_gramians["zero"],
        tangent_gramian_both=tangent_gramians["both"],
        tangent_gramian_shoulder=tangent_gramians["shoulder"],
        tangent_gramian_wrist=tangent_gramians["wrist"],
        tangent_gramian_zero=tangent_gramians["zero"],
        frozen_gramian_both=W_frozen,
        full_rank_both=matrix_rank(full_gramians["both"]),
        full_rank_shoulder=matrix_rank(full_gramians["shoulder"]),
        full_rank_wrist=matrix_rank(full_gramians["wrist"]),
        full_rank_zero=matrix_rank(full_gramians["zero"]),
        tangent_rank_both=tangent_ranks["both"],
        tangent_rank_shoulder=tangent_ranks["shoulder"],
        tangent_rank_wrist=tangent_ranks["wrist"],
        tangent_rank_zero=tangent_ranks["zero"],
        additivity_residual_norm=add_residual,
        pulse_agreement_relative_error=pulse_err,
    )


def generate_nominal_downswing_trajectory(
    dt: float = 0.002,
    steps: int = 140,
    params: DoublePendulumParameters | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Generate a nominal downswing rollout for authority qualification."""
    p = params or DoublePendulumParameters.default()
    # Starting configuration: arm cocked, wrists hinged, zero velocity
    state = np.array([2.5, 1.5, 0.0, 0.0], dtype=np.float64)
    states = [state.copy()]
    controls = []

    for step_idx in range(steps):
        t = step_idx * dt
        # Nominal control profile: initial shoulder drive then late wrist uncocking
        tau1 = 150.0 * max(0.0, 1.0 - t / 0.25)
        tau2 = 20.0 * (1.0 if t > 0.15 else 0.0)
        u = np.array([tau1, tau2], dtype=np.float64)
        controls.append(u)
        state = discrete_rk4_step(state, u, dt, p)
        states.append(state.copy())

    return np.array(states), np.array(controls)
