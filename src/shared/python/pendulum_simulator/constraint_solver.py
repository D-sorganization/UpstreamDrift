"""
Constraint enforcement for the golfer closed kinematic loop.

Uses Baumgarte stabilization to maintain the holonomic constraints
during ODE integration.  The augmented equations of motion become:

    M * qddot + C + G = tau + Phi_q^T * lambda
    Phi_q * qddot = -Phi_qq * qdot * qdot - 2*alpha*Phi_q*qdot - beta^2*Phi

where lambda is the vector of constraint (Lagrange) multiplier forces,
alpha and beta are stabilization gains, and Phi is the constraint vector.

Design by Contract
------------------
- All inputs validated with assertions.
- Constraint Jacobian must have full row rank for the system to be solvable.

DRY
---
Delegates constraint evaluation to physics_golfer.constraint_vector
and constraint_jacobian.
"""

from __future__ import annotations

import logging

import numpy as np

from . import native_backend as _native_backend
from .golfer_constraints import analytical_constraint_jacobian as constraint_jacobian
from .golfer_constraints import (
    constraint_vector,
    friction_torque_vector,
)
from .golfer_dynamics import analytical_coriolis as coriolis_matrix
from .golfer_dynamics import analytical_gravity_vector as gravity_vector
from .golfer_dynamics import analytical_mass_matrix as mass_matrix
from .physics_golfer import (
    N_CONSTRAINTS,
    N_DOF,
    GolferParams,
    State,
    TorqueFunc,
)

logger = logging.getLogger(__name__)

# Baumgarte stabilization gains (default values)
DEFAULT_ALPHA = 5.0
DEFAULT_BETA = 5.0


def _solve_constrained_dynamics(
    state: State,
    t: float,
    params: GolferParams,
    torque_func: TorqueFunc,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    torque_limits: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the augmented KKT system for accelerations and multipliers.

    Solves the KKT system:
        [M    Phi_q^T] [qddot ] = [tau + tau_f - C - G               ]
        [Phi_q  0    ] [lambda ] = [-gamma - 2*alpha*Phi_dot - beta^2*Phi]

    Parameters
    ----------
    state : np.ndarray, shape (16,)
        [q (8), qdot (8)]
    t : float
        Current time
    params : GolferParams
    torque_func : callable
    alpha, beta : float
        Baumgarte stabilization gains

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Joint accelerations and constraint multipliers.
    """
    assert state.shape == (2 * N_DOF,), f"State shape must be ({2 * N_DOF},)"
    assert np.all(np.isfinite(state)), "State must be finite"

    q = state[:N_DOF]
    qdot = state[N_DOF:]

    # Applied torques (7 joint torques, club DOF has no independent torque)
    tau_tuple = torque_func(t)
    tau = np.zeros(N_DOF)
    tau[:7] = tau_tuple

    # Torque saturation (#1150)
    if torque_limits is not None:
        from .physics import clamp_torque_ndof

        tau[:7] = clamp_torque_ndof(tau[:7], torque_limits[:7])

    native_result = _native_backend.golfer_constrained_dynamics(
        q, qdot, tau, params, alpha, beta
    )
    if native_result is not None:
        qddot, lambda_forces = native_result
        assert np.all(np.isfinite(qddot)), f"qddot has non-finite values: {qddot}"
        assert np.all(np.isfinite(lambda_forces)), (
            f"Constraint forces have non-finite values: {lambda_forces}"
        )
        return qddot, lambda_forces

    # Compute dynamic terms
    M = mass_matrix(q, params)
    C = coriolis_matrix(q, qdot, params)
    G = gravity_vector(q, params)

    tau_f = friction_torque_vector(qdot, params)

    # Right-hand side of unconstrained EOM
    rhs_dyn = tau + tau_f - C - G

    # Constraint terms
    Phi = constraint_vector(q, params)
    Phi_q = constraint_jacobian(q, params)

    # Constraint velocity: Phi_dot = Phi_q * qdot
    Phi_dot = Phi_q @ qdot

    # Gamma term: Phi_qq * qdot * qdot (computed numerically)
    gamma = _constraint_acceleration_bias(q, qdot, params)

    # Baumgarte RHS
    rhs_constraint = -gamma - 2.0 * alpha * Phi_dot - beta**2 * Phi

    # Assemble KKT system
    n = N_DOF
    m = N_CONSTRAINTS
    KKT = np.zeros((n + m, n + m))
    KKT[:n, :n] = M
    KKT[:n, n:] = Phi_q.T
    KKT[n:, :n] = Phi_q

    rhs = np.zeros(n + m)
    rhs[:n] = rhs_dyn
    rhs[n:] = rhs_constraint

    # Solve KKT system
    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        logger.warning("KKT system singular, falling back to least-squares solver")
        # Fallback: use least-squares if KKT is singular
        sol, _, _, _ = np.linalg.lstsq(KKT, rhs, rcond=None)

    qddot = sol[:n]
    lambda_forces = sol[n:]

    assert np.all(np.isfinite(qddot)), f"qddot has non-finite values: {qddot}"
    assert np.all(np.isfinite(lambda_forces)), (
        f"Constraint forces have non-finite values: {lambda_forces}"
    )
    return qddot, lambda_forces


def constrained_accelerations(
    state: State,
    t: float,
    params: GolferParams,
    torque_func: TorqueFunc,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    torque_limits: np.ndarray | None = None,
) -> np.ndarray:
    """Compute constrained accelerations using augmented Lagrangian method."""
    assert state is not None, "state must be provided"
    qddot, _ = _solve_constrained_dynamics(
        state, t, params, torque_func, alpha, beta, torque_limits
    )
    return qddot


def constraint_forces(
    state: State,
    t: float,
    params: GolferParams,
    torque_func: TorqueFunc,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> np.ndarray:
    """Compute the constraint (Lagrange multiplier) forces.

    Returns
    -------
    lambda_vec : np.ndarray, shape (4,) â€” constraint forces
    """
    assert state is not None, "state must be provided"
    _, lambda_forces = _solve_constrained_dynamics(
        state, t, params, torque_func, alpha, beta
    )
    return lambda_forces


def _constraint_acceleration_bias(
    q: np.ndarray, qdot: np.ndarray, params: GolferParams
) -> np.ndarray:
    """Compute gamma = d(Phi_q)/dt * qdot numerically.

    This is the acceleration-level bias term from the constraint.
    """
    assert q is not None, "q must be provided"
    eps = 1e-7
    Phi_q_0 = constraint_jacobian(q, params)
    Phi_q_dt = constraint_jacobian(q + eps * qdot, params)
    Phi_q_dot = (Phi_q_dt - Phi_q_0) / eps
    result: np.ndarray = Phi_q_dot @ qdot
    return result


def analytical_constraint_acceleration_bias(
    q: np.ndarray, qdot: np.ndarray, params: GolferParams
) -> np.ndarray:
    """Backward-compatible alias for the constraint acceleration bias term.

    Despite the historical name, the current implementation still uses a
    numerical directional derivative of the constraint Jacobian. The public
    symbol is kept for API compatibility while the solver delegates to the
    shared bias helper above.
    """
    return _constraint_acceleration_bias(q, qdot, params)


def equations_of_motion(
    state: State,
    t: float,
    params: GolferParams,
    torque_func: TorqueFunc,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    torque_limits: np.ndarray | None = None,
) -> State:
    """State derivative for the constrained golfer model.

    dx/dt = [qdot, qddot] where qddot satisfies the constrained EOM.

    Parameters
    ----------
    state : np.ndarray, shape (16,)
    t : float
    params : GolferParams
    torque_func : callable (t) -> 7-tuple
    torque_limits : np.ndarray, shape (7,), optional
        Per-joint torque saturation limits (#1150).

    Returns
    -------
    state_dot : np.ndarray, shape (16,)
    """
    assert state.shape == (2 * N_DOF,), f"State shape must be ({2 * N_DOF},)"
    assert np.all(np.isfinite(state)), f"State has non-finite values: {state}"

    qdot = state[N_DOF:]
    qddot = constrained_accelerations(
        state, t, params, torque_func, alpha, beta, torque_limits
    )

    state_dot = np.zeros(2 * N_DOF)
    state_dot[:N_DOF] = qdot
    state_dot[N_DOF:] = qddot

    return state_dot


def constraint_violation(state: State, params: GolferParams) -> float:
    """Compute the L2 norm of the constraint violation.

    Useful for monitoring constraint drift during simulation.

    Returns
    -------
    float â€” ||Phi(q)||_2
    """
    assert state is not None, "state must be provided"
    q = state[:N_DOF]
    Phi = constraint_vector(q, params)
    return float(np.linalg.norm(Phi))


def project_to_constraints(
    q: np.ndarray,
    params: GolferParams,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> np.ndarray:
    """Project coordinates onto the constraint manifold using Newton's method.

    Solves Phi(q) = 0 by iterating:
        q_new = q - Phi_q^+ * Phi(q)
    where Phi_q^+ is the pseudoinverse of the constraint Jacobian.

    Parameters
    ----------
    q : np.ndarray, shape (8,)
    params : GolferParams
    max_iter : int
    tol : float

    Returns
    -------
    q_projected : np.ndarray, shape (8,)
    """
    assert q.shape == (N_DOF,), f"q must have shape ({N_DOF},), got {q.shape}"
    assert np.all(np.isfinite(q)), "q must be finite"
    assert max_iter > 0, f"max_iter must be positive, got {max_iter}"
    assert tol > 0, f"tol must be positive, got {tol}"

    native_projection = _native_backend.golfer_project_to_constraints(
        q, params, max_iter, tol
    )
    if native_projection is not None:
        residual = float(np.linalg.norm(constraint_vector(native_projection, params)))
        if residual < tol:
            return native_projection

    q = q.copy()
    for _ in range(max_iter):
        Phi = constraint_vector(q, params)
        if np.linalg.norm(Phi) < tol:
            return q
        Phi_q = constraint_jacobian(q, params)
        # Use pseudoinverse for robustness
        dq = Phi_q.T @ np.linalg.solve(
            Phi_q @ Phi_q.T + 1e-12 * np.eye(N_CONSTRAINTS), Phi
        )
        q -= dq

    residual = float(np.linalg.norm(constraint_vector(q, params)))
    raise RuntimeError(
        "Constraint projection did not converge "
        f"within {max_iter} iterations (residual={residual:.3e})"
    )


def project_velocity(
    q: np.ndarray,
    qdot: np.ndarray,
    params: GolferParams,
) -> np.ndarray:
    """Project velocity to satisfy Phi_q * qdot = 0.

    Minimum-norm correction: qdot_new = qdot - Phi_q^+ * (Phi_q * qdot)

    Returns
    -------
    qdot_projected : np.ndarray, shape (8,)
    """
    assert q is not None, "q must be provided"
    native_projection = _native_backend.golfer_project_velocity(q, qdot, params)
    if native_projection is not None:
        native_violation = constraint_jacobian(q, params) @ native_projection
        if np.linalg.norm(native_violation) < 1e-6:
            return native_projection

    Phi_q = constraint_jacobian(q, params)
    violation = Phi_q @ qdot
    # Pseudoinverse correction
    correction = Phi_q.T @ np.linalg.solve(
        Phi_q @ Phi_q.T + 1e-12 * np.eye(N_CONSTRAINTS), violation
    )
    projected: np.ndarray = qdot - correction
    return projected
