"""
double_pendulum.py

Planar double pendulum model with:

- Dynamics: x_dot = f(x, u)
- Natural torque field: tau_nat(q, qdot, qddot)
- Example PD input and simulation
- Optional end-effector wrench reconstruction

State vector:
    x = [q1, q2, q1dot, q2dot]

This file is intended as a standalone module that you can
import into a Streamlit app or run directly.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from shared.python.constants import GRAVITY_M_S2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical parameters for the double pendulum
# ---------------------------------------------------------------------------

m1 = 1.0  # mass of link 1
m2 = 1.0  # mass of link 2
l1 = 1.0  # length of link 1
l2 = 1.0  # length of link 2
c1 = 0.5  # COM distance of link 1 from joint 1
c2 = 0.5  # COM distance of link 2 from joint 2
I1 = 0.05  # inertia of link 1 about its COM (out of plane)
I2 = 0.05  # inertia of link 2 about its COM (out of plane)
GRAVITY = GRAVITY_M_S2  # gravity from shared constants


# ---------------------------------------------------------------------------
# Inertia matrix M(q)
# ---------------------------------------------------------------------------


def M_matrix(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Inertia matrix M(q) for the planar 2-link manipulator / double pendulum.

    Parameters
    ----------
    q : array_like, shape (2,)
        Joint angles [q1, q2].

    Returns
    -------
    M : ndarray, shape (2, 2)
        Inertia matrix.
    """
    q1, q2 = q
    cos2 = np.cos(q2)

    m11 = I1 + I2 + m1 * c1**2 + m2 * (l1**2 + c2**2 + 2 * l1 * c2 * cos2)
    m12 = I2 + m2 * (c2**2 + l1 * c2 * cos2)
    m21 = m12
    m22 = I2 + m2 * c2**2

    return np.array([[m11, m12], [m21, m22]], dtype=float)


# ---------------------------------------------------------------------------
# Coriolis / centrifugal term C(q, qdot) * qdot
# ---------------------------------------------------------------------------


def C_times_qdot(
    q: npt.NDArray[np.float64], qdot: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute C(q, qdot) * qdot for the double pendulum.

    Parameters
    ----------
    q : array_like, shape (2,)
        Joint angles [q1, q2].
    qdot : array_like, shape (2,)
        Joint velocities [q1dot, q2dot].

    Returns
    -------
    Cq : ndarray, shape (2,)
        Vector C(q, qdot) * qdot.
    """
    q1, q2 = q
    q1dot, q2dot = qdot

    # Common factor
    h = -m2 * l1 * c2 * np.sin(q2)

    # This is the product C(q, qdot) * qdot
    c1_term = h * q2dot * (2.0 * q1dot + q2dot)
    c2_term = -h * q1dot**2

    return np.array([c1_term, c2_term], dtype=float)


# ---------------------------------------------------------------------------
# Gravity term g(q)
# ---------------------------------------------------------------------------


def g_vector(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Gravity torque vector g(q) for the double pendulum.

    Parameters
    ----------
    q : array_like, shape (2,)
        Joint angles [q1, q2].

    Returns
    -------
    gq : ndarray, shape (2,)
        Gravity torques [g1, g2].
    """
    q1, q2 = q

    g1 = (m1 * c1 + m2 * l1) * GRAVITY * np.sin(q1) + m2 * c2 * GRAVITY * np.sin(q1 + q2)
    g2 = m2 * c2 * GRAVITY * np.sin(q1 + q2)

    return np.array([g1, g2], dtype=float)


# ---------------------------------------------------------------------------
# Natural torque field tau_nat(q, qdot, qddot)
# ---------------------------------------------------------------------------


def tau_natural(
    q: npt.NDArray[np.float64],
    qdot: npt.NDArray[np.float64],
    qddot: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Natural torque field for the double pendulum:

        tau_nat = M(q) * qddot + C(q, qdot) * qdot + g(q)

    This is the "natural" or "drift" torque field arising from inertia,
    Coriolis/centrifugal effects, and gravity for the given motion.

    Parameters
    ----------
    q : array_like, shape (2,)
        Joint angles [q1, q2].
    qdot : array_like, shape (2,)
        Joint velocities [q1dot, q2dot].
    qddot : array_like, shape (2,)
        Joint accelerations [q1ddot, q2ddot].

    Returns
    -------
    tau_nat : ndarray, shape (2,)
        Natural torque vector.
    """
    mq = M_matrix(q) @ qddot
    cq = C_times_qdot(q, qdot)
    gq = g_vector(q)
    return mq + cq + gq


# ---------------------------------------------------------------------------
# State dynamics: x_dot = f(x, u)
# ---------------------------------------------------------------------------


def double_pendulum_dynamics(
    t: float,
    x: npt.NDArray[np.float64],
    u_func: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """
    State-space dynamics for the double pendulum.

    State x = [q1, q2, q1dot, q2dot].
    Inputs u = [u1, u2] are *joint torques* (fully actuated).

    Dynamics:
        qddot = M(q)^(-1) [ u - C(q, qdot) qdot - g(q) ]

    Parameters
    ----------
    t : float
        Time.
    x : ndarray, shape (4,)
        State vector [q1, q2, q1dot, q2dot].
    u_func : callable
        Function u_func(t, x) -> np.array([u1, u2]) giving active torques.

    Returns
    -------
    xdot : ndarray, shape (4,)
        Time derivative of the state.
    """
    q = x[0:2]
    qdot = x[2:4]

    u = u_func(t, x)  # active torques

    mq = M_matrix(q)
    cq = C_times_qdot(q, qdot)
    gq = g_vector(q)

    # qddot = M^-1 (u - C(q,qdot) qdot - g(q))
    qddot = np.linalg.solve(mq, u - cq - gq)

    return np.concatenate([qdot, qddot])


# ---------------------------------------------------------------------------
# Example input: simple PD around q = [0, 0]
# ---------------------------------------------------------------------------


def u_pd(
    _t: float,
    x: npt.NDArray[np.float64],
    kp: float = 10.0,
    kd: float = 2.0,
) -> npt.NDArray[np.float64]:
    """
    Simple PD control law around the downward configuration q = [0, 0].

    This is NOT meant to be biomechanically realistic; it's just an example
    input to get non-trivial dynamics and torque fields.

    Parameters
    ----------
    t : float
        Time (unused here but included for signature compatibility).
    x : ndarray, shape (4,)
        State vector [q1, q2, q1dot, q2dot].
    kp : float
        Proportional gain.
    kd : float
        Derivative gain.

    Returns
    -------
    u : ndarray, shape (2,)
        Joint torques [u1, u2].
    """
    q = x[0:2]
    qdot = x[2:4]

    q_des = np.array([0.0, 0.0], dtype=np.float64)
    qdot_des = np.array([0.0, 0.0], dtype=np.float64)

    result = -kp * (q - q_des) - kd * (qdot - qdot_des)
    return np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# Trajectory post-processing: reconstruct tau_nat(t)
# ---------------------------------------------------------------------------


def compute_tau_natural_trajectory(
    sol: Any,  # scipy.integrate.OdeResult doesn't have proper type stubs
    u_func: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Given a solution object `sol` from solve_ivp and an input function u_func,
    compute tau_nat(t) at each time sample.

    Parameters
    ----------
    sol : OdeSolution
        Solution object returned by solve_ivp.
    u_func : callable
        Function u_func(t, x) -> u used in the simulation.

    Returns
    -------
    t : ndarray, shape (N,)
        Time samples.
    tau_nat_traj : ndarray, shape (N, 2)
        Natural torque trajectory at each time sample.
    """
    t = np.asarray(sol.t, dtype=np.float64)
    x = np.asarray(sol.y.T, dtype=np.float64)
    N = x.shape[0]
    tau_nat_traj = np.zeros((N, 2), dtype=np.float64)

    # Optimization: The dynamics equation is:
    #   qddot = M^-1 (u - C qdot - g)
    # The natural torque definition is:
    #   tau_nat = M qddot + C qdot + g
    # Substituting qddot gives:
    #   tau_nat = M (M^-1 (u - C qdot - g)) + C qdot + g
    #           = (u - C qdot - g) + C qdot + g
    #           = u
    # Therefore, we can skip the expensive matrix calculations and just evaluate u_func.

    for i in range(N):
        ti = t[i]
        xi = x[i]
        tau_nat_traj[i, :] = u_func(ti, xi)

    return t, tau_nat_traj


# ---------------------------------------------------------------------------
# End-effector Jacobian and wrench reconstruction (optional)
# ---------------------------------------------------------------------------


def J_end_effector(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Planar end-effector Jacobian for the tip of link 2.

    We treat the end-effector twist as [vx, vy, omega_z]^T.

    Parameters
    ----------
    q : array_like, shape (2,)
        Joint angles [q1, q2].

    Returns
    -------
    J : ndarray, shape (3, 2)
        Jacobian such that v = J(q) qdot, with v = [vx, vy, omega_z].
    """
    q1, q2 = q
    s1 = np.sin(q1)
    c1 = np.cos(q1)
    s12 = np.sin(q1 + q2)
    c12 = np.cos(q1 + q2)

    dx_dq1 = -l1 * s1 - l2 * s12
    dx_dq2 = -l2 * s12
    dy_dq1 = l1 * c1 + l2 * c12
    dy_dq2 = l2 * c12

    return np.array([[dx_dq1, dx_dq2], [dy_dq1, dy_dq2], [1.0, 1.0]], dtype=float)


def wrench_from_torque(
    q: npt.NDArray[np.float64], tau: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Approximate planar wrench [Fx, Fy, Mz] at the end-effector from joint torques.

    Uses the relation tau ~ J(q)^T w, inverted via a pseudoinverse:
        w ~ (J(q)^T)^+ tau

    Parameters
    ----------
    q : array_like, shape (2,)
        Joint angles [q1, q2].
    tau : array_like, shape (2,)
        Joint torques.

    Returns
    -------
    w : ndarray, shape (3,)
        Approximate planar wrench [Fx, Fy, Mz].
    """
    J = J_end_effector(q)
    return np.linalg.pinv(J.T) @ tau


def natural_wrench(
    q: npt.NDArray[np.float64],
    qdot: npt.NDArray[np.float64],
    qddot: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Natural wrench at the end-effector corresponding to the natural torque field.

        w_nat = (J(q)^T)^+ tau_nat(q, qdot, qddot)

    Parameters
    ----------
    q : array_like, shape (2,)
        Joint angles [q1, q2].
    qdot : array_like, shape (2,)
        Joint velocities [q1dot, q2dot].
    qddot : array_like, shape (2,)
        Joint accelerations [q1ddot, q2ddot].

    Returns
    -------
    w_nat : ndarray, shape (3,)
        Natural planar wrench [Fx, Fy, Mz].
    """
    tau_nat = tau_natural(q, qdot, qddot)
    return wrench_from_torque(q, tau_nat)


# ---------------------------------------------------------------------------
# Example usage / quick test
# ---------------------------------------------------------------------------


def run_example() -> None:
    """
    Run a simple simulation with PD input and print some diagnostics.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Initial condition: some arbitrary initial angles/velocities
    x0 = np.array([0.5, -0.5, 0.0, 0.0], dtype=float)
    t_span = (0.0, 5.0)
    t_eval = np.linspace(t_span[0], t_span[1], 501)

    # Solve the ODE
    sol = solve_ivp(
        fun=lambda t, x: double_pendulum_dynamics(t, x, lambda tt, xx: u_pd(tt, xx)),
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8,
    )

    # Compute natural torque trajectory
    t_samples, tau_nat_traj = compute_tau_natural_trajectory(
        sol, lambda t, x: u_pd(t, x)
    )

    # Simple console output
    logger.info("Simulation completed.")
    logger.info("Number of time steps: %d", len(t_samples))
    logger.info("First few natural torque samples:\n%s", tau_nat_traj[:5, :])


if __name__ == "__main__":
    run_example()
