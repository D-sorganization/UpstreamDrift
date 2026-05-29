"""ZTCF / ZVCF counterfactual-acceleration primitives (epic task M7).

This module reproduces the **ZTCF** (zero-torque counterfactual) and **ZVCF**
(zero-velocity counterfactual) decompositions of the golf double-pendulum's
acceleration, expressed entirely in terms of the backend-agnostic
:class:`~simulation_backends.protocol.DynamicsProvider` primitives
(``mass_matrix`` and ``bias_forces``). Because the ODE reference backend and the
MuJoCo CPU backend both implement that Protocol, the *same* functions evaluate
on either engine and must agree (cross-validation, epic task M7).

Definitions (pointwise, at a single measured state ``(q, v)`` with control
``tau``)::

    ZTCF accel:  qddot = solve(M(q), -bias(q, v))         # the drift field f(x)
    ZVCF accel:  qddot = solve(M(q),  tau - bias(q, 0))   # velocity zeroed

For the planar double pendulum the bias force is ``C(q, v) v + g(q) + d(v)``.
At ``v = 0`` the Coriolis term (quadratic in velocity) and the viscous damping
term (linear in velocity) both vanish, so ``bias(q, 0) == g(q)``. The ZVCF
expression therefore reduces to ``solve(M, tau - g(q))``, matching the
analytical ground truth in
:meth:`PendulumPhysicsEngine.compute_zvcf`.

# AGENT-NOTE: These are POINTWISE / INSTANTANEOUS decompositions evaluated at
# each measured state along a trajectory -- they are NOT forward-integrated
# counterfactual rollouts. ``evaluate_ztcf_along_trajectory`` maps the ZTCF
# operator over the *measured* (q, v) samples and returns the instantaneous
# drift acceleration at each one; it does NOT integrate a zero-torque system
# forward in time. Do not "fix" this into a time integration -- the pointwise
# semantics are the whole point (epic task M7.3). The same caveat applies to
# every function in this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.core.contracts import check_finite, require
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from .protocol import DynamicsProvider

logger = get_logger(__name__)

__all__ = [
    "drift_and_control_split",
    "evaluate_ztcf_along_trajectory",
    "ztcf_acceleration",
    "zvcf_acceleration",
]


def _as_state_vector(name: str, value: np.ndarray) -> np.ndarray:
    """Coerce ``value`` to a finite 1-D float vector (shared precondition guard).

    Args:
        name: Argument name, used in diagnostic messages.
        value: Array-like to validate and coerce.

    Returns:
        A contiguous 1-D ``float`` :class:`numpy.ndarray`.

    Raises:
        ValueError: If ``value`` is not 1-D or contains non-finite entries.
    """
    arr = np.asarray(value, dtype=float).reshape(-1)
    require(arr.size > 0, f"{name} must be non-empty", value=arr.shape)
    require(check_finite(arr), f"{name} must contain only finite values", value=value)
    return arr


def _solve_mass(
    provider: DynamicsProvider, q: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Solve ``M(q) x = rhs`` using the provider's inertia matrix.

    Args:
        provider: Any object satisfying :class:`DynamicsProvider` (LOD: only its
            ``mass_matrix`` method is touched here).
        q: Joint positions ``(n,)`` at which to evaluate ``M``.
        rhs: Right-hand side ``(n,)``.

    Returns:
        The solution ``x = M(q)^-1 rhs``, shape ``(n,)``.

    Raises:
        ValueError: If ``M(q)`` is not an ``(n, n)`` matrix matching ``rhs`` or
            the resulting acceleration is non-finite (e.g. singular inertia).
    """
    mass = np.asarray(provider.mass_matrix(q), dtype=float)
    n = rhs.shape[0]
    require(
        mass.shape == (n, n),
        f"mass_matrix(q) must be ({n}, {n}); got {mass.shape}",
        value=mass.shape,
    )
    qddot = np.linalg.solve(mass, rhs)
    # Postcondition: a finite acceleration (a singular/ill-conditioned M would
    # surface here rather than silently propagating NaN downstream).
    require(
        check_finite(qddot),
        "solve(M(q), rhs) produced non-finite acceleration (singular inertia?)",
        value=qddot,
    )
    return qddot


def ztcf_acceleration(
    provider: DynamicsProvider, q: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Zero-torque counterfactual (ZTCF) acceleration at a single state.

    Computes the *instantaneous* drift-field acceleration ``f(x)`` -- the
    acceleration the mechanism would experience at this exact ``(q, v)`` with
    **zero applied torque**::

        qddot = solve(M(q), -bias(q, v))

    # AGENT-NOTE: Pointwise / instantaneous, evaluated at the supplied measured
    # state. This is NOT a forward-integrated zero-torque rollout.

    Args:
        provider: Backend exposing ``mass_matrix`` and ``bias_forces`` (the
            :class:`DynamicsProvider` Protocol; ODE or MuJoCo CPU backend).
        q: Joint positions ``(n,)`` [rad].
        v: Joint velocities ``(n,)`` [rad/s].

    Returns:
        Drift acceleration ``(n,)`` [rad/s^2].

    Postcondition:
        Result has shape ``(n,)`` and is finite.

    Raises:
        ValueError: If ``q``/``v`` are empty, non-finite, or differ in length.
    """
    q_arr = _as_state_vector("q", q)
    v_arr = _as_state_vector("v", v)
    require(
        q_arr.shape == v_arr.shape,
        f"q and v must share shape; got {q_arr.shape} vs {v_arr.shape}",
        value=(q_arr.shape, v_arr.shape),
    )
    bias = np.asarray(provider.bias_forces(q_arr, v_arr), dtype=float)
    require(
        bias.shape == q_arr.shape,
        f"bias_forces(q, v) must be {q_arr.shape}; got {bias.shape}",
        value=bias.shape,
    )
    return _solve_mass(provider, q_arr, -bias)


def zvcf_acceleration(
    provider: DynamicsProvider, q: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    """Zero-velocity counterfactual (ZVCF) acceleration at a single state.

    Computes the *instantaneous* acceleration with the **velocity zeroed but the
    applied control preserved**::

        qddot = solve(M(q), tau - bias(q, 0))

    With ``v = 0`` the Coriolis and viscous-damping contributions vanish, so
    ``bias(q, 0)`` reduces to the gravity vector ``g(q)`` -- this matches
    :meth:`PendulumPhysicsEngine.compute_zvcf` (which uses ``-g(q) + tau``)
    exactly.

    # AGENT-NOTE: Pointwise / instantaneous, evaluated at the supplied measured
    # position with velocity set to zero. This is NOT a forward-integrated
    # zero-velocity rollout.

    Args:
        provider: Backend exposing ``mass_matrix`` and ``bias_forces`` (the
            :class:`DynamicsProvider` Protocol; ODE or MuJoCo CPU backend).
        q: Joint positions ``(n,)`` [rad].
        tau: Applied generalised control/torque ``(n,)`` [N*m].

    Returns:
        Zero-velocity acceleration ``(n,)`` [rad/s^2].

    Postcondition:
        Result has shape ``(n,)`` and is finite.

    Raises:
        ValueError: If ``q``/``tau`` are empty, non-finite, or differ in length.
    """
    q_arr = _as_state_vector("q", q)
    tau_arr = _as_state_vector("tau", tau)
    require(
        q_arr.shape == tau_arr.shape,
        f"q and tau must share shape; got {q_arr.shape} vs {tau_arr.shape}",
        value=(q_arr.shape, tau_arr.shape),
    )
    bias_zero_v = np.asarray(
        provider.bias_forces(q_arr, np.zeros_like(q_arr)), dtype=float
    )
    require(
        bias_zero_v.shape == q_arr.shape,
        f"bias_forces(q, 0) must be {q_arr.shape}; got {bias_zero_v.shape}",
        value=bias_zero_v.shape,
    )
    return _solve_mass(provider, q_arr, tau_arr - bias_zero_v)


def evaluate_ztcf_along_trajectory(
    provider: DynamicsProvider, q_traj: np.ndarray, v_traj: np.ndarray
) -> np.ndarray:
    """Evaluate the ZTCF acceleration pointwise at every sampled state.

    Maps :func:`ztcf_acceleration` over each ``(q_traj[t], v_traj[t])`` sample
    and stacks the results. The output row ``t`` is the *instantaneous* drift
    acceleration at measured sample ``t``.

    # AGENT-NOTE: This is a POINTWISE evaluation along an ALREADY-MEASURED
    # trajectory -- each row is the instantaneous zero-torque drift acceleration
    # at that measured state. It is NOT a forward integration of a zero-torque
    # system, and must not be "fixed" into one (epic task M7.3). The measured
    # trajectory is the input; the drift field sampled on it is the output.

    Args:
        provider: Backend exposing ``mass_matrix`` and ``bias_forces``.
        q_traj: Position history, shape ``(T, n)`` [rad].
        v_traj: Velocity history, shape ``(T, n)`` [rad/s].

    Returns:
        ZTCF accelerations, shape ``(T, n)`` [rad/s^2].

    Postcondition:
        Output shape equals ``q_traj``'s shape and every entry is finite.

    Raises:
        ValueError: If ``q_traj``/``v_traj`` are not 2-D, disagree in shape, or
            contain non-finite entries.
    """
    q_mat = np.asarray(q_traj, dtype=float)
    v_mat = np.asarray(v_traj, dtype=float)
    require(q_mat.ndim == 2, "q_traj must be 2-D (T, n)", value=q_mat.shape)
    require(v_mat.ndim == 2, "v_traj must be 2-D (T, n)", value=v_mat.shape)
    require(
        q_mat.shape == v_mat.shape,
        f"q_traj and v_traj must share shape; got {q_mat.shape} vs {v_mat.shape}",
        value=(q_mat.shape, v_mat.shape),
    )
    require(
        check_finite(q_mat) and check_finite(v_mat),
        "q_traj and v_traj must contain only finite values",
    )

    out = np.empty_like(q_mat)
    for idx in range(q_mat.shape[0]):
        out[idx] = ztcf_acceleration(provider, q_mat[idx], v_mat[idx])
    return out


def drift_and_control_split(
    provider: DynamicsProvider,
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the acceleration into its drift and control contributions.

    Because the equation of motion is affine in the control, the total
    acceleration ``solve(M, tau - bias(q, v))`` decomposes additively into:

    * the **drift** term ``f = solve(M(q), -bias(q, v))`` -- identical to
      :func:`ztcf_acceleration`; and
    * the **control** term ``solve(M(q), tau)``.

    Their sum is the actual acceleration under ``tau``, so this is the
    instantaneous ``qddot = f(x) + g(x) u`` decomposition at the measured state.

    # AGENT-NOTE: Pointwise / instantaneous decomposition at a single measured
    # state -- NOT a forward-integrated counterfactual. Do not turn this into a
    # time integration (epic task M7.3).

    Args:
        provider: Backend exposing ``mass_matrix`` and ``bias_forces``.
        q: Joint positions ``(n,)`` [rad].
        v: Joint velocities ``(n,)`` [rad/s].
        tau: Applied generalised control/torque ``(n,)`` [N*m].

    Returns:
        A ``(drift, control)`` tuple, each shape ``(n,)`` [rad/s^2], where
        ``drift`` is the ZTCF acceleration and ``control`` is ``M(q)^-1 tau``.

    Postcondition:
        ``drift + control`` equals ``solve(M(q), tau - bias(q, v))``.

    Raises:
        ValueError: If ``q``/``v``/``tau`` are empty, non-finite, or differ in
            length.
    """
    drift = ztcf_acceleration(provider, q, v)
    q_arr = _as_state_vector("q", q)
    tau_arr = _as_state_vector("tau", tau)
    require(
        q_arr.shape == tau_arr.shape,
        f"q and tau must share shape; got {q_arr.shape} vs {tau_arr.shape}",
        value=(q_arr.shape, tau_arr.shape),
    )
    control = _solve_mass(provider, q_arr, tau_arr)
    return drift, control
