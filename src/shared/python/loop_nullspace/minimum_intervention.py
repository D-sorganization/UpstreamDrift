"""Minimum-intervention (task-relevant) TVLQR feedback gains (NSX-93, #12722).

*Minimum intervention* means the controller corrects a deviation only to the
extent that it changes the task outcome.  Here the task is the weighted outcome
at the physical impact event, ``y = C_imp x_N`` with weights ``W_y`` (face angle
only, or face plus other channels).  The gain is the MOSAIC
:func:`~estimation.mosaic.local_policy.tvlqr_gains` with

* running state cost ``Q = 0`` (no cost for being off the reference *per se*),
* input cost ``R`` (effort, positive definite), and
* terminal weight ``C_imp^T W_y C_imp``.

Why this corrects only task-relevant deviations.  With ``Q = 0`` the optimal
cost-to-go from ``(t, x)`` depends on ``x`` only through the weighted outcome the
state would produce under zero control, ``W^{1/2} C Phi(N, t) x`` where ``Phi``
is the open-loop propagator (a state change ``d`` with ``C Phi d = 0`` can be
absorbed by applying the same controls, so the cost is unchanged).  Hence its
Hessian ``V_t`` annihilates ``ker(W^{1/2} C Phi(N, t))`` -- the kernel of the
NSX-90 state kernel ``S_x[t]`` for the rows ``W^{1/2} C`` -- and so does the gain
``K_t = (R + B^T V_{t+1} B)^{-1} B^T V_{t+1} A_t`` because ``A_t d`` stays in
that kernel.  :func:`task_irrelevant_gain_norm` evaluates ``max_t ||K_t N_t||``
for an orthonormal basis ``N_t`` of that kernel; the acceptance test requires it
below ``1e-10`` and observes machine-precision values, because the statement is
an identity, not an approximation.

The gain is designed on the *delay-free* plant and applied to a delayed state
(:mod:`feedback_layer`); the loss of margin that costs is reported by
:func:`layer_linear.frozen_delay_margin`, not designed away.  A predictor-based
delay-compensated design is a different controller and is not implemented here.

Gain convention.  ``u = -K dx`` for the tangent-state deviation ``dx = (xi,
eta)``; :func:`joint_space_gains` rewrites it in full joint-coordinate errors
``e = ref - state`` as ``u = Kq e_q + Kv e_v`` for the replay.  Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.estimation.mosaic.local_policy import (
    LinearizedDynamics,
    tvlqr_gains,
)
from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.variance_propagation import adjoint_kernels

__all__ = [
    "MinimumInterventionDesign",
    "face_only_weights",
    "joint_space_gains",
    "minimum_intervention_design",
    "task_irrelevant_gain_norm",
]

_RANK_RTOL = 1e-12
_PSD_TOL = 1e-12


def face_only_weights(n_outcomes: int, face_index: int, weight: float = 1.0) -> Array:
    """``W_y`` with a single positive entry: only the face outcome is task-relevant."""
    if n_outcomes < 1:
        raise ValueError(f"n_outcomes must be >= 1, got {n_outcomes}")
    if not 0 <= face_index < n_outcomes:
        raise ValueError(f"face_index must be in [0, {n_outcomes}), got {face_index}")
    if not (np.isfinite(weight) and weight > 0.0):
        raise ValueError(f"weight must be positive and finite, got {weight}")
    out = np.zeros((n_outcomes, n_outcomes))
    out[face_index, face_index] = weight
    return out


def _psd(value: ArrayLike, size: int, name: str) -> Array:
    w = np.asarray(value, dtype=np.float64)
    if w.shape != (size, size) or not np.all(np.isfinite(w)):
        raise ValueError(f"{name} must be finite with shape ({size}, {size})")
    if np.max(np.abs(w - w.T)) > 1e-9 * max(1.0, float(np.max(np.abs(w)))):
        raise ValueError(f"{name} must be symmetric")
    if np.linalg.eigvalsh(0.5 * (w + w.T))[0] < -_PSD_TOL * max(1.0, np.abs(w).max()):
        raise ValueError(f"{name} must be positive semi-definite")
    return 0.5 * (w + w.T)


def _weighted_rows(outcome_rows: Array, weights: Array) -> Array:
    """``W^{1/2} C`` restricted to the positive-weight eigen-directions of ``W``."""
    vals, vecs = np.linalg.eigh(weights)
    keep = vals > _PSD_TOL * max(1.0, float(vals.max()))
    return np.asarray(np.sqrt(vals[keep])[:, None] * (vecs[:, keep].T @ outcome_rows))


@dataclass(frozen=True)
class MinimumInterventionDesign:
    """Gains ``K`` ``(T, n_u, n_x)`` with ``u = -K dx``, and what they were built from."""

    gains: Array
    terminal_weight: Array
    outcome_weights: Array
    input_weight: Array


def _check_design_inputs(
    plant: LinearizedDynamics,
    outcome_rows: ArrayLike,
    outcome_weights: ArrayLike,
    input_weight: ArrayLike,
) -> tuple[Array, Array, Array]:
    n_x, n_u = plant.b.shape[1], plant.b.shape[2]
    rows = np.asarray(outcome_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != n_x or not np.all(np.isfinite(rows)):
        raise ValueError(f"outcome_rows must be finite with {n_x} columns")
    weights = _psd(outcome_weights, rows.shape[0], "outcome_weights")
    r_mat = np.asarray(input_weight, dtype=np.float64)
    if r_mat.shape != (n_u, n_u) or not np.all(np.isfinite(r_mat)):
        raise ValueError(f"input_weight must be finite with shape ({n_u}, {n_u})")
    r_mat = 0.5 * (r_mat + r_mat.T)
    if np.linalg.eigvalsh(r_mat)[0] <= 0.0:
        raise ValueError("input_weight must be positive definite")
    return rows, weights, r_mat


def minimum_intervention_design(
    plant: LinearizedDynamics,
    outcome_rows: ArrayLike,
    outcome_weights: ArrayLike,
    input_weight: ArrayLike,
) -> MinimumInterventionDesign:
    """TVLQR with ``Q = 0`` and terminal weight ``C^T W_y C`` (see module docstring).

    Args:
        plant: ``(A_t, B_t)`` with ``B`` the *control* input (``n_u`` torques).
        outcome_rows: ``C_imp`` ``(n_y, n_x)`` at the impact event.
        outcome_weights: ``W_y`` ``(n_y, n_y)`` symmetric PSD (zero rows = not
            task-relevant, e.g. :func:`face_only_weights`).
        input_weight: ``R`` ``(n_u, n_u)`` positive definite.

    Raises:
        ValueError: Shapes, non-finite entries, ``W_y`` not PSD or ``R`` not PD.

    Postcondition: the gains are finite and annihilate ``ker(W^{1/2} C Phi)``.
    """
    rows, weights, r_mat = _check_design_inputs(
        plant, outcome_rows, outcome_weights, input_weight
    )
    terminal = rows.T @ weights @ rows
    gains = tvlqr_gains(plant, np.zeros((rows.shape[1],) * 2), r_mat, terminal)
    return MinimumInterventionDesign(gains, terminal, weights, r_mat)


def task_irrelevant_gain_norm(
    gains: ArrayLike,
    plant: LinearizedDynamics,
    outcome_rows: ArrayLike,
    outcome_weights: ArrayLike,
) -> float:
    """``max_t ||K_t N_t||_2`` over the task-irrelevant directions ``N_t``.

    ``N_t`` is an orthonormal basis of ``ker S_x[t]`` for the weighted rows
    (NSX-90 :func:`adjoint_kernels`); the rank cut is ``1e-12 sigma_max``.
    ``0.0`` if every direction is task-relevant.
    """
    k_all = np.asarray(gains, dtype=np.float64)
    n_x = plant.a.shape[1]
    if k_all.shape != (plant.n_steps, plant.b.shape[2], n_x):
        raise ValueError(
            f"gains must have shape {(plant.n_steps, plant.b.shape[2], n_x)}"
        )
    rows = np.asarray(outcome_rows, dtype=np.float64)
    weights = _psd(outcome_weights, rows.shape[0], "outcome_weights")
    kernels = adjoint_kernels(plant, _weighted_rows(rows, weights))
    worst = 0.0
    for t in range(plant.n_steps):
        _, sing, vt = np.linalg.svd(kernels.state[t])
        rank = int((sing > _RANK_RTOL * max(float(sing[0]), 1.0)).sum())
        null = vt[rank:].T
        if null.shape[1]:
            worst = max(worst, float(np.linalg.norm(k_all[t] @ null, 2)))
    return worst


def joint_space_gains(
    k_tangent: ArrayLike, basis: ArrayLike, offset: ArrayLike
) -> tuple[Array, Array]:
    """Tangent-state gains as full joint-coordinate error gains ``(Kq, Kv)``.

    With ``dq = P xi`` and ``dv = P eta + O xi`` (NSX-84) the tangent deviation is
    ``xi = P^T dq``, ``eta = P^T (dv - O xi)``, so ``-K dx = -(K_xi - K_eta P^T O)
    P^T dq - K_eta P^T dv``.  The replay feeds ``e = ref - state = -dev``, hence
    ``u = Kq e_q + Kv e_v`` with ``Kq = (K_xi - K_eta P^T O) P^T`` and ``Kv =
    K_eta P^T``.  Shapes: ``K (n_u, 2k)``, ``P, O (n_v, k)``; a leading step axis
    ``(T, ...)`` on all three is allowed.

    Raises:
        ValueError: Inconsistent shapes or non-finite input.
    """
    k_arr = np.asarray(k_tangent, dtype=np.float64)
    p_arr = np.asarray(basis, dtype=np.float64)
    o_arr = np.asarray(offset, dtype=np.float64)
    single = k_arr.ndim == 2
    if single:
        k_arr, p_arr, o_arr = k_arr[None], p_arr[None], o_arr[None]
    if k_arr.ndim != 3 or p_arr.ndim != 3 or p_arr.shape != o_arr.shape:
        raise ValueError("gains, basis and offset must be (n_u,2k)/(n_v,k)/(n_v,k)")
    k = p_arr.shape[2]
    if k_arr.shape[2] != 2 * k or not np.all(np.isfinite(k_arr)):
        raise ValueError(f"tangent gains need {2 * k} columns (xi, eta)")
    if not (p_arr.shape[0] == o_arr.shape[0] == k_arr.shape[0]):
        raise ValueError("gains, basis and offset must cover the same steps")
    k_xi, k_eta = k_arr[:, :, :k], k_arr[:, :, k:]
    p_t = np.swapaxes(p_arr, 1, 2)
    kq = (k_xi - k_eta @ (p_t @ o_arr)) @ p_t
    kv = k_eta @ p_t
    return (kq[0], kv[0]) if single else (kq, kv)
