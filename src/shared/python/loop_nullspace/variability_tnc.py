"""Tolerance-noise-covariation (TNC) decomposition (Muller and Sternad, NSX-97).

For execution variables ``x_i`` (swings) and a result map ``f`` with target
``y*``, the cost is ``C(D) = mean_i ||w * (f(x_i) - y*)||^2``.  Procedures:

* **Permutation.**  Each variable is permuted independently across trials
  (marginals kept, covariation destroyed).  ``C_perm`` is the mean over
  ``n_permutations`` permutations.
* **Shift.**  The permuted clouds are translated by the single ``delta`` that
  minimises their mean cost (``C_shift``); the dispersion is then collapsed
  to the shifted mean (``C_floor``, the unreachable part of the target).
* **Rotation.**  The centred cloud is rotated by Haar-random orthogonal maps
  about its mean (dispersion shape kept, orientation randomised) giving an
  independent estimate of the covariation effect.

Costs: ``T = C_perm - C_shift`` (location), ``N = C_shift - C_floor``
(dispersion), covariation ``Cov = C_perm - C_0`` (positive = covariation
lowers cost).  Exact identity: ``C_0 = T + N + C_floor - Cov``.  ``N`` can be
slightly negative for strongly non-linear ``f`` (Jensen); the identity holds
regardless.  The seed is declared (``DEFAULT_SEED``) and recorded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .variability_ucm import finite_matrix

Array = NDArray[np.float64]
OutcomeFn = Callable[[Array], Array]
DEFAULT_SEED = 20261016


@dataclass(frozen=True)
class TNCResult:
    """Costs, the Cov decomposition and the declared seed."""

    cost_measured: float
    cost_permuted: float
    cost_shifted: float
    cost_floor: float
    tolerance_cost: float
    noise_cost: float
    covariation_cost: float
    covariation_cost_rotation: float
    p_value_covariation: float
    identity_residual: float
    seed: int
    n_permutations: int
    n_rotations: int
    n_trials: int


def linear_outcome_fn(jacobian: ArrayLike, x_ref: ArrayLike | None = None) -> OutcomeFn:
    """First-order result map ``y = C (x - x_ref)`` (``x_ref = 0`` if omitted)."""
    jac = finite_matrix(jacobian, "jacobian")
    ref = np.zeros(jac.shape[1]) if x_ref is None else np.asarray(x_ref, dtype=float)

    def fn(x: Array) -> Array:
        return np.asarray((x - ref) @ jac.T)

    return fn


def permute_across_trials(states: Array, rng: np.random.Generator) -> Array:
    """Permute each column independently across trials (kills covariation)."""
    out = np.empty_like(states)
    for j in range(states.shape[1]):
        out[:, j] = states[rng.permutation(states.shape[0]), j]
    return out


def _haar(dim: int, rng: np.random.Generator) -> Array:
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    return np.asarray(q * np.sign(np.diag(r)))


def _cost(fn: OutcomeFn, x: Array, target: Array, w: Array) -> float:
    return float(np.mean(np.sum((w * (fn(x) - target)) ** 2, axis=1)))


def _optimal_shift(fn: OutcomeFn, clouds: Array, target: Array, w: Array) -> Array:
    def resid(delta: Array) -> Array:
        return np.asarray((w * (fn(clouds + delta) - target)).ravel())

    return np.asarray(least_squares(resid, np.zeros(clouds.shape[1])).x)


def tnc_decompose(
    states: ArrayLike,
    outcome_fn: OutcomeFn,
    target: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    n_permutations: int = 200,
    n_rotations: int = 200,
    seed: int = DEFAULT_SEED,
) -> TNCResult:
    """Müller-Sternad TNC decomposition of measured swing states.

    Postcondition: ``identity_residual`` (|C_0 - (T + N + C_floor - Cov)|,
    relative) is at round-off.

    Raises:
        ValueError: If fewer than two trials, counts below one, or ``target``
            and ``weights`` disagree with the outcome dimension.
    """
    x = finite_matrix(states, "states")
    if x.shape[0] < 2:
        raise ValueError("TNC needs at least two trials")
    if n_permutations < 1 or n_rotations < 1:
        raise ValueError("n_permutations and n_rotations must be >= 1")
    y_t = np.asarray(target, dtype=np.float64)
    m = int(np.asarray(outcome_fn(x[:1])).shape[1])
    w = np.ones(m) if weights is None else np.asarray(weights, dtype=np.float64)
    if y_t.shape != (m,) or w.shape != (m,) or np.any(w < 0):
        raise ValueError("target and weights must be non-negative vectors of length m")
    rng = np.random.default_rng(seed)

    c0 = _cost(outcome_fn, x, y_t, w)
    perms = np.stack([permute_across_trials(x, rng) for _ in range(n_permutations)])
    perm_costs = np.array([_cost(outcome_fn, p, y_t, w) for p in perms])
    c_perm = float(perm_costs.mean())
    flat = perms.reshape(-1, x.shape[1])
    delta = _optimal_shift(outcome_fn, flat, y_t, w)
    c_shift = _cost(outcome_fn, flat + delta, y_t, w)
    mean_shifted = (flat + delta).mean(axis=0, keepdims=True)
    c_floor = _cost(outcome_fn, mean_shifted, y_t, w)

    centred = x - x.mean(axis=0)
    rot_costs = [
        _cost(outcome_fn, x.mean(axis=0) + centred @ _haar(x.shape[1], rng), y_t, w)
        for _ in range(n_rotations)
    ]
    tol, noise, cov = c_perm - c_shift, c_shift - c_floor, c_perm - c0
    resid = abs(c0 - (tol + noise + c_floor - cov)) / max(abs(c0), 1e-300)
    p_val = (1.0 + float(np.sum(perm_costs <= c0))) / (1.0 + n_permutations)
    return TNCResult(
        cost_measured=c0,
        cost_permuted=c_perm,
        cost_shifted=c_shift,
        cost_floor=c_floor,
        tolerance_cost=tol,
        noise_cost=noise,
        covariation_cost=cov,
        covariation_cost_rotation=float(np.mean(rot_costs)) - c0,
        p_value_covariation=p_val,
        identity_residual=float(resid),
        seed=seed,
        n_permutations=n_permutations,
        n_rotations=n_rotations,
        n_trials=x.shape[0],
    )
