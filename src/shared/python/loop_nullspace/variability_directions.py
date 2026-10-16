"""Predicted versus measured outcome-covariance directions (NSX-97).

Predicted covariance ``C Sigma_x C^T`` uses noise fitted on training
sessions; measured covariance comes from held-out swings.  The comparison is
the principal angles between the top-``k`` eigen-subspaces.  The null keeps
every outcome's marginal distribution but permutes each outcome column
independently across swings (covariation destroyed); ``p`` is the share of
null mean angles at or below the observed one.  Outcomes of mixed units must
be rescaled via ``outcome_scale``.  Angles are not wrapped; supply unwrapped
angle outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .variability_tnc import DEFAULT_SEED, permute_across_trials
from .variability_ucm import finite_matrix

Array = NDArray[np.float64]


@dataclass(frozen=True)
class DirectionComparison:
    """Principal angles (deg), their mean, the permutation p-value and seed."""

    k: int
    angles_deg: tuple[float, ...]
    mean_angle_deg: float
    null_mean_angle_deg: float
    p_value: float
    n_permutations: int
    seed: int


def predicted_outcome_covariance(jacobian: ArrayLike, state_cov: ArrayLike) -> Array:
    """``C Sigma C^T`` for the outcome Jacobian ``C`` (m x n)."""
    jac = finite_matrix(jacobian, "jacobian")
    cov = finite_matrix(state_cov, "state_cov")
    if cov.shape != (jac.shape[1],) * 2:
        raise ValueError("state_cov must be n x n matching the jacobian columns")
    return np.asarray(jac @ cov @ jac.T)


def principal_angles_deg(u1: ArrayLike, u2: ArrayLike) -> Array:
    """Principal angles in degrees between the column spans of two bases."""
    q1 = np.linalg.qr(np.asarray(u1, dtype=float))[0]
    q2 = np.linalg.qr(np.asarray(u2, dtype=float))[0]
    sv = np.clip(np.linalg.svd(q1.T @ q2, compute_uv=False), -1.0, 1.0)
    return np.asarray(np.sort(np.degrees(np.arccos(sv))))


def _cov(samples: Array) -> Array:
    """Sample covariance with variables in columns, as a float64 matrix."""
    return np.asarray(np.cov(samples, rowvar=False), dtype=np.float64)


def _top_basis(cov: Array, k: int) -> Array:
    _, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    return np.asarray(vecs[:, ::-1][:, :k])


def compare_outcome_directions(
    predicted_cov: ArrayLike,
    measured_outcomes: ArrayLike,
    *,
    k: int = 2,
    outcome_scale: ArrayLike | None = None,
    n_permutations: int = 500,
    seed: int = DEFAULT_SEED,
) -> DirectionComparison:
    """Compare top-``k`` predicted and measured outcome-covariance directions.

    Raises:
        ValueError: On shape mismatch, ``k`` outside ``[1, m - 1]``, fewer
            than three swings or non-positive scale.
    """
    y = finite_matrix(measured_outcomes, "measured_outcomes")
    pred = finite_matrix(predicted_cov, "predicted_cov")
    m = y.shape[1]
    if pred.shape != (m, m):
        raise ValueError("predicted_cov must be m x m for m measured outcomes")
    if not 1 <= k < m or y.shape[0] < 3 or n_permutations < 1:
        raise ValueError("need 1 <= k < m, >= 3 swings and n_permutations >= 1")
    if outcome_scale is not None:
        s = np.asarray(outcome_scale, dtype=float)
        if s.shape != (m,) or not np.all(s > 0):
            raise ValueError("outcome_scale must be a positive vector of length m")
        y, pred = y / s, pred / np.outer(s, s)
    u_pred = _top_basis(pred, k)
    obs = float(principal_angles_deg(u_pred, _top_basis(_cov(y), k)).mean())
    rng = np.random.default_rng(seed)
    null = np.array(
        [
            principal_angles_deg(
                u_pred,
                _top_basis(_cov(permute_across_trials(y, rng)), k),
            ).mean()
            for _ in range(n_permutations)
        ]
    )
    angles = principal_angles_deg(u_pred, _top_basis(_cov(y), k))
    return DirectionComparison(
        k=k,
        angles_deg=tuple(float(a) for a in angles),
        mean_angle_deg=obs,
        null_mean_angle_deg=float(null.mean()),
        p_value=(1.0 + float(np.sum(null <= obs))) / (1.0 + n_permutations),
        n_permutations=n_permutations,
        seed=seed,
    )
