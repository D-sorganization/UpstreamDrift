"""Uncontrolled-manifold decomposition of impact-state variability (NSX-97).

Given measured impact states ``x_i`` and the impact-outcome Jacobian ``C``
(``ImpactOutcomeMap.jacobian``), the state covariance is split into

* ``V_UCM``: variance in ``ker C`` (outcome-neutral to first order), and
* ``V_ORT``: variance in the orthogonal complement ``row(C)``.

Both are normalised **per dimension** (``V/dim``).  The synergy index is
``dV = (V_UCM/d_ucm - V_ORT/d_ort) / (V_TOT/n)`` with
``n = d_ucm + d_ort``; it lies in ``[-n/d_ort, n/d_ucm]``.  The Fisher-type
transform is ``dVz = 0.5 ln((n/d_ort + dV) / (n/d_ucm - dV))`` (Latash et al.
2010; Verrel 2010 bounds), clamped at ``Z_CLIP`` at the bounds.  With unequal dimensions the transform is
not zero at ``dV = 0``: an isotropic cloud gives ``0.5 ln(d_ucm/d_ort)``.

States must be expressed in comparable units; ``state_scale`` rescales each
dimension (``z = x / scale``) before the kernel is taken.  The result
characterises variability *structure* only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
Z_CLIP = 1e-12
RANK_RTOL = 1e-10


@dataclass(frozen=True)
class UCMResult:
    """UCM/ORT variances (totals and per dimension), ``dV`` and ``dVz``."""

    n_trials: int
    n_state: int
    n_ucm: int
    n_ort: int
    v_ucm: float
    v_ort: float
    v_ucm_per_dim: float
    v_ort_per_dim: float
    v_total_per_dim: float
    delta_v: float
    delta_v_z: float


def finite_matrix(value: ArrayLike, name: str) -> Array:
    """Return a finite 2-D float copy or raise ``ValueError``."""
    arr = np.array(value, dtype=np.float64, copy=True)
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 2-D array")
    return arr


def ucm_decompose(
    states: ArrayLike, jacobian: ArrayLike, state_scale: ArrayLike | None = None
) -> UCMResult:
    """Decompose state covariance about ``ker jacobian``.

    Postconditions: ``v_ucm + v_ort`` equals the trace of the sample
    covariance; ``d_ucm + d_ort = n`` with ``d_ort = rank(jacobian)``.

    Raises:
        ValueError: If shapes disagree, fewer than two trials, non-finite
            data, ``state_scale`` non-positive, or the kernel or its
            complement is empty.
    """
    x = finite_matrix(states, "states")
    jac = finite_matrix(jacobian, "jacobian")
    n_trials, n = x.shape
    if n_trials < 2:
        raise ValueError("UCM needs at least two trials")
    if jac.shape[1] != n:
        raise ValueError(f"jacobian has {jac.shape[1]} columns but states have {n}")
    if state_scale is not None:
        scale = np.asarray(state_scale, dtype=np.float64)
        if scale.shape != (n,) or not np.all(scale > 0.0):
            raise ValueError("state_scale must be a positive vector of length n")
        x = x / scale
        jac = jac * scale
    _, sv, vt = np.linalg.svd(jac)
    rank = int(np.sum(sv > RANK_RTOL * max(1.0, sv[0] if sv.size else 0.0)))
    if rank == 0 or rank == n:
        raise ValueError(
            f"UCM needs 0 < rank(jacobian) < n; got rank {rank} for n = {n}"
        )
    d_ort, d_ucm = rank, n - rank
    xc = x - x.mean(axis=0)
    v_ort = float(np.sum((xc @ vt[:rank].T) ** 2) / (n_trials - 1))
    v_ucm = float(np.sum((xc @ vt[rank:].T) ** 2) / (n_trials - 1))
    v_tot_pd = (v_ucm + v_ort) / n
    if v_tot_pd <= 0.0:
        raise ValueError("states have zero total variance")
    delta_v = (v_ucm / d_ucm - v_ort / d_ort) / v_tot_pd
    hi, lo = n / d_ucm, n / d_ort
    num = max(lo + delta_v, Z_CLIP)
    den = max(hi - delta_v, Z_CLIP)
    return UCMResult(
        n_trials=n_trials,
        n_state=n,
        n_ucm=d_ucm,
        n_ort=d_ort,
        v_ucm=v_ucm,
        v_ort=v_ort,
        v_ucm_per_dim=v_ucm / d_ucm,
        v_ort_per_dim=v_ort / d_ort,
        v_total_per_dim=v_tot_pd,
        delta_v=float(delta_v),
        delta_v_z=float(0.5 * np.log(num / den)),
    )
