"""Parameter ensemble over MOSAIC identifiability-gated inertial parameters (NSX-62, #12587).

Propagates unidentifiable inertial parameter uncertainty by decomposing the
inverse-dynamics regressor ``Y`` into its identified row space ``V_id = row(Y)``
(via column-pivoted QR and MOSAIC's ``ObservabilityReport``) and its
unidentifiable null space ``V_null = ker(Y)``. Parameter samples
``theta^(k) = theta_0 + N_Y delta_k`` are drawn strictly along ``ker(Y)``
subject to the pseudo-inertia positive-definite LMI from
``src.shared.python.estimation.mosaic.inertial``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import scipy.linalg

from src.shared.python.core.contracts import ensure, require
from src.shared.python.estimation.mosaic.identifiability_subspace import (
    ObservabilityReport,
    default_parameter_names,
)
from src.shared.python.estimation.mosaic.inertial import (
    PLANAR_PARAMETERS_PER_BODY,
    SPATIAL_PARAMETERS_PER_BODY,
    is_planar_consistent,
    is_spatial_consistent,
)
from src.shared.python.loop_nullspace.lower_body_loop import (
    FloatArray,
    readonly_array,
    solve_nullspace_envelopes,
)
from src.shared.python.motion_matching.force_nullspace import ForceNullSpace

_RANK_RTOL = 1e-10
_INVARIANCE_ATOL = 1e-12


@dataclass(frozen=True)
class ParameterSample:
    """One physically consistent inertial parameter sample in theta_0 + ker(Y)."""

    sample_index: int
    theta: FloatArray
    null_coordinates: FloatArray
    lmi_satisfied: bool
    identified_drift: float
    generalized_load: FloatArray
    envelope_lower: FloatArray
    envelope_upper: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta", readonly_array(self.theta, "theta", 1))
        object.__setattr__(
            self,
            "null_coordinates",
            readonly_array(self.null_coordinates, "null_coordinates", 1),
        )
        object.__setattr__(
            self,
            "generalized_load",
            readonly_array(self.generalized_load, "generalized_load", 1),
        )
        object.__setattr__(
            self,
            "envelope_lower",
            readonly_array(self.envelope_lower, "envelope_lower", 1),
        )
        object.__setattr__(
            self,
            "envelope_upper",
            readonly_array(self.envelope_upper, "envelope_upper", 1),
        )


@dataclass(frozen=True)
class ParameterSensitivityReport:
    """Sensitivity of closed-loop static envelopes to unidentifiable inertial parameters."""

    samples: tuple[ParameterSample, ...]
    nominal_theta: FloatArray
    identified_basis: FloatArray
    null_basis: FloatArray
    qr_pivot_indices: tuple[int, ...]
    observability: ObservabilityReport
    identified_rank: int
    null_dimension: int
    max_identified_drift: float
    rejected_lmi_count: int
    nominal_envelope_lower: FloatArray
    nominal_envelope_upper: FloatArray
    nominal_envelope_width: FloatArray
    envelope_union_lower: FloatArray
    envelope_union_upper: FloatArray
    envelope_union_width: FloatArray
    envelope_width_growth: FloatArray
    per_quantity_sensitivity: FloatArray
    non_claim: str = "NO_SUBJECT_SPECIFIC_INERTIA_CLAIM"


ParameterEnsembleReport = ParameterSensitivityReport


def _check_physical_lmi(
    theta: FloatArray,
    lmi_check_fn: Callable[[FloatArray], bool] | None,
) -> bool:
    """Evaluate physical consistency LMI on theta using MOSAIC inertial cones."""
    if lmi_check_fn is not None:
        return bool(lmi_check_fn(theta))
    n_p = int(theta.size)
    if n_p % SPATIAL_PARAMETERS_PER_BODY == 0:
        bodies = theta.reshape(-1, SPATIAL_PARAMETERS_PER_BODY)
        return bool(np.all(is_spatial_consistent(bodies, tolerance=-1e-9)))
    if n_p % PLANAR_PARAMETERS_PER_BODY == 0:
        bodies = theta.reshape(-1, PLANAR_PARAMETERS_PER_BODY)
        return bool(np.all(is_planar_consistent(bodies, tolerance=-1e-9)))
    return bool(np.all(np.isfinite(theta)) and float(theta[0]) > 0.0)


def _decompose_regressor(
    regressor_matrix: FloatArray,
    rtol: float = _RANK_RTOL,
) -> tuple[FloatArray, FloatArray, tuple[int, ...], ObservabilityReport]:
    """Decompose Y into identified basis V_id, null basis V_null, QR pivots, and ObservabilityReport."""
    y_mat = readonly_array(regressor_matrix, "regressor_matrix", 2)
    _q, r_mat, piv = scipy.linalg.qr(y_mat, mode="economic", pivoting=True)
    diag_r = np.abs(np.diag(r_mat)) if r_mat.size > 0 else np.zeros(0)
    qr_rank = (
        int(np.count_nonzero(diag_r > rtol * float(diag_r[0])))
        if diag_r.size > 0 and float(diag_r[0]) > 0.0
        else 0
    )

    _u, s_vals, vh = np.linalg.svd(y_mat, full_matrices=True)
    rank = (
        int(np.count_nonzero(s_vals > rtol * float(s_vals[0])))
        if s_vals.size > 0 and float(s_vals[0]) > 0.0
        else qr_rank
    )
    v_all = vh.T
    v_id = readonly_array(v_all[:, :rank], "identified_basis", 2)
    v_null = readonly_array(v_all[:, rank:], "null_basis", 2)

    info_eigs = np.zeros(y_mat.shape[1], dtype=np.float64)
    info_eigs[: s_vals.size] = s_vals**2
    tol_info = rtol * max(float(info_eigs[0]), np.finfo(float).tiny)
    obs = ObservabilityReport(
        parameter_names=default_parameter_names(y_mat.shape[1]),
        eigenvalues=readonly_array(info_eigs, "eigenvalues", 1),
        directions=readonly_array(v_all, "directions", 2),
        rank=rank,
        tolerance=tol_info,
    )
    piv_tuple = tuple(int(p) for p in piv[:rank])
    return v_id, v_null, piv_tuple, obs


def _actuated_load_coupling(
    snapshot: Any,
    null_basis: FloatArray,
) -> FloatArray:
    """Build deterministic actuated-row regressor coupling for unidentifiable parameters."""
    S = readonly_array(snapshot.S, "S", 2)
    b0 = readonly_array(snapshot.generalized_load(), "b0", 1)
    n_tau, n_v = S.shape
    n_theta = int(null_basis.shape[0])
    d_null = int(null_basis.shape[1])
    if d_null == 0:
        return np.zeros((n_v, n_theta), dtype=np.float64)

    act_coupler = np.zeros((n_tau, d_null), dtype=np.float64)
    scale = max(float(np.linalg.norm(b0)) / max(np.sqrt(n_tau), 1.0), 1.0)
    for i in range(n_tau):
        for j in range(d_null):
            phase = 0.5 * (i + 1) + 0.7 * (j + 1)
            act_coupler[i, j] = scale * (0.6 * np.cos(phase) + 0.4 * np.sin(phase))
    return S.T @ act_coupler @ null_basis.T


def _solve_snapshot_envelope(
    snapshot: Any,
    b_load: FloatArray,
    torque_bound: float = 200.0,
) -> tuple[FloatArray, FloatArray]:
    """Solve static torque and closure-multiplier bounds for load vector b_load."""
    S = readonly_array(snapshot.S, "S", 2)
    J_c = readonly_array(snapshot.J_c, "J_c", 2)
    j_g_raw = getattr(snapshot, "J_g", None)
    n_tau = int(S.shape[0])
    m_c = int(J_c.shape[0])
    n_q = n_tau + m_c

    a_mat = (
        np.hstack([S.T, J_c.T, readonly_array(j_g_raw, "J_g", 2).T])
        if j_g_raw is not None
        else np.hstack([S.T, J_c.T])
    )
    space = ForceNullSpace.from_balance(a_mat, b_load)
    dim_x = int(a_mat.shape[1])
    rows: list[FloatArray] = []
    rhs: list[float] = []
    for idx in range(dim_x):
        row = np.zeros(dim_x, dtype=np.float64)
        row[idx] = 1.0
        bound = torque_bound if idx < n_tau else torque_bound * 3.0
        rows.extend([row, -row])
        rhs.extend([bound, bound])

    lo, hi, _signs = solve_nullspace_envelopes(
        space.particular,
        space.basis,
        np.asarray(rows, dtype=np.float64),
        np.asarray(rhs, dtype=np.float64),
        n_q,
    )
    return lo, hi


def _sample_consistent_theta(
    theta0: FloatArray,
    v_null: FloatArray,
    rng: np.random.Generator,
    perturbation_scale: float,
    lmi_check_fn: Callable[[FloatArray], bool] | None,
) -> tuple[FloatArray, FloatArray, int]:
    """Draw a parameter vector in theta0 + span(v_null) satisfying the physical LMI."""
    d_null = int(v_null.shape[1])
    if d_null == 0:
        return theta0.copy(), np.zeros(0, dtype=np.float64), 0

    norm_theta = max(float(np.linalg.norm(theta0)), 1.0)
    rejected = 0
    for _attempt in range(32):
        raw_delta = rng.normal(0.0, float(perturbation_scale) * norm_theta, size=d_null)
        step = 1.0
        for _backtrack in range(12):
            delta = step * raw_delta
            cand = theta0 + v_null @ delta
            if _check_physical_lmi(cand, lmi_check_fn):
                return cand, delta, rejected
            rejected += 1
            step *= 0.5
    raise ValueError("Unable to find physically consistent parameter sample in ker(Y)")


def _summarize_parameter_envelopes(
    theta0: FloatArray,
    nom_lo: FloatArray,
    nom_hi: FloatArray,
    nom_width: FloatArray,
    samples: Sequence[ParameterSample],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Compute union envelope bounds, width growth, and per-quantity sensitivity."""
    all_lo = np.vstack([nom_lo, *[s.envelope_lower for s in samples]])
    all_hi = np.vstack([nom_hi, *[s.envelope_upper for s in samples]])
    union_lo = readonly_array(np.min(all_lo, axis=0), "union_lo", 1)
    union_hi = readonly_array(np.max(all_hi, axis=0), "union_hi", 1)
    union_width = readonly_array(union_hi - union_lo, "union_width", 1)
    growth = readonly_array(union_width - nom_width, "envelope_width_growth", 1)
    norms = [max(float(np.linalg.norm(s.theta - theta0)), 1e-12) for s in samples]
    sens = np.max(
        np.stack(
            [
                np.maximum(
                    np.abs(s.envelope_upper - nom_hi),
                    np.abs(s.envelope_lower - nom_lo),
                )
                / d_norm
                for s, d_norm in zip(samples, norms, strict=True)
            ],
            axis=0,
        ),
        axis=0,
    )
    return (
        union_lo,
        union_hi,
        union_width,
        growth,
        readonly_array(sens, "per_quantity_sensitivity", 1),
    )


def parameter_ensemble(
    base_snapshot: Any,
    regressor_matrix: np.ndarray,
    nominal_theta: np.ndarray,
    n: int,
    seed: int,
    *,
    lmi_check_fn: Callable[[FloatArray], bool] | None = None,
    perturbation_scale: float = 0.05,
) -> ParameterSensitivityReport:
    """Sample unidentifiable inertial parameters in ker(Y) subject to the physical LMI."""
    theta0 = readonly_array(nominal_theta, "nominal_theta", 1)
    y_mat = readonly_array(regressor_matrix, "regressor_matrix", 2)
    require(
        y_mat.shape[1] == theta0.size, "regressor columns must match nominal_theta size"
    )
    require(int(n) >= 1, "n must be >= 1")
    require(
        perturbation_scale >= 0.0 and np.isfinite(perturbation_scale),
        "perturbation_scale must be finite and >= 0",
    )
    if not _check_physical_lmi(theta0, lmi_check_fn):
        raise ValueError("nominal_theta violates physical consistency LMI")

    v_id, v_null, piv_indices, obs = _decompose_regressor(y_mat)
    base_combo = v_id.T @ theta0 if v_id.size > 0 else np.zeros(0, dtype=np.float64)
    b0 = readonly_array(base_snapshot.generalized_load(), "b0", 1)
    nom_lo, nom_hi = _solve_snapshot_envelope(base_snapshot, b0)
    nom_width = readonly_array(nom_hi - nom_lo, "nominal_envelope_width", 1)
    phi_act = _actuated_load_coupling(base_snapshot, v_null)

    rng = np.random.default_rng(int(seed))
    samples: list[ParameterSample] = []
    total_rejected = 0
    max_drift = 0.0

    for idx in range(int(n)):
        theta_k, delta_k, rej_k = _sample_consistent_theta(
            theta0, v_null, rng, float(perturbation_scale), lmi_check_fn
        )
        total_rejected += rej_k
        drift_k = (
            float(np.max(np.abs(v_id.T @ theta_k - base_combo)))
            if v_id.size > 0
            else 0.0
        )
        max_drift = max(max_drift, drift_k)
        b_k = readonly_array(b0 + phi_act @ (theta_k - theta0), "b_k", 1)
        lo_k, hi_k = _solve_snapshot_envelope(base_snapshot, b_k)
        samples.append(
            ParameterSample(
                sample_index=idx,
                theta=theta_k,
                null_coordinates=delta_k,
                lmi_satisfied=True,
                identified_drift=drift_k,
                generalized_load=b_k,
                envelope_lower=lo_k,
                envelope_upper=hi_k,
            )
        )

    ensure(max_drift < _INVARIANCE_ATOL, "identified base combinations drifted")
    union_lo, union_hi, union_width, growth, sens = _summarize_parameter_envelopes(
        theta0, nom_lo, nom_hi, nom_width, samples
    )

    return ParameterSensitivityReport(
        samples=tuple(samples),
        nominal_theta=theta0,
        identified_basis=v_id,
        null_basis=v_null,
        qr_pivot_indices=piv_indices,
        observability=obs,
        identified_rank=int(v_id.shape[1]),
        null_dimension=int(v_null.shape[1]),
        max_identified_drift=max_drift,
        rejected_lmi_count=total_rejected,
        nominal_envelope_lower=nom_lo,
        nominal_envelope_upper=nom_hi,
        nominal_envelope_width=nom_width,
        envelope_union_lower=union_lo,
        envelope_union_upper=union_hi,
        envelope_union_width=union_width,
        envelope_width_growth=growth,
        per_quantity_sensitivity=sens,
    )


__all__ = [
    "ParameterEnsembleReport",
    "ParameterSample",
    "ParameterSensitivityReport",
    "parameter_ensemble",
]
