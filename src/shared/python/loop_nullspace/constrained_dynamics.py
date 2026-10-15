"""Constrained EquationsOfMotionProvider and KKT-projected ZTCF/ZVCF (NSX-30, #12570).

Provides the single canonical implementation of KKT operational-space closure
projection M^-1(I - J_c^T Lambda_c J_c M^-1) with optional Baumgarte stabilization
and six-axis evidence tagging for closed-loop counterfactual accelerations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.contracts import (
    AllocationDependence,
    ClosedLoopSnapshot,
    FloatArray,
    QuantityRecord,
    _readonly_float_array,
    _validate_spd_mass_matrix,
)
from src.shared.python.loop_nullspace.decomposition import decompose
from src.shared.python.simulation_backends.protocol import (
    ConstrainedEquationsOfMotionProvider,
    EquationsOfMotionProvider,
)

__all__ = [
    "ConstrainedEquationsOfMotionProvider",
    "compute_constrained_ztcf_zvcf",
    "constrained_acceleration",
]

_ALLOWED_CONSTRAINED_ACCEL_OPTIONS = frozenset(
    {
        "S_T",
        "q_err",
        "v_err",
        "baumgarte_alpha",
        "baumgarte_beta",
        "regularization",
    }
)


def _solve_operational_space_multiplier(
    g_c: FloatArray, rhs_lambda: FloatArray, regularization: float
) -> FloatArray:
    """Solve (J_c M^-1 J_c^T) lambda_c = rhs_lambda via rank-revealing SVD pseudoinverse."""
    if g_c.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    u, s, vt = np.linalg.svd(g_c, full_matrices=False)
    if s.size == 0 or s[0] <= 0.0:
        return np.zeros(g_c.shape[0], dtype=np.float64)
    cutoff = max(float(regularization), 1e-14) * float(s[0])
    rank = int(np.count_nonzero(s > cutoff))
    if rank == 0:
        return np.zeros(g_c.shape[0], dtype=np.float64)
    u_r = u[:, :rank]
    s_r = s[:rank]
    vt_r = vt[:rank, :]
    return np.asarray(vt_r.T @ ((u_r.T @ rhs_lambda) / s_r), dtype=np.float64)


def _parse_constrained_accel_options(
    options: dict[str, Any], n_v: int, m_c: int
) -> tuple[FloatArray | None, FloatArray, float]:
    """Validate and extract optional keyword arguments for constrained_acceleration."""
    unknown = set(options.keys()) - _ALLOWED_CONSTRAINED_ACCEL_OPTIONS
    if unknown:
        raise TypeError(
            f"Unexpected keyword argument(s) for constrained_acceleration: {sorted(unknown)}"
        )
    s_t_raw = options.get("S_T")
    s_t = _readonly_float_array(s_t_raw, "S_T", ndim=2) if s_t_raw is not None else None
    if s_t is not None and s_t.shape[0] != n_v:
        raise ValueError(f"S_T must have {n_v} rows, got {s_t.shape}")

    alpha = float(options.get("baumgarte_alpha", 0.0))
    beta = float(options.get("baumgarte_beta", 0.0))
    reg = float(options.get("regularization", 1e-10))
    if not (
        np.isfinite(alpha) and np.isfinite(beta) and np.isfinite(reg) and reg >= 0.0
    ):
        raise ValueError("baumgarte gains and regularization must be finite (reg >= 0)")

    stabilization = np.zeros(m_c, dtype=np.float64)
    v_err = options.get("v_err")
    if v_err is not None:
        v_err_arr = _readonly_float_array(v_err, "v_err", shape=(m_c,))
        stabilization = stabilization + 2.0 * alpha * v_err_arr
    q_err = options.get("q_err")
    if q_err is not None:
        q_err_arr = _readonly_float_array(q_err, "q_err", shape=(m_c,))
        stabilization = stabilization + (beta**2) * q_err_arr
    return s_t, stabilization, reg


def constrained_acceleration(
    M: np.ndarray,
    h: np.ndarray,
    tau: np.ndarray,
    J_c: np.ndarray,
    Jdot_v: np.ndarray,
    **options: Any,
) -> tuple[FloatArray, FloatArray]:
    """Compute (q_ddot, lambda_c) via the KKT operational-space projector (NSX-30).

    Solves the coupled system with closure reactions re-solved:
        M q_ddot + h = tau_gen + J_c^T lambda_c
        J_c q_ddot + Jdot_v + 2*alpha*v_err + beta^2*q_err = 0

    Accepted keyword options:
        S_T: Optional (n_v, n_tau) actuation transpose mapping tau -> tau_gen.
        q_err: Optional (m_c,) position closure residual Phi(q).
        v_err: Optional (m_c,) velocity closure residual J_c(q) v.
        baumgarte_alpha: Velocity stabilization gain alpha >= 0 (default 0.0).
        baumgarte_beta: Position stabilization gain beta >= 0 (default 0.0).
        regularization: Relative singular-value cutoff for rank-deficient J_c (default 1e-10).
    """
    h_arr = _readonly_float_array(h, "h", ndim=1)
    n_v = int(h_arr.shape[0])
    if n_v == 0:
        raise ValueError("h must be non-empty")
    m_arr = _validate_spd_mass_matrix(np.asarray(M, dtype=np.float64), n_v)
    j_c_arr = _readonly_float_array(J_c, "J_c", ndim=2)
    if j_c_arr.shape[1] != n_v:
        raise ValueError(f"J_c column count {j_c_arr.shape[1]} must match n_v ({n_v})")
    m_c = int(j_c_arr.shape[0])
    jdot_arr = _readonly_float_array(Jdot_v, "Jdot_v", shape=(m_c,))
    tau_arr = _readonly_float_array(tau, "tau", ndim=1)

    s_t, stabilization, reg = _parse_constrained_accel_options(options, n_v, m_c)
    if s_t is not None:
        if tau_arr.shape != (s_t.shape[1],):
            raise ValueError(
                f"tau shape {tau_arr.shape} must match S_T columns ({s_t.shape[1]},)"
            )
        tau_gen = np.asarray(s_t @ tau_arr, dtype=np.float64)
    else:
        if tau_arr.shape != (n_v,):
            raise ValueError(f"tau must have shape ({n_v},), got {tau_arr.shape}")
        tau_gen = tau_arr

    drift_total = jdot_arr + stabilization
    minv_rhs = np.linalg.solve(m_arr, tau_gen - h_arr)
    minv_jct = np.linalg.solve(m_arr, j_c_arr.T)
    g_c = np.asarray(j_c_arr @ minv_jct, dtype=np.float64)
    rhs_lambda = -(drift_total + j_c_arr @ minv_rhs)
    lambda_c = _solve_operational_space_multiplier(g_c, rhs_lambda, reg)
    q_ddot = minv_rhs + minv_jct @ lambda_c

    return (
        _readonly_float_array(q_ddot, "q_ddot", shape=(n_v,)),
        _readonly_float_array(lambda_c, "lambda_c", shape=(m_c,)),
    )


def _operational_space_projector(
    M: FloatArray, J_c: FloatArray, regularization: float = 1e-10
) -> FloatArray:
    """Compute the KKT operational-space control projector P_KKT = M^-1(I - J_c^T Lambda_c J_c M^-1)."""
    n_v = int(M.shape[0])
    minv = np.linalg.inv(M)
    g_c = J_c @ minv @ J_c.T
    u, s, vt = np.linalg.svd(g_c, full_matrices=False)
    if s.size > 0 and s[0] > 0.0:
        cutoff = max(float(regularization), 1e-14) * float(s[0])
        rank = int(np.count_nonzero(s > cutoff))
        lambda_c = (
            vt[:rank, :].T @ np.diag(1.0 / s[:rank]) @ u[:, :rank].T
            if rank > 0
            else np.zeros_like(g_c)
        )
    else:
        lambda_c = np.zeros_like(g_c)
    proj = minv @ (np.eye(n_v, dtype=np.float64) - J_c.T @ lambda_c @ J_c @ minv)
    return _readonly_float_array(proj, "P_KKT", shape=(n_v, n_v))


def _package_constrained_ztcf_records(
    snapshot: ClosedLoopSnapshot,
    a_ztcf: FloatArray,
    a_zvcf: FloatArray,
    a_ctrl_total: FloatArray,
    per_actuator: FloatArray,
) -> dict[str, QuantityRecord]:
    """Wrap constrained ZTCF, ZVCF, control, and attribution arrays into QuantityRecords."""
    model_prov = snapshot.model_provenance
    input_prov = snapshot.input_provenance
    ztcf_rec = QuantityRecord(
        values=a_ztcf,
        units="rad/s^2",
        allocation_dependence=AllocationDependence.INVARIANT,
        conventions=("C-CONSTRAINED-ZTCF",),
        hypotheses=(),
        model_provenance=model_prov,
        input_provenance=input_prov,
    )
    zvcf_rec = QuantityRecord(
        values=a_zvcf,
        units="rad/s^2",
        allocation_dependence=AllocationDependence.INVARIANT,
        conventions=("C-CONSTRAINED-ZVCF",),
        hypotheses=(),
        model_provenance=model_prov,
        input_provenance=input_prov,
    )
    ctrl_rec = QuantityRecord(
        values=a_ctrl_total,
        units="rad/s^2",
        allocation_dependence=AllocationDependence.INVARIANT,
        conventions=("C-CONSTRAINED-ZTCF",),
        hypotheses=(),
        model_provenance=model_prov,
        input_provenance=input_prov,
    )
    attr_rec = QuantityRecord(
        values=per_actuator,
        units="rad/s^2",
        allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
        conventions=("C-CONSTRAINED-ZTCF",),
        hypotheses=("H-PARTICULAR-PSEUDOINVERSE",),
        model_provenance=model_prov,
        input_provenance=input_prov,
    )
    return {
        "ztcf": ztcf_rec,
        "a_ztcf": ztcf_rec,
        "zvcf": zvcf_rec,
        "a_zvcf": zvcf_rec,
        "control_acceleration": ctrl_rec,
        "a_ctrl_total": ctrl_rec,
        "per_actuator_attribution": attr_rec,
    }


def compute_constrained_ztcf_zvcf(
    snapshot: ClosedLoopSnapshot,
    *,
    provider: EquationsOfMotionProvider | None = None,
    regularization: float = 1e-10,
) -> dict[str, QuantityRecord]:
    """Compute constrained ZTCF, constrained ZVCF, and invariant vs family-dependent attributions."""
    if not isinstance(snapshot, ClosedLoopSnapshot):
        raise TypeError("snapshot must be a ClosedLoopSnapshot instance")

    n_v = snapshot.nv()
    m_c = snapshot.nc()
    zero_tau = np.zeros(n_v, dtype=np.float64)

    a_ztcf, _ = constrained_acceleration(
        snapshot.M,
        snapshot.h,
        zero_tau,
        snapshot.J_c,
        snapshot.Jdot_v,
        regularization=regularization,
    )
    h_zero_v = (
        np.asarray(
            provider.bias_forces(snapshot.q, np.zeros(n_v, dtype=np.float64)),
            dtype=np.float64,
        )
        if provider is not None
        else snapshot.h
    )
    a_zvcf, _ = constrained_acceleration(
        snapshot.M,
        h_zero_v,
        zero_tau,
        snapshot.J_c,
        np.zeros(m_c, dtype=np.float64),
        regularization=regularization,
    )

    a_ctrl_total = snapshot.a - a_ztcf
    decomp = decompose(snapshot)
    n_tau = int(snapshot.S.shape[0])
    tau_p = decomp.x_particular[:n_tau]
    p_kkt = _operational_space_projector(snapshot.M, snapshot.J_c, regularization)

    per_actuator = np.zeros((n_tau, n_v), dtype=np.float64)
    for k in range(n_tau):
        per_actuator[k, :] = p_kkt @ (snapshot.S[k, :] * float(tau_p[k]))

    return _package_constrained_ztcf_records(
        snapshot, a_ztcf, a_zvcf, a_ctrl_total, per_actuator
    )
