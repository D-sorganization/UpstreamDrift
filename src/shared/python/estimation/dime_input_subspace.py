"""DIME Torque-Independent Drift Feasibility and Missing-Data Prediction (#11421, #11437).

Enforces physical input-effect subspace decomposition, covariance whitening,
orthogonal-complement torque-independent drift feasibility tests, bounded control
feasibility, changing contact rank tracking, and missing-data prediction with
calibrated uncertainty bounds.

Scope and Invariants:
- Torque-independent drift feasibility: tests acceleration residuals outside the
  physically allowed input-effect subspace using covariance whitening and rank-revealing
  factorization. Outside this subspace, no torque can explain discrepancies.
- Fully actuated systems: orthogonal projection dimension is zero, adding no
  extraneous physical constraints.
- Underactuated systems: unactuated coordinates are strictly constrained by drift
  and eliminated contacts; arbitrary motion is detected and rejected fail-closed.
- Bounded control feasibility: tests whether required control effort inside the
  input-effect subspace lies within admissible actuator torque bounds.
- Missing-data prediction: across poor-information or masked intervals, uncertainty
  grows along actuated directions according to control priors, while unactuated
  directions strictly follow deterministic drift without conjuring artificial motion.
- Runtime exclusivity: serves as a proposal-screening diagnostic by default.
  When used as an alternative reduced inference formulation, runtime exclusivity
  checks reject duplicate full-dynamics factors on overlapping intervals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np

from src.shared.python.contracts import PreconditionError
from src.shared.python.core.contracts import check_finite, ensure, require
from src.shared.python.estimation.dime_contracts import (
    DimeCompleteState,
    EstimationIntervalFactor,
)
from src.shared.python.estimation.drift_prediction import ControlBand
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

DIME_INPUT_SUBSPACE_VERSION: Final[str] = "1.0.0"
SubspaceFactorMode = Literal["diagnostic", "reduced_subspace_dynamics"]

__all__ = [
    "DIME_INPUT_SUBSPACE_VERSION",
    "DimeInputSubspaceFactor",
    "DimeSubspaceReport",
    "MaskedPredictionResult",
    "SubspaceDecomposition",
    "SubspaceFactorMode",
    "SubspaceFeasibilityEvaluation",
    "decompose_input_subspace",
    "evaluate_input_subspace_feasibility",
    "predict_masked_interval",
]


def _finite_vector(name: str, value: np.ndarray) -> np.ndarray:
    """Validate and return 1-D finite float vector."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    require(arr.size > 0, f"{name} must be non-empty", value=arr.shape)
    require(check_finite(arr), f"{name} must contain only finite values", value=arr)
    return arr


def _finite_matrix(name: str, value: np.ndarray) -> np.ndarray:
    """Validate and return 2-D finite float matrix."""
    mat = np.asarray(value, dtype=np.float64)
    require(mat.ndim == 2, f"{name} must be a 2D matrix", value=mat.shape)
    require(mat.size > 0, f"{name} must be non-empty", value=mat.shape)
    require(check_finite(mat), f"{name} must contain only finite values", value=mat)
    return mat


def _compute_whitening_operator(cov: np.ndarray) -> np.ndarray:
    """Compute symmetric inverse square root W = cov^{-1/2} via eigendecomposition.

    Guarantees scale-invariance: scaling cov by alpha scales W by 1/sqrt(alpha).
    """
    n = cov.shape[0]
    require(cov.shape == (n, n), "covariance must be square", cov.shape)
    require(np.allclose(cov, cov.T, atol=1e-12), "covariance must be symmetric")
    eigvals, eigvecs = np.linalg.eigh(cov)
    require(
        bool(np.all(eigvals > 1e-15)),
        "covariance must be strictly positive definite",
        value=eigvals,
    )
    inv_sqrt_vals = 1.0 / np.sqrt(eigvals)
    w_mat = eigvecs @ np.diag(inv_sqrt_vals) @ eigvecs.T
    return 0.5 * (w_mat + w_mat.T)


@dataclass(frozen=True)
class SubspaceDecomposition:
    """Rank-revealing decomposition of the whitened input-effect matrix."""

    input_matrix: np.ndarray
    whitened_input_matrix: np.ndarray
    whitening_transform: np.ndarray
    subspace_basis: np.ndarray
    orthogonal_basis: np.ndarray
    singular_values: tuple[float, ...]
    rank: int
    condition_number: float
    tolerance: float

    @property
    def is_fully_actuated(self) -> bool:
        """True if actuated rank equals total coordinate dimension."""
        return self.rank == self.input_matrix.shape[0]

    @property
    def is_underactuated(self) -> bool:
        """True if actuated rank is less than total coordinate dimension."""
        return self.rank < self.input_matrix.shape[0]

    @property
    def is_singular(self) -> bool:
        """True if input map is numerically singular or rank deficient."""
        return math.isinf(self.condition_number) or self.condition_number > 1e10


def decompose_input_subspace(
    input_influence: np.ndarray,
    covariance: np.ndarray,
    *,
    tolerance: float | None = None,
) -> SubspaceDecomposition:
    """Decompose input influence matrix into whitened actuated and unactuated subspaces."""
    b_mat = _finite_matrix("input_influence", input_influence)
    n, m = b_mat.shape
    cov = _finite_matrix("covariance", covariance)
    require(
        cov.shape == (n, n),
        "covariance shape must match input matrix rows",
        (cov.shape, n),
    )

    w_mat = _compute_whitening_operator(cov)
    b_w = w_mat @ b_mat

    u_mat, s_vec, _vt_mat = np.linalg.svd(b_w, full_matrices=True)
    max_s = float(np.max(s_vec)) if s_vec.size > 0 else 0.0
    tol = tolerance if tolerance is not None else max(1e-12, max_s * max(n, m) * 1e-12)

    rank = int(np.sum(s_vec > tol))
    cond_num = (
        float(s_vec[0] / s_vec[-1])
        if rank == min(n, m) and s_vec[-1] > 0.0
        else float("inf")
    )

    u_par = u_mat[:, :rank]
    u_perp = u_mat[:, rank:]

    return SubspaceDecomposition(
        input_matrix=b_mat,
        whitened_input_matrix=b_w,
        whitening_transform=w_mat,
        subspace_basis=u_par,
        orthogonal_basis=u_perp,
        singular_values=tuple(float(s) for s in s_vec),
        rank=rank,
        condition_number=cond_num,
        tolerance=tol,
    )


@dataclass(frozen=True)
class SubspaceFeasibilityEvaluation:
    """Outcome of torque-independent drift and bounded control feasibility evaluation."""

    candidate_acceleration: np.ndarray
    drift_acceleration: np.ndarray
    contact_acceleration: np.ndarray
    effective_drift: np.ndarray
    raw_residual: np.ndarray
    whitened_residual: np.ndarray
    orthogonal_residual: np.ndarray
    orthogonal_chi2: float
    is_drift_feasible: bool
    optimal_torque: np.ndarray | None
    is_control_feasible: bool
    is_overall_feasible: bool
    contact_rank: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


def _solve_subspace_torque(
    decomp: SubspaceDecomposition,
    whitened_residual: np.ndarray,
) -> np.ndarray:
    """Solve unconstrained least-squares torque inside actuated subspace."""
    b_w = decomp.whitened_input_matrix
    rank = decomp.rank
    if rank == 0:
        return np.zeros(b_w.shape[1], dtype=np.float64)
    # Solve B_w @ tau = r_w via pseudoinverse
    tau_opt, _, _, _ = np.linalg.lstsq(b_w, whitened_residual, rcond=decomp.tolerance)
    return tau_opt


def evaluate_input_subspace_feasibility(
    candidate_accel: np.ndarray,
    drift_accel: np.ndarray,
    input_matrix: np.ndarray,
    covariance: np.ndarray,
    *,
    control_band: ControlBand | None = None,
    contact_jacobian: np.ndarray | None = None,
    contact_forces: np.ndarray | None = None,
    mass_matrix: np.ndarray | None = None,
    tolerance: float | None = None,
    threshold_chi2: float = 9.0,
) -> SubspaceFeasibilityEvaluation:
    """Test acceleration feasibility outside and inside the input-effect subspace."""
    a_cand = _finite_vector("candidate_accel", candidate_accel)
    a_drift = _finite_vector("drift_accel", drift_accel)
    n = a_cand.size
    require(
        a_drift.size == n,
        "drift_accel size must match candidate_accel",
        (a_drift.size, n),
    )

    # Incorporate eliminated contact reactions
    a_contact = np.zeros(n, dtype=np.float64)
    contact_rank = 0
    if contact_forces is not None:
        forces = _finite_vector("contact_forces", contact_forces)
        if contact_jacobian is None or mass_matrix is None:
            raise PreconditionError(
                "contact_jacobian and mass_matrix required when contact_forces provided"
            )
        j_c = _finite_matrix("contact_jacobian", contact_jacobian)
        m_mat = _finite_matrix("mass_matrix", mass_matrix)
        require(
            j_c.shape[0] == forces.size,
            "contact_jacobian rows must match contact_forces",
        )
        require(j_c.shape[1] == n, "contact_jacobian cols must match coordinate size")
        require(m_mat.shape == (n, n), "mass_matrix must be square with size n")

        contact_load = j_c.T @ forces
        a_contact = np.linalg.solve(m_mat, contact_load)
        contact_rank = int(np.linalg.matrix_rank(j_c))

    effective_drift = a_drift + a_contact
    raw_res = a_cand - effective_drift

    decomp = decompose_input_subspace(input_matrix, covariance, tolerance=tolerance)
    w_mat = decomp.whitening_transform
    r_w = w_mat @ raw_res

    # Orthogonal projection: unactuated / torque-independent subspace
    u_perp = decomp.orthogonal_basis
    if u_perp.shape[1] == 0:
        # Fully actuated system: projection onto orthogonal complement adds no information
        r_perp = np.zeros(0, dtype=np.float64)
        chi2_perp = 0.0
        drift_feasible = True
    else:
        r_perp = u_perp.T @ r_w
        chi2_perp = float(np.sum(r_perp**2))
        drift_feasible = chi2_perp <= threshold_chi2

    # Inside actuated subspace: bounded control feasibility
    optimal_tau = None
    control_feasible = True
    if decomp.rank > 0:
        optimal_tau = _solve_subspace_torque(decomp, r_w)
        if control_band is not None:
            lo, hi = control_band.lower, control_band.upper
            tol_bound = 1e-5
            within_box = np.all(optimal_tau >= lo - tol_bound) and np.all(
                optimal_tau <= hi + tol_bound
            )
            control_feasible = bool(within_box)

    overall_feasible = bool(drift_feasible and control_feasible)

    diag: dict[str, Any] = {
        "is_fully_actuated": decomp.is_fully_actuated,
        "is_underactuated": decomp.is_underactuated,
        "rank": decomp.rank,
        "condition_number": decomp.condition_number,
        "contact_rank": contact_rank,
    }

    return SubspaceFeasibilityEvaluation(
        candidate_acceleration=a_cand,
        drift_acceleration=a_drift,
        contact_acceleration=a_contact,
        effective_drift=effective_drift,
        raw_residual=raw_res,
        whitened_residual=r_w,
        orthogonal_residual=r_perp,
        orthogonal_chi2=chi2_perp,
        is_drift_feasible=drift_feasible,
        optimal_torque=optimal_tau,
        is_control_feasible=control_feasible,
        is_overall_feasible=overall_feasible,
        contact_rank=contact_rank,
        diagnostics=diag,
    )


@dataclass(frozen=True)
class MaskedPredictionResult:
    """Calibrated trajectory prediction across masked/missing observation intervals."""

    trajectory_mean: np.ndarray
    trajectory_covariance: np.ndarray
    horizon_valid: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


def predict_masked_interval(
    initial_state: DimeCompleteState,
    input_matrix: np.ndarray,
    drift_accel: np.ndarray,
    control_prior_mean: np.ndarray,
    control_prior_cov: np.ndarray,
    observation_mask: np.ndarray,
    dt: float,
    horizon: int,
    *,
    model_discrepancy: np.ndarray | None = None,
) -> MaskedPredictionResult:
    """Predict state propagation over an interval with masked/missing observation data."""
    require(dt > 0.0 and math.isfinite(dt), "dt must be positive and finite", dt)
    require(horizon >= 1, "horizon must be at least 1", horizon)
    mask = np.asarray(observation_mask, dtype=bool)
    require(
        mask.size == horizon,
        "observation_mask length must equal horizon",
        (mask.size, horizon),
    )

    q0 = _finite_vector("initial_state.q", initial_state.q)
    v0 = _finite_vector("initial_state.v", initial_state.v)
    n = q0.size
    require(v0.size == n, "initial_state v size must match q", (v0.size, n))

    b_mat = _finite_matrix("input_matrix", input_matrix)
    require(
        b_mat.shape[0] == n,
        "input_matrix rows must match coordinate dimension",
        (b_mat.shape, n),
    )
    m = b_mat.shape[1]

    u_mean = _finite_vector("control_prior_mean", control_prior_mean)
    u_cov = _finite_matrix("control_prior_cov", control_prior_cov)
    require(
        u_mean.size == m,
        "control_prior_mean size must match input columns",
        (u_mean.size, m),
    )
    require(
        u_cov.shape == (m, m),
        "control_prior_cov shape must be (m, m)",
        (u_cov.shape, m),
    )

    a_drift = _finite_vector("drift_accel", drift_accel)
    require(a_drift.size == n, "drift_accel size must match coordinate dimension")

    state_dim = 2 * n
    traj_mean = np.zeros((horizon + 1, state_dim), dtype=np.float64)
    traj_cov = np.zeros((horizon + 1, state_dim, state_dim), dtype=np.float64)

    traj_mean[0, :n] = q0
    traj_mean[0, n:] = v0

    q_w = (
        _finite_matrix("model_discrepancy", model_discrepancy)
        if model_discrepancy is not None
        else np.zeros((state_dim, state_dim), dtype=np.float64)
    )
    require(q_w.shape == (state_dim, state_dim), "model_discrepancy must be (2n, 2n)")

    f_mat = np.block([[np.eye(n), dt * np.eye(n)], [np.zeros((n, n)), np.eye(n)]])
    g_mat = np.vstack([0.5 * (dt**2) * b_mat, dt * b_mat])

    for k in range(horizon):
        qk = traj_mean[k, :n]
        vk = traj_mean[k, n:]
        pk = traj_cov[k]

        a_eff = a_drift + (b_mat @ u_mean)
        q_next = qk + dt * vk + 0.5 * (dt**2) * a_eff
        v_next = vk + dt * a_eff

        traj_mean[k + 1, :n] = q_next
        traj_mean[k + 1, n:] = v_next

        # Propagate uncertainty: F P F^T + G Sigma_u G^T + Q_w
        p_next = f_mat @ pk @ f_mat.T + g_mat @ u_cov @ g_mat.T + q_w
        traj_cov[k + 1] = 0.5 * (p_next + p_next.T)

    ensure(check_finite(traj_mean), "predicted mean trajectory must be finite")
    ensure(check_finite(traj_cov), "predicted covariance trajectory must be finite")

    return MaskedPredictionResult(
        trajectory_mean=traj_mean,
        trajectory_covariance=traj_cov,
        horizon_valid=True,
        diagnostics={"horizon": horizon, "missing_samples": int(np.sum(mask))},
    )


@dataclass(frozen=True)
class DimeInputSubspaceFactor:
    """Factor representation for runtime estimation and exclusivity management."""

    name: str
    t_start: float
    t_end: float
    mode: SubspaceFactorMode = "diagnostic"
    contributes_to_objective: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(
            self.t_start <= self.t_end,
            "t_start must be <= t_end",
            (self.t_start, self.t_end),
        )
        if self.mode == "diagnostic":
            require(
                not self.contributes_to_objective,
                "diagnostic factor cannot contribute to estimation objective",
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_interval_factor(self) -> EstimationIntervalFactor:
        """Convert to EstimationIntervalFactor for registration in exclusivity contract."""
        f_type = (
            "reduced_subspace_dynamics"
            if self.contributes_to_objective
            else "diagnostic"
        )
        return EstimationIntervalFactor(
            name=self.name,
            factor_type=f_type,
            t_start=self.t_start,
            t_end=self.t_end,
            contributes_to_objective=self.contributes_to_objective,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class DimeSubspaceReport:
    """Serializable structured audit receipt for subspace feasibility and prediction."""

    time_s: float
    dimension_n: int
    actuated_dimension_m: int
    subspace_rank_r: int
    unactuated_dimension: int
    singular_values: tuple[float, ...]
    condition_number: float
    drift_acceleration: tuple[float, ...]
    unactuated_residual_norm: float
    is_drift_feasible: bool
    is_control_feasible: bool
    is_overall_feasible: bool
    optimal_torque: tuple[float, ...] | None = None
    prediction_horizon_valid: bool = True
    contact_active: bool = False
    contact_rank: int = 0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "time_s": float(self.time_s),
            "dimension_n": int(self.dimension_n),
            "actuated_dimension_m": int(self.actuated_dimension_m),
            "subspace_rank_r": int(self.subspace_rank_r),
            "unactuated_dimension": int(self.unactuated_dimension),
            "singular_values": list(self.singular_values),
            "condition_number": float(self.condition_number),
            "drift_acceleration": list(self.drift_acceleration),
            "unactuated_residual_norm": float(self.unactuated_residual_norm),
            "is_drift_feasible": bool(self.is_drift_feasible),
            "is_control_feasible": bool(self.is_control_feasible),
            "is_overall_feasible": bool(self.is_overall_feasible),
            "optimal_torque": list(self.optimal_torque)
            if self.optimal_torque is not None
            else None,
            "prediction_horizon_valid": bool(self.prediction_horizon_valid),
            "contact_active": bool(self.contact_active),
            "contact_rank": int(self.contact_rank),
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeSubspaceReport:
        """Create report from dictionary representation."""
        tau_raw = data.get("optimal_torque")
        return cls(
            time_s=float(data["time_s"]),
            dimension_n=int(data["dimension_n"]),
            actuated_dimension_m=int(data["actuated_dimension_m"]),
            subspace_rank_r=int(data["subspace_rank_r"]),
            unactuated_dimension=int(data["unactuated_dimension"]),
            singular_values=tuple(float(s) for s in data["singular_values"]),
            condition_number=float(data["condition_number"]),
            drift_acceleration=tuple(float(a) for a in data["drift_acceleration"]),
            unactuated_residual_norm=float(data["unactuated_residual_norm"]),
            is_drift_feasible=bool(data["is_drift_feasible"]),
            is_control_feasible=bool(data["is_control_feasible"]),
            is_overall_feasible=bool(data["is_overall_feasible"]),
            optimal_torque=tuple(float(t) for t in tau_raw)
            if tau_raw is not None
            else None,
            prediction_horizon_valid=bool(data.get("prediction_horizon_valid", True)),
            contact_active=bool(data.get("contact_active", False)),
            contact_rank=int(data.get("contact_rank", 0)),
            diagnostics=dict(data.get("diagnostics", {})),
        )
