"""Trajectory Conic Allocation With Strict Curvature Guarantee (NSX-23 / NSX-24, #12566, #12567).

Implements prior-conditioned trajectory allocation over the closed-loop equality null space:
    x_t(z_t) = x_{p,t} + N_t z_t
with strictly convex quadratic curvature (R6), capacity-weighted effort, discrete torque-rate
smoothness, asymmetric c+/c- gross mechanical power epigraphs, and optional reference tracking.
All allocated outputs are tagged FAMILY_DEPENDENT with explicit hypothesis provenance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

Array: TypeAlias = NDArray[np.float64]

CURVATURE_MIN_EIGENVALUE: float = 1e-8
DEFAULT_HYPOTHESIS_ID: str = "H-ALLOC-CONVEX-PRIOR"
DEFAULT_CONVENTION_ID: str = "C-METRIC-CAPACITY"


def _readonly_array(values: Any, name: str, *, ndim: int | None = None) -> Array:
    """Convert input to a finite, read-only float64 ndarray."""
    arr = np.array(values, dtype=np.float64, copy=True)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {arr.ndim}")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class ObjectiveSpec:
    """Prior-conditioned strictly convex trajectory allocation objective specification."""

    effort_weight: float = 1.0
    rate_weight: float = 0.0
    c_plus: float | Array = 0.0
    c_minus: float | Array = 0.0
    reference_weight: float = 0.0
    tau_max: Array | Sequence[float] | None = None
    hypotheses: tuple[str, ...] = (DEFAULT_HYPOTHESIS_ID,)
    conventions: tuple[str, ...] = (DEFAULT_CONVENTION_ID,)

    def __post_init__(self) -> None:
        for attr in ("effort_weight", "rate_weight", "reference_weight"):
            val = float(getattr(self, attr))
            if not np.isfinite(val) or val < 0.0:
                raise ValueError(f"{attr} must be finite and non-negative, got {val}")
            object.__setattr__(self, attr, val)

        cp = np.asarray(self.c_plus, dtype=np.float64)
        cm = np.asarray(self.c_minus, dtype=np.float64)
        if not np.isfinite(cp).all() or np.any(cp < 0.0):
            raise ValueError("c_plus must be finite and non-negative")
        if not np.isfinite(cm).all() or np.any(cm < 0.0):
            raise ValueError("c_minus must be finite and non-negative")
        if cp.ndim == 0:
            object.__setattr__(self, "c_plus", float(cp))
        else:
            object.__setattr__(self, "c_plus", _readonly_array(cp, "c_plus", ndim=1))
        if cm.ndim == 0:
            object.__setattr__(self, "c_minus", float(cm))
        else:
            object.__setattr__(self, "c_minus", _readonly_array(cm, "c_minus", ndim=1))

        if self.tau_max is not None:
            tm = _readonly_array(self.tau_max, "tau_max", ndim=1)
            if np.any(tm <= 0.0):
                raise ValueError("tau_max entries must be strictly positive")
            object.__setattr__(self, "tau_max", tm)

        hyps = tuple(str(h) for h in self.hypotheses if str(h).strip())
        convs = tuple(str(c) for c in self.conventions if str(c).strip())
        object.__setattr__(self, "hypotheses", hyps or (DEFAULT_HYPOTHESIS_ID,))
        object.__setattr__(self, "conventions", convs or (DEFAULT_CONVENTION_ID,))


@dataclass(frozen=True)
class GrossPowerDecomposition:
    """R6 asymmetric c+/c- gross-power identity decomposition and dead-ratio audit."""

    lhs_asymmetric_cost: float
    rhs_identity_cost: float
    identity_residual: float
    gross_abs_power: float
    net_power: float
    positive_power: float
    negative_power: float
    c_plus: float
    c_minus: float
    is_symmetric_weights: bool
    is_uniform_across_joints: bool
    closure_velocity_residual: float
    net_power_nullspace_sensitivity: float
    is_dead_ratio_parameter: bool


@dataclass(frozen=True)
class TrajectoryFeasibleFrame:
    """Single-frame null-space parameterization tau(z) = tau_p + N_tau z and constraints."""

    tau_particular: Array
    null_basis_tau: Array
    lambda_particular: Array | None = None
    null_basis_lambda: Array | None = None
    v_actuated: Array | None = None
    J_c: Array | None = None
    tau_lower: Array | None = None
    tau_upper: Array | None = None
    z_lower: Array | None = None
    z_upper: Array | None = None
    A_ub_z: Array | None = None
    b_ub_z: Array | None = None
    tau_capacity: Array | None = None
    hypotheses: tuple[str, ...] = ("H-BOX-CAPACITY",)

    def __post_init__(self) -> None:
        tp = _readonly_array(self.tau_particular, "tau_particular", ndim=1)
        nt = _readonly_array(self.null_basis_tau, "null_basis_tau", ndim=2)
        if nt.shape[0] != tp.shape[0]:
            raise ValueError(
                f"null_basis_tau row count {nt.shape[0]} must match tau_particular ({tp.shape[0]})"
            )
        object.__setattr__(self, "tau_particular", tp)
        object.__setattr__(self, "null_basis_tau", nt)
        if self.tau_capacity is not None:
            cap = _readonly_array(self.tau_capacity, "tau_capacity", ndim=1)
            if np.any(cap <= 0.0) or cap.shape != tp.shape:
                raise ValueError(
                    "tau_capacity must be positive and match tau_particular shape"
                )
            object.__setattr__(self, "tau_capacity", cap)

    def n_tau(self) -> int:
        """Return actuated torque dimension."""
        return int(self.tau_particular.shape[0])

    def n_z(self) -> int:
        """Return equality null-space dimension."""
        return int(self.null_basis_tau.shape[1])


FeasibleSet: TypeAlias = TrajectoryFeasibleFrame


@dataclass(frozen=True)
class AllocationQuantityRecord:
    """Evidence-tagged quantity record for allocated trajectory outputs (R1)."""

    values: Array
    units: str = "N*m"
    allocation_dependence: str = "FAMILY_DEPENDENT"
    conventions: tuple[str, ...] = (DEFAULT_CONVENTION_ID,)
    hypotheses: tuple[str, ...] = (DEFAULT_HYPOTHESIS_ID,)
    validation_status: str = "UNVALIDATED"


@dataclass(frozen=True)
class AllocationResult:
    """Trajectory conic allocation solution, KKT certificate, and R6 power audit."""

    tau: Array
    lambda_c: Array
    z: Array
    duals: Array
    equality_duals: Array
    kkt_residual_norm: float
    primal_residual: float
    dual_residual: float
    complementarity_residual: float
    objective_value: float
    solve_time_s: float
    converged: bool
    gross_power_info: GrossPowerDecomposition
    evidence_class: str = "FAMILY_DEPENDENT"
    allocation_dependence: str = "FAMILY_DEPENDENT"
    hypotheses: tuple[str, ...] = (DEFAULT_HYPOTHESIS_ID,)
    conventions: tuple[str, ...] = (DEFAULT_CONVENTION_ID,)
    quantity_record: AllocationQuantityRecord = field(init=False)

    def __post_init__(self) -> None:
        if self.evidence_class != "FAMILY_DEPENDENT":
            raise ValueError("Allocated trajectory quantities must be FAMILY_DEPENDENT")
        if len(self.hypotheses) == 0:
            raise ValueError("FAMILY_DEPENDENT AllocationResult requires hypotheses")
        rec = AllocationQuantityRecord(
            values=self.tau,
            units="N*m",
            allocation_dependence="FAMILY_DEPENDENT",
            conventions=self.conventions,
            hypotheses=self.hypotheses,
        )
        object.__setattr__(self, "quantity_record", rec)

    def to_quantity_record(self) -> AllocationQuantityRecord:
        """Return the evidence-tagged QuantityRecord representation."""
        return self.quantity_record


class ZeroSideConsistency(str, Enum):
    """Three-case classification for locking a limb/side to zero torque (NSX-15 / NSX-24)."""

    CONSISTENT_UNIQUE = "consistent_unique"
    CONSISTENT_NONUNIQUE = "consistent_nonunique"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True)
class ZeroSideCheckResult:
    """Result of checking whether zeroing specified joints is algebraically consistent."""

    status: ZeroSideConsistency
    residual_norm: float
    residual_scaled: float
    nullity: int


def check_zero_side_consistency(
    tau_rnea: Array,
    actuated_indices: Sequence[int] | NDArray[np.integer[Any]],
    zero_joint_indices: Sequence[int] | NDArray[np.integer[Any]],
    j_ground: Array | None = None,
    j_grip: Array | None = None,
    *,
    atol: float = 1e-4,
) -> ZeroSideCheckResult:
    """Classify zero-side constraint consistency into UNIQUE, NONUNIQUE, or INCONSISTENT."""
    b = np.asarray(tau_rnea, dtype=np.float64).ravel()
    nv = b.size
    zero_set = {int(i) for i in zero_joint_indices}
    free_act = [int(i) for i in actuated_indices if int(i) not in zero_set]

    s_free = np.zeros((nv, len(free_act)), dtype=np.float64)
    for col, idx in enumerate(free_act):
        s_free[idx, col] = 1.0

    blocks: list[Array] = [s_free]
    if j_ground is not None and j_ground.size > 0:
        blocks.append(np.asarray(j_ground, dtype=np.float64).T)
    if j_grip is not None and j_grip.size > 0:
        blocks.append(np.asarray(j_grip, dtype=np.float64).T)
    a_mat = np.hstack(blocks) if blocks else np.zeros((nv, 0), dtype=np.float64)

    if a_mat.shape[1] == 0:
        res_norm = float(np.linalg.norm(b, ord=np.inf))
        rank = 0
    else:
        sol, _, rank_val, _ = np.linalg.lstsq(a_mat, b, rcond=None)
        res_norm = float(np.linalg.norm(a_mat @ sol - b, ord=np.inf))
        rank = int(rank_val)

    scale = max(1.0, float(np.linalg.norm(b, ord=np.inf)))
    res_scaled = res_norm / scale
    nullity = max(0, int(a_mat.shape[1]) - rank)
    if res_norm > atol:
        status = ZeroSideConsistency.INCONSISTENT
    elif nullity == 0:
        status = ZeroSideConsistency.CONSISTENT_UNIQUE
    else:
        status = ZeroSideConsistency.CONSISTENT_NONUNIQUE
    return ZeroSideCheckResult(
        status=status,
        residual_norm=res_norm,
        residual_scaled=res_scaled,
        nullity=nullity,
    )


def decompose_gross_power_cost(
    power_samples: Array,
    c_plus: float | Array,
    c_minus: float | Array,
    *,
    j_c_v: Array | None = None,
    null_basis_tau: Array | None = None,
    v: Array | None = None,
    tol: float = 1e-6,
) -> GrossPowerDecomposition:
    """Verify the R6 c+/c- gross-power identity and detect dead uniform ratio parameters."""
    p = _readonly_array(power_samples, "power_samples")
    cp = np.asarray(c_plus, dtype=np.float64)
    cm = np.asarray(c_minus, dtype=np.float64)
    if np.any(cp < 0.0) or np.any(cm < 0.0):
        raise ValueError("c_plus and c_minus must be non-negative")

    p_pos = np.maximum(p, 0.0)
    p_neg = np.maximum(-p, 0.0)
    lhs = float(np.sum(cp * p_pos + cm * p_neg))
    rhs = float(np.sum(0.5 * (cp + cm) * np.abs(p) + 0.5 * (cp - cm) * p))
    identity_res = abs(lhs - rhs)

    cp_uniform = bool(cp.ndim == 0 or np.allclose(cp, cp.flat[0], atol=1e-12))
    cm_uniform = bool(cm.ndim == 0 or np.allclose(cm, cm.flat[0], atol=1e-12))
    is_uniform = cp_uniform and cm_uniform
    cp_scalar = float(cp.flat[0]) if cp.size > 0 else 0.0
    cm_scalar = float(cm.flat[0]) if cm.size > 0 else 0.0
    is_sym = bool(is_uniform and abs(cp_scalar - cm_scalar) <= 1e-12)

    jc_v_res = (
        0.0
        if j_c_v is None
        else float(np.linalg.norm(np.asarray(j_c_v, dtype=np.float64), ord=np.inf))
    )
    ns_sens = 0.0
    if null_basis_tau is not None and v is not None:
        nt = np.asarray(null_basis_tau, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        if nt.size > 0:
            ns_sens = float(np.linalg.norm(nt.T @ v_arr, ord=np.inf))

    power_active = (cp_scalar + cm_scalar) > 0.0
    manifold_invariant = jc_v_res <= tol and ns_sens <= tol
    is_dead_ratio = bool(is_uniform and power_active and manifold_invariant)

    return GrossPowerDecomposition(
        lhs_asymmetric_cost=lhs,
        rhs_identity_cost=rhs,
        identity_residual=identity_res,
        gross_abs_power=float(np.sum(np.abs(p))),
        net_power=float(np.sum(p)),
        positive_power=float(np.sum(p_pos)),
        negative_power=float(np.sum(p_neg)),
        c_plus=cp_scalar,
        c_minus=cm_scalar,
        is_symmetric_weights=is_sym,
        is_uniform_across_joints=is_uniform,
        closure_velocity_residual=jc_v_res,
        net_power_nullspace_sensitivity=ns_sens,
        is_dead_ratio_parameter=is_dead_ratio,
    )


def _normalize_frame(item: Any) -> TrajectoryFeasibleFrame:
    """Adapt FeasibleSet / GoldenValues / TrajectoryFeasibleFrame into TrajectoryFeasibleFrame."""
    if isinstance(item, TrajectoryFeasibleFrame):
        return item
    if hasattr(item, "tau_particular") and hasattr(item, "null_basis_tau"):
        return TrajectoryFeasibleFrame(
            tau_particular=np.asarray(item.tau_particular, dtype=np.float64),
            null_basis_tau=np.asarray(item.null_basis_tau, dtype=np.float64),
            lambda_particular=getattr(item, "lambda_particular", None),
            null_basis_lambda=getattr(item, "null_basis_lambda", None),
            v_actuated=getattr(item, "v_actuated", None),
            J_c=getattr(item, "J_c", None),
            tau_lower=getattr(item, "tau_lower", None),
            tau_upper=getattr(item, "tau_upper", None),
            z_lower=getattr(item, "z_lower", None),
            z_upper=getattr(item, "z_upper", None),
            A_ub_z=getattr(item, "A_ub_z", None),
            b_ub_z=getattr(item, "b_ub_z", None),
            tau_capacity=getattr(item, "tau_capacity", None),
            hypotheses=tuple(getattr(item, "hypotheses", ("H-BOX-CAPACITY",))),
        )
    if hasattr(item, "x_particular") and hasattr(item, "null_basis"):
        xp = np.asarray(item.x_particular, dtype=np.float64)
        nb = np.asarray(item.null_basis, dtype=np.float64)
        raw_n_tau = getattr(item, "n_tau", None)
        n_tau = int(raw_n_tau) if raw_n_tau is not None else int(xp.shape[0])
        return TrajectoryFeasibleFrame(
            tau_particular=xp[:n_tau],
            null_basis_tau=nb[:n_tau, :],
            lambda_particular=xp[n_tau:] if xp.shape[0] > n_tau else None,
            null_basis_lambda=nb[n_tau:, :] if nb.shape[0] > n_tau else None,
            z_lower=getattr(item, "z_min", None),
            z_upper=getattr(item, "z_max", None),
            tau_lower=getattr(item, "tau_min", None),
            tau_upper=getattr(item, "tau_max", None),
        )
    raise TypeError(f"Unsupported feasible frame type: {type(item)!r}")


def verify_strict_convexity(
    objective: ObjectiveSpec,
    feasible_sets: Sequence[Any] | None = None,
    *,
    min_eigenvalue: float = CURVATURE_MIN_EIGENVALUE,
) -> None:
    """Verify strict reduced-space quadratic curvature (R6 DbC build-time precondition)."""
    if not isinstance(objective, ObjectiveSpec):
        raise TypeError("objective must be an ObjectiveSpec instance")

    base_quad = objective.effort_weight + objective.reference_weight
    if base_quad < min_eigenvalue:
        raise ValueError(
            "ObjectiveSpec is not strictly convex: missing positive quadratic curvature term "
            f"(require effort_weight > 0 or reference_weight > 0 so curvature >= {min_eigenvalue:.1e}; "
            f"got effort_weight={objective.effort_weight}, reference_weight={objective.reference_weight})"
        )

    if feasible_sets is None:
        return

    for idx, raw_frame in enumerate(feasible_sets):
        frame = _normalize_frame(raw_frame)
        n_tau = frame.n_tau()
        n_z = frame.n_z()
        if n_z == 0:
            continue
        cap = (
            objective.tau_max
            if objective.tau_max is not None
            else (
                frame.tau_capacity
                if frame.tau_capacity is not None
                else np.ones(n_tau, dtype=np.float64)
            )
        )
        inv_cap_sq = 1.0 / (np.asarray(cap, dtype=np.float64) ** 2)
        h_diag = 2.0 * (
            objective.effort_weight * inv_cap_sq + objective.reference_weight
        )
        nt = frame.null_basis_tau
        h_red = nt.T @ (h_diag[:, None] * nt)
        eigvals = np.linalg.eigvalsh(0.5 * (h_red + h_red.T))
        lam_min = float(np.min(eigvals))
        if lam_min < min_eigenvalue:
            raise ValueError(
                f"Reduced Hessian is not strictly convex at frame {idx}: "
                f"lambda_min(N^T H_quad N) = {lam_min:.3e} < {min_eigenvalue:.3e} "
                "(missing positive quadratic curvature term on null-space coordinates)"
            )


def _frame_inequalities(frame: TrajectoryFeasibleFrame) -> tuple[Array, Array]:
    """Assemble per-frame linear inequality rows G_t z_t <= h_t."""
    n_z = frame.n_z()
    rows_g: list[Array] = []
    rows_h: list[Array] = []
    eye_z = np.eye(n_z, dtype=np.float64)

    if frame.z_lower is not None:
        zl = np.asarray(frame.z_lower, dtype=np.float64)
        mask = np.isfinite(zl)
        if np.any(mask):
            rows_g.append(-eye_z[mask])
            rows_h.append(-zl[mask])
    if frame.z_upper is not None:
        zu = np.asarray(frame.z_upper, dtype=np.float64)
        mask = np.isfinite(zu)
        if np.any(mask):
            rows_g.append(eye_z[mask])
            rows_h.append(zu[mask])
    if frame.tau_lower is not None:
        tl = np.asarray(frame.tau_lower, dtype=np.float64)
        mask = np.isfinite(tl)
        if np.any(mask):
            rows_g.append(-frame.null_basis_tau[mask])
            rows_h.append(-(tl[mask] - frame.tau_particular[mask]))
    if frame.tau_upper is not None:
        tu = np.asarray(frame.tau_upper, dtype=np.float64)
        mask = np.isfinite(tu)
        if np.any(mask):
            rows_g.append(frame.null_basis_tau[mask])
            rows_h.append(tu[mask] - frame.tau_particular[mask])
    if frame.A_ub_z is not None and frame.b_ub_z is not None:
        a_ub = np.asarray(frame.A_ub_z, dtype=np.float64)
        b_ub = np.asarray(frame.b_ub_z, dtype=np.float64)
        if a_ub.shape[0] > 0:
            rows_g.append(a_ub)
            rows_h.append(b_ub)

    if not rows_g:
        return np.zeros((0, n_z), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.vstack(rows_g), np.concatenate(rows_h)


def _solve_newton_kkt(
    h_mat: Array,
    a_eq: Array,
    rhs_x: Array,
    rhs_y: Array,
) -> tuple[Array, Array]:
    """Solve the primal-dual Newton step for (dx, dy)."""
    n = h_mat.shape[0]
    p = a_eq.shape[0]
    if p == 0:
        try:
            cho = linalg.cho_factor(h_mat)
            x_step = np.asarray(linalg.cho_solve(cho, rhs_x), dtype=np.float64)
            return x_step, np.zeros(0, dtype=np.float64)
        except linalg.LinAlgError:
            x_step = np.asarray(linalg.solve(h_mat, rhs_x), dtype=np.float64)
            return x_step, np.zeros(0, dtype=np.float64)
    kkt = np.block([[h_mat, a_eq.T], [a_eq, -1e-12 * np.eye(p, dtype=np.float64)]])
    sol = np.asarray(
        linalg.solve(kkt, np.concatenate([rhs_x, rhs_y])), dtype=np.float64
    )
    return sol[:n], sol[n:]


def _normalize_inequality_rows(
    g_mat: Array,
    h_vec: Array,
) -> tuple[Array, Array, Array, float]:
    """Normalize rows of G x <= h by Euclidean row norm and drop zero-norm rows."""
    n = g_mat.shape[1]
    if g_mat.shape[0] == 0:
        return (
            np.zeros((0, n), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            0.0,
        )
    norms = np.linalg.norm(g_mat, axis=1)
    active = norms > 1e-12
    zero_viol = (
        float(np.max(np.maximum(0.0, -h_vec[~active]))) if np.any(~active) else 0.0
    )
    if not np.any(active):
        return (
            np.zeros((0, n), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            norms,
            zero_viol,
        )
    inv_norms = 1.0 / norms[active]
    return (
        g_mat[active] * inv_norms[:, np.newaxis],
        h_vec[active] * inv_norms,
        norms,
        zero_viol,
    )


def _solve_ldp_whitened(
    g_bar: Array,
    h_bar: Array,
) -> tuple[Array, Array, bool]:
    """Solve Lawson-Hanson (1974) Ch. 23 Least Distance Programming via NNLS."""
    from scipy import optimize

    m, n = g_bar.shape
    beta = max(1.0, float(np.max(np.abs(h_bar))))
    h_scaled = h_bar / beta
    e_ldp = np.vstack([-g_bar.T, -h_scaled[np.newaxis, :]])
    f_ldp = np.zeros(n + 1, dtype=np.float64)
    f_ldp[-1] = 1.0
    u_nnls_raw, _ = optimize.nnls(e_ldp, f_ldp, maxiter=max(10 * m, 2000))
    u_nnls = np.asarray(u_nnls_raw, dtype=np.float64)
    r_ldp = np.asarray(e_ldp @ u_nnls - f_ldp, dtype=np.float64)
    denom = float(-r_ldp[-1])
    if denom <= 1e-15:
        return (
            np.zeros(n, dtype=np.float64),
            np.zeros(m, dtype=np.float64),
            True,
        )
    y_opt = np.asarray(beta * (r_ldp[:-1] / denom), dtype=np.float64)
    mu_bar = np.asarray(beta * (u_nnls / denom), dtype=np.float64)
    return y_opt, mu_bar, False


def solve_conic_qp(
    p_mat: Array,
    q_vec: Array,
    g_mat: Array,
    h_vec: Array,
    a_eq: Array | None = None,
    b_eq: Array | None = None,
    x_init: Array | None = None,
) -> tuple[Array, Array, Array, float, float, float, bool]:
    """Solve strictly convex conic QP min 0.5 x^T P x + q^T x s.t. G x <= h via Lawson-Hanson LDP."""
    _ = x_init
    n, m_raw = p_mat.shape[0], g_mat.shape[0]
    a_mat = np.zeros((0, n), dtype=np.float64) if a_eq is None else a_eq
    b_vec = np.zeros(0, dtype=np.float64) if b_eq is None else b_eq
    p = a_mat.shape[0]
    z_m = np.zeros(m_raw, dtype=np.float64)
    z_0 = np.zeros(0, dtype=np.float64)

    if n == 0:
        viol0 = float(np.max(np.maximum(0.0, -h_vec))) if m_raw > 0 else 0.0
        return (
            z_0,
            z_m,
            np.zeros(p, dtype=np.float64),
            viol0,
            0.0,
            0.0,
            bool(viol0 <= 1e-5),
        )

    p_sym = 0.5 * (p_mat + p_mat.T) + 1e-11 * np.eye(n, dtype=np.float64)
    if m_raw == 0 or p > 0:
        x_sol, y_sol = _solve_newton_kkt(p_sym, a_mat, -q_vec, b_vec)
        p_res = float(np.max(np.abs(a_mat @ x_sol - b_vec))) if p > 0 else 0.0
        d_res = float(
            np.max(np.abs(p_mat @ x_sol + q_vec + (a_mat.T @ y_sol if p > 0 else 0.0)))
        )
        return x_sol, z_m, y_sol, p_res, d_res, 0.0, p_res <= 1e-5

    l_chol = np.asarray(linalg.cholesky(p_sym, lower=True), dtype=np.float64)
    x_unc = np.asarray(linalg.cho_solve((l_chol, True), -q_vec), dtype=np.float64)
    if np.all(g_mat @ x_unc <= h_vec + 1e-12):
        d_res = float(np.max(np.abs(p_mat @ x_unc + q_vec))) / (
            1.0 + float(np.max(np.abs(q_vec)))
        )
        return x_unc, z_m, z_0, 0.0, d_res, 0.0, True

    g_tilde = np.asarray(
        linalg.solve_triangular(l_chol, g_mat.T, lower=True), dtype=np.float64
    ).T
    g_bar, h_bar, row_norms, zero_viol = _normalize_inequality_rows(
        g_tilde, h_vec - g_mat @ x_unc
    )
    m = g_bar.shape[0]
    if m == 0:
        return x_unc, z_m, z_0, zero_viol, 0.0, 0.0, zero_viol <= 1e-5

    y_opt, mu_bar, infeasible = _solve_ldp_whitened(g_bar, h_bar)
    step = np.asarray(
        linalg.solve_triangular(l_chol.T, y_opt, lower=False), dtype=np.float64
    )
    x = x_unc + step
    duals_full = np.zeros(m_raw, dtype=np.float64)
    active_mask = row_norms > 1e-12
    duals_full[active_mask] = mu_bar / row_norms[active_mask]

    p_scale = 1.0 + float(np.max(np.abs(h_vec)))
    d_scale = 1.0 + float(np.max(np.abs(q_vec)))
    slack_bar = np.maximum(0.0, h_bar - g_bar @ y_opt)
    primal_viol = max(zero_viol, max(0.0, float(np.max(g_mat @ x - h_vec)))) / p_scale
    dual_res = float(np.max(np.abs(p_sym @ x + q_vec + g_mat.T @ duals_full))) / d_scale
    comp_res = float(np.abs(slack_bar @ mu_bar)) / (
        max(1, m) * (1.0 + 0.5 * float(y_opt @ y_opt))
    )
    converged = (not infeasible) and max(primal_viol, dual_res, comp_res) < 1e-4
    return x, duals_full, z_0, primal_viol, dual_res, comp_res, converged


def _build_trajectory_qp_matrices(
    frames: Sequence[TrajectoryFeasibleFrame],
    objective: ObjectiveSpec,
    reference: Array | None,
    dt: float,
    slices: Sequence[slice],
    total_z: int,
) -> tuple[Array, Array, Array, Array, bool]:
    """Assemble (P, q, G, h, has_power) for trajectory allocation with reduced epigraph."""
    t_count = len(frames)
    n_tau = frames[0].n_tau()
    has_power = (
        float(np.max(np.asarray(objective.c_plus))) > 0.0
        or float(np.max(np.asarray(objective.c_minus))) > 0.0
    ) and any(f.v_actuated is not None for f in frames)

    n_epi = t_count * n_tau if has_power else 0
    total_vars = total_z + n_epi
    p_mat = np.zeros((total_vars, total_vars), dtype=np.float64)
    q_vec = np.zeros(total_vars, dtype=np.float64)
    g_blocks: list[Array] = []
    h_blocks: list[Array] = []

    cp_vec = np.broadcast_to(np.asarray(objective.c_plus, dtype=np.float64), (n_tau,))
    cm_vec = np.broadcast_to(np.asarray(objective.c_minus, dtype=np.float64), (n_tau,))

    for t, frame in enumerate(frames):
        sl = slices[t]
        nt = frame.null_basis_tau
        tp = frame.tau_particular
        cap = (
            objective.tau_max
            if objective.tau_max is not None
            else (
                frame.tau_capacity
                if frame.tau_capacity is not None
                else np.ones(n_tau, dtype=np.float64)
            )
        )
        inv_cap_sq = 1.0 / (np.asarray(cap, dtype=np.float64) ** 2)
        if objective.effort_weight > 0.0:
            w_eff = 2.0 * objective.effort_weight * inv_cap_sq
            p_mat[sl, sl] += nt.T @ (w_eff[:, None] * nt)
            q_vec[sl] += nt.T @ (w_eff * tp)
        if objective.reference_weight > 0.0 and reference is not None:
            w_ref = 2.0 * objective.reference_weight
            p_mat[sl, sl] += w_ref * (nt.T @ nt)
            q_vec[sl] += w_ref * (nt.T @ (tp - reference[t]))

        g_t, h_t = _frame_inequalities(frame)
        if g_t.shape[0] > 0:
            g_row = np.zeros((g_t.shape[0], total_vars), dtype=np.float64)
            g_row[:, sl] = g_t
            g_blocks.append(g_row)
            h_blocks.append(h_t)

        if has_power:
            v_t = (
                np.zeros(n_tau, dtype=np.float64)
                if frame.v_actuated is None
                else np.asarray(frame.v_actuated, dtype=np.float64)
            )
            u_idx = total_z + t * n_tau + np.arange(n_tau)
            q_vec[u_idx] += cp_vec + cm_vec
            q_vec[sl] -= nt.T @ (cm_vec * v_t)
            p_mat[u_idx, u_idx] += 1e-5

            g_epi = np.zeros((2 * n_tau, total_vars), dtype=np.float64)
            g_epi[np.arange(n_tau), u_idx] = -1.0
            g_epi[n_tau + np.arange(n_tau), sl] = v_t[:, None] * nt
            g_epi[n_tau + np.arange(n_tau), u_idx] = -1.0
            h_epi = np.concatenate([np.zeros(n_tau, dtype=np.float64), -(v_t * tp)])
            g_blocks.append(g_epi)
            h_blocks.append(h_epi)

    if objective.rate_weight > 0.0 and t_count > 1:
        alpha = 2.0 * objective.rate_weight / (dt * dt)
        for t in range(1, t_count):
            sl_curr, sl_prev = slices[t], slices[t - 1]
            nt_curr, nt_prev = frames[t].null_basis_tau, frames[t - 1].null_basis_tau
            dtau_p = frames[t].tau_particular - frames[t - 1].tau_particular
            p_mat[sl_curr, sl_curr] += alpha * (nt_curr.T @ nt_curr)
            p_mat[sl_prev, sl_prev] += alpha * (nt_prev.T @ nt_prev)
            cross = alpha * (nt_curr.T @ nt_prev)
            p_mat[sl_curr, sl_prev] -= cross
            p_mat[sl_prev, sl_curr] -= cross.T
            q_vec[sl_curr] += alpha * (nt_curr.T @ dtau_p)
            q_vec[sl_prev] -= alpha * (nt_prev.T @ dtau_p)

    g_all = (
        np.vstack(g_blocks) if g_blocks else np.zeros((0, total_vars), dtype=np.float64)
    )
    h_all = np.concatenate(h_blocks) if h_blocks else np.zeros(0, dtype=np.float64)
    return p_mat, q_vec, g_all, h_all, has_power


def _unpack_trajectory_solution(
    frames: Sequence[TrajectoryFeasibleFrame],
    slices: Sequence[slice],
    z_sol: Array,
    n_tau: int,
) -> tuple[Array, Array, list[Array], list[Array]]:
    """Recover per-frame (tau, lambda_c, power_list, jc_v_list) from null-space coordinates z."""
    tau_list: list[Array] = []
    lam_list: list[Array] = []
    power_list: list[Array] = []
    jc_v_list: list[Array] = []
    for t, frame in enumerate(frames):
        z_t = z_sol[slices[t]]
        tau_t = frame.tau_particular + frame.null_basis_tau @ z_t
        tau_list.append(tau_t)
        if frame.lambda_particular is not None and frame.null_basis_lambda is not None:
            lam_list.append(frame.lambda_particular + frame.null_basis_lambda @ z_t)
        else:
            lam_list.append(np.zeros(0, dtype=np.float64))
        v_t = (
            np.zeros(n_tau, dtype=np.float64)
            if frame.v_actuated is None
            else frame.v_actuated
        )
        power_list.append(tau_t * v_t)
        if frame.J_c is not None and frame.v_actuated is not None:
            jc_v_list.append(frame.J_c @ frame.v_actuated)
    tau_arr = _readonly_array(np.stack(tau_list, axis=0), "tau")
    lam_arr = _readonly_array(np.stack(lam_list, axis=0), "lambda_c")
    return tau_arr, lam_arr, power_list, jc_v_list


def allocate_trajectory(
    feasible_sets: Sequence[Any],
    objective: ObjectiveSpec,
    *,
    reference: Array | None = None,
    dt: float = 0.01,
    initial_z: Array | None = None,
) -> AllocationResult:
    """Solve the strictly convex prior-conditioned trajectory allocation problem (NSX-23)."""
    if len(feasible_sets) == 0:
        raise ValueError("feasible_sets must be non-empty")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")

    frames = [_normalize_frame(f) for f in feasible_sets]
    verify_strict_convexity(objective, frames)

    t_count = len(frames)
    n_tau = frames[0].n_tau()
    ref_arr: Array | None = None
    if reference is not None:
        ref_arr = _readonly_array(reference, "reference")
        if ref_arr.ndim == 1:
            ref_arr = ref_arr.reshape(1, -1)
        if ref_arr.shape != (t_count, n_tau):
            raise ValueError(
                f"reference shape {ref_arr.shape} must match ({t_count}, {n_tau})"
            )
    elif objective.reference_weight > 0.0:
        ref_arr = np.zeros((t_count, n_tau), dtype=np.float64)

    dims = [f.n_z() for f in frames]
    offsets = np.cumsum([0, *dims])
    slices = [slice(int(offsets[t]), int(offsets[t + 1])) for t in range(t_count)]
    total_z = int(offsets[-1])

    t0 = time.perf_counter()
    p_mat, q_vec, g_mat, h_vec, _has_power = _build_trajectory_qp_matrices(
        frames, objective, ref_arr, dt, slices, total_z
    )
    _ = initial_z
    x_sol, duals, eq_duals, p_res, d_res, c_res, converged = solve_conic_qp(
        p_mat, q_vec, g_mat, h_vec
    )
    solve_time_s = float(time.perf_counter() - t0)

    z_sol = x_sol[:total_z]
    tau_arr, lam_arr, power_list, jc_v_list = _unpack_trajectory_solution(
        frames, slices, z_sol, n_tau
    )
    z_out = (
        _readonly_array(z_sol.reshape(t_count, dims[0]), "z")
        if len(set(dims)) == 1
        else _readonly_array(z_sol, "z")
    )
    gp_info = decompose_gross_power_cost(
        np.concatenate(power_list),
        c_plus=objective.c_plus,
        c_minus=objective.c_minus,
        j_c_v=np.concatenate(jc_v_list) if jc_v_list else None,
    )
    kkt_norm = float(max(p_res, d_res, c_res))
    obj_val = float(0.5 * (x_sol @ (p_mat @ x_sol)) + q_vec @ x_sol)
    frame_hyps = tuple(dict.fromkeys([h for f in frames for h in f.hypotheses]))
    all_hyps = tuple(dict.fromkeys([*objective.hypotheses, *frame_hyps]))

    return AllocationResult(
        tau=tau_arr,
        lambda_c=lam_arr,
        z=z_out,
        duals=_readonly_array(duals, "duals"),
        equality_duals=_readonly_array(eq_duals, "equality_duals"),
        kkt_residual_norm=kkt_norm,
        primal_residual=p_res,
        dual_residual=d_res,
        complementarity_residual=c_res,
        objective_value=obj_val,
        solve_time_s=solve_time_s,
        converged=converged,
        gross_power_info=gp_info,
        hypotheses=all_hyps or (DEFAULT_HYPOTHESIS_ID,),
        conventions=objective.conventions,
    )


def solve_equality_reduced_qp(
    a_eq: Array,
    b_eq: Array,
    h_diag: Array,
    lb_x: Array,
    ub_x: Array,
    c_ineq: Array,
) -> tuple[Array, AllocationResult]:
    """Reduce A_eq x = b_eq (eliminating fixed variables) into FeasibleSet and solve."""
    lb = np.asarray(lb_x, dtype=np.float64)
    ub = np.asarray(ub_x, dtype=np.float64)
    fixed_mask = np.isfinite(lb) & np.isfinite(ub) & (np.abs(ub - lb) <= 1e-12)
    free_mask = ~fixed_mask

    x_fixed = lb[fixed_mask]
    b_rem = np.asarray(b_eq, dtype=np.float64) - a_eq[:, fixed_mask] @ x_fixed
    a_free = np.asarray(a_eq[:, free_mask], dtype=np.float64)
    h_free = np.asarray(h_diag[free_mask], dtype=np.float64)

    d_vec = np.sqrt(h_free)
    d_inv = 1.0 / d_vec
    a_scaled = a_free * d_inv[np.newaxis, :]

    u, s_vals, vh = linalg.svd(a_scaled, full_matrices=True)
    tol = (
        max(a_scaled.shape)
        * np.finfo(np.float64).eps
        * (s_vals[0] if s_vals.size else 1.0)
    )
    rank = int(np.sum(s_vals > tol))

    if rank > 0:
        ut_b = u[:, :rank].T @ b_rem
        y_p = vh[:rank, :].T @ (ut_b / s_vals[:rank])
    else:
        y_p = np.zeros(a_scaled.shape[1], dtype=np.float64)
    n_y = vh[rank:, :].T

    x_p_free = y_p * d_inv
    n_x_free = n_y * d_inv[:, np.newaxis]

    a_ub_z: Array | None = None
    b_ub_z: Array | None = None
    if c_ineq.shape[0] > 0:
        c_free = c_ineq[:, free_mask]
        c_rhs = c_ineq[:, fixed_mask] @ x_fixed
        a_ub_z = -c_free @ n_x_free
        b_ub_z = c_rhs + c_free @ x_p_free

    frame = TrajectoryFeasibleFrame(
        tau_particular=x_p_free,
        null_basis_tau=n_x_free,
        tau_lower=lb[free_mask],
        tau_upper=ub[free_mask],
        A_ub_z=a_ub_z,
        b_ub_z=b_ub_z,
        tau_capacity=d_inv,
        hypotheses=("H-CONTACT-FRICTION-PYRAMID", "H-ACTUATOR-BOUNDS"),
    )
    obj = ObjectiveSpec(
        effort_weight=0.5,
        hypotheses=("H-ALLOC-CONVEX-PRIOR", "H-CONTACT-FRICTION-PYRAMID"),
    )
    res = allocate_trajectory([frame], obj)
    x_full = np.zeros(lb.size, dtype=np.float64)
    x_full[fixed_mask] = x_fixed
    x_full[free_mask] = res.tau[0]
    return x_full, res


__all__ = [
    "CURVATURE_MIN_EIGENVALUE",
    "AllocationQuantityRecord",
    "AllocationResult",
    "FeasibleSet",
    "GrossPowerDecomposition",
    "ObjectiveSpec",
    "TrajectoryFeasibleFrame",
    "ZeroSideCheckResult",
    "ZeroSideConsistency",
    "allocate_trajectory",
    "check_zero_side_consistency",
    "decompose_gross_power_cost",
    "solve_conic_qp",
    "solve_equality_reduced_qp",
    "verify_strict_convexity",
]
