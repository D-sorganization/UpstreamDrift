"""Lower-body two-foot unilateral contact loop with optional GRF observations (NSX-60, #12585).

Extends the closed-loop inverse-dynamics balance equation to include the active
two-foot ground contact Jacobian ``J_g``:

    M a + h = S^T tau + J_c^T lambda_c + J_g^T f_g

When ground reaction forces (GRF) are unobserved (``grf_observed is None``),
``f_g`` remains in the decision vector subject to unilateral normal-force and
Coulomb friction-cone bounds. When ``grf_observed`` is supplied with zero
uncertainty, ``J_g^T f_{g,obs}`` is substituted into the effective right-hand
side ``b_eff = M a + h - J_g^T f_{g,obs}``, reducing the ground-wrench nullity
by ``rank(J_g^T)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from src.shared.python.core.contracts import ensure, require
from src.shared.python.motion_matching.force_nullspace import ForceNullSpace

FloatArray = NDArray[np.float64]
_SIGN_TOL = 1e-9
_GOLDEN_PATH = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "research"
    / "loop_nullspace"
    / "derivations"
    / "golden_fixtures.json"
)


def readonly_array(arr: Any, name: str, ndim: int | None = None) -> FloatArray:
    """Return a validated finite read-only float64 array."""
    out = np.array(arr, dtype=np.float64, copy=True)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite entries")
    out.setflags(write=False)
    return out


def classify_sign(lower: float, upper: float, tol: float = _SIGN_TOL) -> str:
    """Classify an interval [lower, upper] into POSITIVE, NEGATIVE, ZERO, or INDETERMINATE."""
    if abs(lower) <= tol and abs(upper) <= tol:
        return "ZERO"
    if lower > tol:
        return "POSITIVE"
    if upper < -tol:
        return "NEGATIVE"
    return "INDETERMINATE"


@dataclass(frozen=True)
class LowerBodySnapshot:
    """Lightweight closed-loop snapshot carrying hand closure and foot contact Jacobians."""

    q: FloatArray
    v: FloatArray
    a: FloatArray
    M: FloatArray
    h: FloatArray
    S: FloatArray
    J_c: FloatArray
    Jdot_v: FloatArray
    J_g: FloatArray
    actuated_indices: tuple[int, ...]
    unactuated_indices: tuple[int, ...]
    grf_observed: FloatArray | None = None

    def __post_init__(self) -> None:
        for field_name in ("q", "v", "a", "h", "Jdot_v"):
            object.__setattr__(
                self,
                field_name,
                readonly_array(getattr(self, field_name), field_name, 1),
            )
        for field_name in ("M", "S", "J_c", "J_g"):
            object.__setattr__(
                self,
                field_name,
                readonly_array(getattr(self, field_name), field_name, 2),
            )
        if self.grf_observed is not None:
            object.__setattr__(
                self,
                "grf_observed",
                readonly_array(self.grf_observed, "grf_observed", 1),
            )

    def generalized_load(self) -> FloatArray:
        """Return generalized inverse-dynamics load b = M a + h."""
        return readonly_array(self.M @ self.a + self.h, "b", 1)


@dataclass(frozen=True)
class LowerBodyLoopResult:
    """Decomposition and static envelope comparison for the lower-body + hand loop."""

    equality_nullity: int
    free_equality_nullity: int
    observed_equality_nullity: int
    nullity_reduction: int
    ground_wrench_nullity_reduction: int
    rank_A: int
    rank_Jg: int
    rank_Jg_u: int
    dimensions: dict[str, int]
    particular: FloatArray
    null_basis: FloatArray
    effective_rhs: FloatArray
    torque_lower: FloatArray
    torque_upper: FloatArray
    torque_sign_class: tuple[str, ...]
    hand_split_lower: FloatArray
    hand_split_upper: FloatArray
    hand_split_sign_class: tuple[str, ...]
    changed_torque_signs: tuple[int, ...]
    changed_hand_split_signs: tuple[int, ...]
    grf_observed_used: bool
    grf_uncertainty_sigma: float


def _extract_indices(
    snapshot: Any, n_v: int, S: FloatArray
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    topo = getattr(snapshot, "topology", None)
    if topo is not None and hasattr(topo, "actuated_indices"):
        return tuple(int(i) for i in topo.actuated_indices), tuple(
            int(i) for i in topo.unactuated_indices
        )
    if hasattr(snapshot, "actuated_indices") and hasattr(
        snapshot, "unactuated_indices"
    ):
        return tuple(int(i) for i in snapshot.actuated_indices), tuple(
            int(i) for i in snapshot.unactuated_indices
        )
    act = tuple(int(np.argmax(np.abs(row))) for row in S)
    unact = tuple(i for i in range(n_v) if i not in set(act))
    return act, unact


def _build_ground_inequalities(
    n_tau: int,
    m_c: int,
    m_g: int,
    mu_ground: float,
    torque_bound: float,
    grf_box: tuple[FloatArray, FloatArray] | None,
) -> tuple[FloatArray, FloatArray]:
    """Assemble linear inequalities G_x x <= h_x on x = [tau, lambda_c, f_g]."""
    dim_x = n_tau + m_c + m_g
    rows: list[FloatArray] = []
    rhs: list[float] = []

    for i in range(n_tau):
        row_pos = np.zeros(dim_x, dtype=np.float64)
        row_pos[i] = 1.0
        rows.extend([row_pos, -row_pos])
        rhs.extend([torque_bound, torque_bound])

    for j in range(m_c):
        row_lam = np.zeros(dim_x, dtype=np.float64)
        row_lam[n_tau + j] = 1.0
        rows.extend([row_lam, -row_lam])
        rhs.extend([torque_bound * 5.0, torque_bound * 5.0])

    if m_g > 0:
        f_offset = n_tau + m_c
        stride = 2 if m_g % 2 == 0 and m_g % 3 != 0 else (3 if m_g % 3 == 0 else 1)
        for start in range(0, m_g, stride):
            normal_idx = f_offset + start + (stride - 1)
            row_unilat = np.zeros(dim_x, dtype=np.float64)
            row_unilat[normal_idx] = -1.0
            rows.append(row_unilat)
            rhs.append(0.0)
            row_cap = np.zeros(dim_x, dtype=np.float64)
            row_cap[normal_idx] = 1.0
            rows.append(row_cap)
            rhs.append(torque_bound * 10.0)
            for t_pos in range(start, start + stride - 1):
                t_idx = f_offset + t_pos
                for sign in (1.0, -1.0):
                    row_cone = np.zeros(dim_x, dtype=np.float64)
                    row_cone[t_idx] = sign
                    row_cone[normal_idx] = -mu_ground
                    rows.append(row_cone)
                    rhs.append(0.0)

        if grf_box is not None:
            f_lo, f_hi = grf_box
            for k in range(m_g):
                row_k = np.zeros(dim_x, dtype=np.float64)
                row_k[f_offset + k] = 1.0
                rows.extend([row_k, -row_k])
                rhs.extend([float(f_hi[k]), float(-f_lo[k])])

    return np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64)


def solve_nullspace_envelopes(
    particular: FloatArray,
    basis: FloatArray,
    g_x: FloatArray,
    h_x: FloatArray,
    n_quantities: int,
) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    """Minimize and maximize the first n_quantities components of x(z) = x_p + N z."""
    if basis.shape[1] == 0:
        vals = readonly_array(particular[:n_quantities], "particular_slice", 1)
        signs = tuple(classify_sign(float(v), float(v)) for v in vals)
        return vals, vals, signs

    a_ub = g_x @ basis
    b_ub = h_x - g_x @ particular
    d = basis.shape[1]
    bounds = [(-1e4, 1e4)] * d
    lower = np.zeros(n_quantities, dtype=np.float64)
    upper = np.zeros(n_quantities, dtype=np.float64)

    for idx in range(n_quantities):
        c_z = basis[idx, :]
        res_min = linprog(c_z, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        res_max = linprog(-c_z, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        p_i = float(particular[idx])
        lower[idx] = p_i + float(res_min.fun) if res_min.success else p_i
        upper[idx] = p_i - float(res_max.fun) if res_max.success else p_i

    signs = tuple(
        classify_sign(float(lower[i]), float(upper[i])) for i in range(n_quantities)
    )
    return (
        readonly_array(lower, "envelope_lower", 1),
        readonly_array(upper, "envelope_upper", 1),
        signs,
    )


def _compute_dimensions(
    S: FloatArray,
    J_c: FloatArray,
    J_g: FloatArray | None,
    unactuated: tuple[int, ...],
    space: ForceNullSpace,
) -> dict[str, int]:
    n_tau, n_v = S.shape
    m_c = int(J_c.shape[0])
    m_g = 0 if J_g is None else int(J_g.shape[0])
    rank_jc = int(np.linalg.matrix_rank(J_c))
    rank_jcu = int(np.linalg.matrix_rank(J_c[:, list(unactuated)])) if unactuated else 0
    nullity = int(space.basis.shape[1])
    return {
        "n_v": n_v,
        "n_tau": n_tau,
        "n_u": len(unactuated),
        "m_c": m_c,
        "m_g": m_g,
        "dim_x": int(space.matrix.shape[1]),
        "rank_A": int(space.rank),
        "equality_nullity": nullity,
        "multiplier_gauge": m_c - rank_jc,
        "rank_Jc": rank_jc,
        "rank_Jc_u": rank_jcu,
        "attainable_torque_variation": rank_jc - rank_jcu,
        "feasible_set_dim": nullity,
    }


def analyze_lower_body_loop(
    snapshot: Any,
    *,
    grf_observed: np.ndarray | None = None,
    grf_uncertainty_sigma: float = 0.0,
    mu_ground: float = 0.6,
    torque_bound: float = 200.0,
) -> LowerBodyLoopResult:
    """Analyze the coupled hand + two-foot ground contact closed loop."""
    require(mu_ground > 0.0 and np.isfinite(mu_ground), "mu_ground must be > 0")
    require(
        torque_bound > 0.0 and np.isfinite(torque_bound), "torque_bound must be > 0"
    )
    require(
        grf_uncertainty_sigma >= 0.0 and np.isfinite(grf_uncertainty_sigma),
        "grf_uncertainty_sigma must be >= 0",
    )
    j_g_raw = getattr(snapshot, "J_g", None)
    if j_g_raw is None:
        raise ValueError("snapshot.J_g is required for lower-body loop analysis")

    S = readonly_array(snapshot.S, "S", 2)
    J_c = readonly_array(snapshot.J_c, "J_c", 2)
    J_g = readonly_array(j_g_raw, "J_g", 2)
    b = readonly_array(snapshot.generalized_load(), "b", 1)
    n_tau, n_v = S.shape
    m_c, m_g = int(J_c.shape[0]), int(J_g.shape[0])
    _act, unact = _extract_indices(snapshot, n_v, S)

    a_free = np.hstack([S.T, J_c.T, J_g.T])
    space_free = ForceNullSpace.from_balance(a_free, b)
    g_free, h_free = _build_ground_inequalities(
        n_tau, m_c, m_g, mu_ground, torque_bound, grf_box=None
    )
    lo_free, hi_free, signs_free = solve_nullspace_envelopes(
        space_free.particular, space_free.basis, g_free, h_free, n_tau + m_c
    )

    rank_jg = int(np.linalg.matrix_rank(J_g.T))
    rank_jgu = int(np.linalg.matrix_rank(J_g[:, list(unact)].T)) if unact else 0
    a_obs_mat = np.hstack([S.T, J_c.T])
    obs_nullity = int((n_tau + m_c) - np.linalg.matrix_rank(a_obs_mat))

    if grf_observed is None:
        dims = _compute_dimensions(S, J_c, J_g, unact, space_free)
        return LowerBodyLoopResult(
            equality_nullity=int(space_free.basis.shape[1]),
            free_equality_nullity=int(space_free.basis.shape[1]),
            observed_equality_nullity=obs_nullity,
            nullity_reduction=0,
            ground_wrench_nullity_reduction=0,
            rank_A=int(space_free.rank),
            rank_Jg=rank_jg,
            rank_Jg_u=rank_jgu,
            dimensions=dims,
            particular=space_free.particular,
            null_basis=space_free.basis,
            effective_rhs=b,
            torque_lower=lo_free[:n_tau],
            torque_upper=hi_free[:n_tau],
            torque_sign_class=signs_free[:n_tau],
            hand_split_lower=lo_free[n_tau : n_tau + m_c],
            hand_split_upper=hi_free[n_tau : n_tau + m_c],
            hand_split_sign_class=signs_free[n_tau : n_tau + m_c],
            changed_torque_signs=(),
            changed_hand_split_signs=(),
            grf_observed_used=False,
            grf_uncertainty_sigma=float(grf_uncertainty_sigma),
        )

    f_obs = readonly_array(grf_observed, "grf_observed", 1)
    if f_obs.shape != (m_g,):
        raise ValueError(f"grf_observed length {f_obs.shape[0]} must match m_g ({m_g})")
    b_eff = readonly_array(b - J_g.T @ f_obs, "b_eff", 1)

    if grf_uncertainty_sigma == 0.0:
        space_active = ForceNullSpace.from_balance(a_obs_mat, b_eff)
        g_act, h_act = _build_ground_inequalities(
            n_tau, m_c, 0, mu_ground, torque_bound, grf_box=None
        )
        lo_act, hi_act, signs_act = solve_nullspace_envelopes(
            space_active.particular, space_active.basis, g_act, h_act, n_tau + m_c
        )
        dims = _compute_dimensions(S, J_c, None, unact, space_active)
    else:
        space_active = space_free
        sigma_vec = np.full(m_g, float(grf_uncertainty_sigma), dtype=np.float64)
        g_act, h_act = _build_ground_inequalities(
            n_tau,
            m_c,
            m_g,
            mu_ground,
            torque_bound,
            grf_box=(f_obs - sigma_vec, f_obs + sigma_vec),
        )
        lo_act, hi_act, signs_act = solve_nullspace_envelopes(
            space_active.particular, space_active.basis, g_act, h_act, n_tau + m_c
        )
        dims = _compute_dimensions(S, J_c, J_g, unact, space_active)

    free_null = int(space_free.basis.shape[1])
    act_null = int(space_active.basis.shape[1])
    changed_tau = tuple(i for i in range(n_tau) if signs_act[i] != signs_free[i])
    changed_split = tuple(
        j for j in range(m_c) if signs_act[n_tau + j] != signs_free[n_tau + j]
    )
    ensure(act_null <= free_null, "observing GRF cannot increase equality nullity")
    return LowerBodyLoopResult(
        equality_nullity=act_null,
        free_equality_nullity=free_null,
        observed_equality_nullity=obs_nullity,
        nullity_reduction=free_null - act_null,
        ground_wrench_nullity_reduction=rank_jg if grf_uncertainty_sigma == 0.0 else 0,
        rank_A=int(space_active.rank),
        rank_Jg=rank_jg,
        rank_Jg_u=rank_jgu,
        dimensions=dims,
        particular=space_active.particular,
        null_basis=space_active.basis,
        effective_rhs=b_eff,
        torque_lower=lo_act[:n_tau],
        torque_upper=hi_act[:n_tau],
        torque_sign_class=signs_act[:n_tau],
        hand_split_lower=lo_act[n_tau : n_tau + m_c],
        hand_split_upper=hi_act[n_tau : n_tau + m_c],
        hand_split_sign_class=signs_act[n_tau : n_tau + m_c],
        changed_torque_signs=changed_tau,
        changed_hand_split_signs=changed_split,
        grf_observed_used=True,
        grf_uncertainty_sigma=float(grf_uncertainty_sigma),
    )


def include_ground_loop(
    snapshot: Any,
    *,
    use_observed_grf: bool = False,
) -> LowerBodyLoopResult:
    """Contract-appendix entry point incorporating J_g with optional observed GRF."""
    obs = getattr(snapshot, "grf_observed", None) if use_observed_grf else None
    if use_observed_grf and obs is None:
        raise ValueError("use_observed_grf=True requires snapshot.grf_observed")
    return analyze_lower_body_loop(snapshot, grf_observed=obs)


def floating_base_two_foot_snapshot() -> tuple[LowerBodySnapshot, dict[str, int]]:
    """Load the golden 7-DOF floating-base two-foot + two-arm closed-loop fixture."""
    payload = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    raw = payload["fixtures"]["floating_base_two_arm"]
    act = tuple(int(i) for i in raw["actuated_indices"])
    unact = tuple(int(i) for i in raw["unactuated_indices"])
    n_v = int(raw["dimensions"]["n_v"])
    b = np.asarray(raw["b"], dtype=np.float64)
    J_c = np.asarray(raw["J_c"], dtype=np.float64)
    J_g = np.asarray(raw["J_g"], dtype=np.float64)
    x_p = np.asarray(raw["x_particular"], dtype=np.float64)
    f_g_obs = x_p[len(act) + J_c.shape[0] :]
    S = np.eye(n_v, dtype=np.float64)[list(act)]
    snap = LowerBodySnapshot(
        q=np.zeros(n_v, dtype=np.float64),
        v=np.zeros(n_v, dtype=np.float64),
        a=b.copy(),
        M=np.eye(n_v, dtype=np.float64),
        h=np.zeros(n_v, dtype=np.float64),
        S=S,
        J_c=J_c,
        Jdot_v=np.zeros(J_c.shape[0], dtype=np.float64),
        J_g=J_g,
        actuated_indices=act,
        unactuated_indices=unact,
        grf_observed=f_g_obs,
    )
    dims = {str(k): int(v) for k, v in raw["dimensions"].items()}
    return snap, dims


def actuated_lower_body_two_foot_snapshot() -> tuple[LowerBodySnapshot, FloatArray]:
    """Return a 7-DOF actuated lower-body + two-arm snapshot with full-row-rank J_g (rank 3)."""
    n_v = 7
    act = tuple(range(n_v))
    S = np.eye(n_v, dtype=np.float64)
    J_c = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.5, -1.0, -0.5],
            [0.0, 0.0, 0.0, 0.0, 0.75, 0.0, -0.75],
        ],
        dtype=np.float64,
    )
    J_g = np.array(
        [
            [1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    f_g_obs = np.array([15.0, 20.0, 80.0], dtype=np.float64)
    b_leg = J_g[:, :3].T @ f_g_obs + np.array([25.0, 30.0, -15.0], dtype=np.float64)
    b_arm = np.array([2.0, -1.0, 1.0, -0.5], dtype=np.float64)
    b = np.concatenate([b_leg, b_arm])
    snap = LowerBodySnapshot(
        q=np.zeros(n_v, dtype=np.float64),
        v=np.zeros(n_v, dtype=np.float64),
        a=b.copy(),
        M=np.eye(n_v, dtype=np.float64),
        h=np.zeros(n_v, dtype=np.float64),
        S=S,
        J_c=J_c,
        Jdot_v=np.zeros(2, dtype=np.float64),
        J_g=J_g,
        actuated_indices=act,
        unactuated_indices=(),
        grf_observed=f_g_obs,
    )
    return snap, readonly_array(f_g_obs, "f_g_obs", 1)


__all__ = [
    "LowerBodyLoopResult",
    "LowerBodySnapshot",
    "actuated_lower_body_two_foot_snapshot",
    "analyze_lower_body_loop",
    "classify_sign",
    "floating_base_two_foot_snapshot",
    "include_ground_loop",
    "readonly_array",
    "solve_nullspace_envelopes",
]
