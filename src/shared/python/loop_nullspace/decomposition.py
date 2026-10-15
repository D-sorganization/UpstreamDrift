"""Closed-loop ForceNullSpace decomposition and zero-side three-case solve (NSX-15, #12560).

Computes the particular solution and null space of A x = b with rank-revealing
per-frame diagnostics (R2), three-case zero-side solve (R3), and metric-declared
minimum-norm torque projection without 'Gauss's principle' misnomers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.contracts import (
    AllocationDependence,
    ClosedLoopSnapshot,
    FloatArray,
    LimbSide,
    QuantityRecord,
    _readonly_float_array,
    _validate_nonempty_str,
)
from src.shared.python.motion_matching.force_nullspace import ForceNullSpace

ZeroSideCase: TypeAlias = Literal[
    "CONSISTENT_UNIQUE",
    "CONSISTENT_NONUNIQUE",
    "INCONSISTENT",
]


@dataclass(frozen=True)
class AllocationMetric:
    """Declared positive-definite quadratic metric W for torque selection.

    Enforces Rule R2/R3: every metric must declare a canonical convention ID
    (such as 'C-METRIC-IDENTITY', 'C-METRIC-CAPACITY', or 'C-METRIC-MINV') and
    must never be labelled 'Gauss' or 'Gauss's principle'.
    """

    convention_id: str
    W: FloatArray

    def __post_init__(self) -> None:
        conv = _validate_nonempty_str(self.convention_id, "convention_id")
        if "gauss" in conv.lower():
            raise ValueError(
                f"Gauss's principle nomenclature is forbidden in convention_id: {conv!r}"
            )
        object.__setattr__(self, "convention_id", conv)

        w_arr = _readonly_float_array(self.W, "W", ndim=2)
        if w_arr.shape[0] == 0 or w_arr.shape[0] != w_arr.shape[1]:
            raise ValueError(f"W must be a non-empty square matrix, got {w_arr.shape}")
        if not np.allclose(w_arr, w_arr.T, atol=1e-10):
            raise ValueError("W must be symmetric")
        sym_w = 0.5 * (w_arr + w_arr.T)
        try:
            np.linalg.cholesky(sym_w)
        except np.linalg.LinAlgError as exc:
            raise ValueError("W must be symmetric positive definite") from exc
        object.__setattr__(self, "W", w_arr)


@dataclass(frozen=True)
class Decomposition:
    """Rank-revealing equality decomposition x = x_particular + null_basis @ z of A x = b."""

    matrix_A: FloatArray
    rhs_b: FloatArray
    x_particular: FloatArray
    null_basis: FloatArray
    rank_A: int
    nullity: int
    multiplier_gauge: int
    attainable_torque_variation: int
    scaled_singular_values: FloatArray
    condition_number: float
    dimensions: dict[str, int]
    x_particular_record: QuantityRecord
    singular_values_record: QuantityRecord
    condition_number_record: QuantityRecord

    @property
    def equality_nullity(self) -> int:
        """Alias for nullity matching NSX-00 Table 1."""
        return self.nullity

    def to_dict(self) -> dict[str, Any]:
        """Serialize per-frame decomposition diagnostics to a JSON-compatible dictionary."""
        return {
            "x_particular": self.x_particular.tolist(),
            "null_basis": self.null_basis.tolist(),
            "rank_A": int(self.rank_A),
            "nullity": int(self.nullity),
            "multiplier_gauge": int(self.multiplier_gauge),
            "attainable_torque_variation": int(self.attainable_torque_variation),
            "scaled_singular_values": self.scaled_singular_values.tolist(),
            "condition_number": float(self.condition_number),
            "dimensions": dict(self.dimensions),
        }


@dataclass(frozen=True)
class ZeroSideResult:
    """Three-case classification and diagnostics for zero-side solve J_{c,side}^T lambda = b_side."""

    case: ZeroSideCase
    side_indices: tuple[int, ...]
    nullity: int
    residual_norm: float
    residual_scaled: float
    residual_vector: FloatArray
    damping: float
    lambda_solution: FloatArray | None
    lambda_least_squares: FloatArray
    kernel_basis: FloatArray
    residual_record: QuantityRecord

    @property
    def status(self) -> ZeroSideCase:
        """Alias for case matching NSX-00 golden keys."""
        return self.case

    @property
    def kernel_dim(self) -> int:
        """Alias for nullity matching NSX-00 golden keys."""
        return self.nullity


def _assemble_balance_matrix(snapshot: ClosedLoopSnapshot) -> FloatArray:
    """Assemble A = [S^T, J_c^T, (J_g^T)] from a ClosedLoopSnapshot."""
    blocks: list[np.ndarray] = [snapshot.S.T, snapshot.J_c.T]
    if snapshot.J_g is not None:
        blocks.append(snapshot.J_g.T)
    return _readonly_float_array(np.hstack(blocks), "A", ndim=2)


def _compute_dimension_table(
    snapshot: ClosedLoopSnapshot,
    matrix_A: FloatArray,
    rank_A: int,
    nullity: int,
    rcond: float,
) -> tuple[int, int, dict[str, int]]:
    """Compute NSX-00 Table 1 dimension counts for a ClosedLoopSnapshot."""
    n_v = snapshot.nv()
    n_tau = int(snapshot.S.shape[0])
    n_u = snapshot.nu()
    m_c = snapshot.nc()
    m_g = 0 if snapshot.J_g is None else int(snapshot.J_g.shape[0])
    dim_x = int(matrix_A.shape[1])

    rank_jc = int(np.linalg.matrix_rank(snapshot.J_c, tol=rcond))
    unactuated = snapshot.unactuated_rows()
    rank_jcu = (
        int(np.linalg.matrix_rank(snapshot.J_c[:, unactuated], tol=rcond))
        if unactuated.size > 0
        else 0
    )
    multiplier_gauge = int(m_c - rank_jc)
    attainable_var = int(rank_jc - rank_jcu)
    dims = {
        "n_v": n_v,
        "n_tau": n_tau,
        "n_u": n_u,
        "m_c": m_c,
        "m_g": m_g,
        "dim_x": dim_x,
        "rank_A": int(rank_A),
        "equality_nullity": int(nullity),
        "multiplier_gauge": multiplier_gauge,
        "rank_Jc": rank_jc,
        "rank_Jc_u": rank_jcu,
        "attainable_torque_variation": attainable_var,
        "feasible_set_dim": int(nullity),
    }
    return multiplier_gauge, attainable_var, dims


def decompose(
    snapshot: ClosedLoopSnapshot,
    *,
    weights: AllocationMetric | None = None,
    metric_convention: str = "C-METRIC-IDENTITY",
    rcond: float = 1e-10,
) -> Decomposition:
    """Compute particular solution, null basis, and rank diagnostics on A x = b.

    Reuses ForceNullSpace.from_balance on A = [S^T, J_c^T, (J_g^T)].
    """
    if not isinstance(snapshot, ClosedLoopSnapshot):
        raise TypeError("snapshot must be a ClosedLoopSnapshot instance")
    if not np.isfinite(rcond) or not (0.0 < rcond < 1.0):
        raise ValueError(f"rcond must be in (0, 1), got {rcond}")

    conv_id = weights.convention_id if weights is not None else metric_convention
    if "gauss" in conv_id.lower():
        raise ValueError(f"Forbidden Gauss nomenclature in convention: {conv_id!r}")

    matrix_A = _assemble_balance_matrix(snapshot)
    rhs_b = snapshot.generalized_load()
    space = ForceNullSpace.from_balance(
        matrix_A,
        rhs_b,
        rank_rtol=rcond,
        balance_tolerance=max(rcond * 1e3, 1e-7),
    )
    analysis = space.analyze()
    x_p = space.particular
    null_basis = space.basis

    if weights is not None and null_basis.shape[1] > 0:
        n_tau = int(snapshot.S.shape[0])
        if weights.W.shape == (n_tau, n_tau):
            n_tau_basis = null_basis[:n_tau, :]
            h_mat = n_tau_basis.T @ weights.W @ n_tau_basis
            g_vec = n_tau_basis.T @ (weights.W @ x_p[:n_tau])
            z_shift = -np.linalg.lstsq(h_mat, g_vec, rcond=rcond)[0]
            x_p = _readonly_float_array(x_p + null_basis @ z_shift, "x_particular")

    mult_gauge, attain_var, dims = _compute_dimension_table(
        snapshot, matrix_A, analysis.rank, analysis.nullity, rcond
    )
    model_prov = snapshot.model_provenance
    input_prov = snapshot.input_provenance
    xp_rec = QuantityRecord(
        values=x_p,
        units="mixed_SI",
        allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
        conventions=(conv_id,),
        hypotheses=("H-PARTICULAR-PSEUDOINVERSE",),
        model_provenance=model_prov,
        input_provenance=input_prov,
    )
    sv_rec = QuantityRecord(
        values=analysis.singular_values,
        units="1",
        allocation_dependence=AllocationDependence.INVARIANT,
        conventions=(conv_id,),
        hypotheses=(),
        model_provenance=model_prov,
        input_provenance=input_prov,
    )
    cond_rec = QuantityRecord(
        values=np.array([analysis.condition_number], dtype=np.float64),
        units="1",
        allocation_dependence=AllocationDependence.INVARIANT,
        conventions=(conv_id,),
        hypotheses=(),
        model_provenance=model_prov,
        input_provenance=input_prov,
    )
    return Decomposition(
        matrix_A=matrix_A,
        rhs_b=rhs_b,
        x_particular=x_p,
        null_basis=null_basis,
        rank_A=analysis.rank,
        nullity=analysis.nullity,
        multiplier_gauge=mult_gauge,
        attainable_torque_variation=attain_var,
        scaled_singular_values=analysis.singular_values,
        condition_number=analysis.condition_number,
        dimensions=dims,
        x_particular_record=xp_rec,
        singular_values_record=sv_rec,
        condition_number_record=cond_rec,
    )


def _resolve_side_dof_indices(
    snapshot: ClosedLoopSnapshot, side: LimbSide | Sequence[int]
) -> tuple[int, ...]:
    """Resolve side name ('left' or 'right') or explicit index sequence to DOF indices."""
    if isinstance(side, str):
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        coord_names = snapshot.topology.coordinate_order
        tokens = ("_l", "left", "ql") if side == "left" else ("_r", "right", "qr")
        matched = [
            idx
            for idx, name in enumerate(coord_names)
            if any(tok in name.lower() for tok in tokens)
        ]
        if not matched:
            raise ValueError(
                f"Could not resolve {side!r} DOF indices from {coord_names}"
            )
        unact = snapshot.unactuated_rows().tolist()
        combined = list(dict.fromkeys([*unact, *matched]))
        return tuple(int(i) for i in combined)

    indices = tuple(int(i) for i in side)
    if len(indices) == 0:
        raise ValueError("side_indices must be non-empty")
    n_v = snapshot.nv()
    for idx in indices:
        if idx < 0 or idx >= n_v:
            raise ValueError(f"side index {idx} out of bounds for n_v={n_v}")
    return indices


def zero_side_solve(
    snapshot: ClosedLoopSnapshot,
    side_indices: LimbSide | Sequence[int],
    *,
    rcond: float = 1e-10,
    damping: float = 0.0,
) -> ZeroSideResult:
    """Classify and evaluate zero-side solve J_{c,side}^T lambda = b_side (R3).

    Returns CONSISTENT_UNIQUE, CONSISTENT_NONUNIQUE(nullity), or
    INCONSISTENT(residual_scaled). Never silently claims a valid zero-side
    solution when the system is inconsistent.
    """
    if not isinstance(snapshot, ClosedLoopSnapshot):
        raise TypeError("snapshot must be a ClosedLoopSnapshot instance")
    if not np.isfinite(rcond) or not (0.0 < rcond < 1.0):
        raise ValueError(f"rcond must be in (0, 1), got {rcond}")
    if not np.isfinite(damping) or damping < 0.0:
        raise ValueError(f"damping must be finite and >= 0, got {damping}")

    resolved = _resolve_side_dof_indices(snapshot, side_indices)
    j_side_t = snapshot.J_c[:, resolved].T
    b_side = snapshot.generalized_load()[list(resolved)]
    m_c = int(j_side_t.shape[1])

    u, s, vt = np.linalg.svd(j_side_t, full_matrices=True)
    rank = int(np.count_nonzero(s > rcond * s[0])) if s.size > 0 and s[0] > 0 else 0
    if rank > 0:
        u_r = u[:, :rank]
        s_r = s[:rank]
        vt_r = vt[:rank, :]
        inv_factors = s_r / (s_r**2 + damping**2) if damping > 0.0 else 1.0 / s_r
        lam_ls = vt_r.T @ (inv_factors * (u_r.T @ b_side))
        b_proj = u_r @ (u_r.T @ b_side)
    else:
        lam_ls = np.zeros(m_c, dtype=np.float64)
        b_proj = np.zeros_like(b_side)

    res_vec = _readonly_float_array(b_side - b_proj, "residual_vector", ndim=1)
    res_norm = float(np.linalg.norm(res_vec))
    scale = max(float(np.linalg.norm(b_side)), 1.0)
    res_scaled = res_norm / scale
    nullity = int(m_c - rank)
    kernel_basis = _readonly_float_array(vt[rank:, :].T, "kernel_basis", ndim=2)
    lam_ls_ro = _readonly_float_array(lam_ls, "lambda_least_squares", ndim=1)

    if res_norm > rcond * scale:
        case: ZeroSideCase = "INCONSISTENT"
        lam_sol: FloatArray | None = None
        reported_nullity = 0
    elif nullity == 0:
        case = "CONSISTENT_UNIQUE"
        lam_sol = lam_ls_ro
        reported_nullity = 0
    else:
        case = "CONSISTENT_NONUNIQUE"
        lam_sol = lam_ls_ro
        reported_nullity = nullity

    res_rec = QuantityRecord(
        values=res_vec,
        units="N*m",
        allocation_dependence=AllocationDependence.INVARIANT,
        conventions=("C-ZERO-SIDE-RESIDUAL",),
        hypotheses=(),
        model_provenance=snapshot.model_provenance,
        input_provenance=snapshot.input_provenance,
    )
    return ZeroSideResult(
        case=case,
        side_indices=resolved,
        nullity=reported_nullity,
        residual_norm=res_norm,
        residual_scaled=res_scaled,
        residual_vector=res_vec,
        damping=float(damping),
        lambda_solution=lam_sol,
        lambda_least_squares=lam_ls_ro,
        kernel_basis=kernel_basis,
        residual_record=res_rec,
    )


def min_metric_torque(
    snapshot: ClosedLoopSnapshot,
    metric: AllocationMetric | ArrayLike,
    *,
    convention_id: str | None = None,
    rcond: float = 1e-10,
) -> QuantityRecord:
    """Compute the minimum-W-norm torque consistent with A x = b.

    Returns an INVARIANT QuantityRecord tagged with conventions=(metric.convention_id,).
    """
    if not isinstance(snapshot, ClosedLoopSnapshot):
        raise TypeError("snapshot must be a ClosedLoopSnapshot instance")
    if isinstance(metric, AllocationMetric):
        alloc_metric = (
            metric
            if convention_id is None
            else AllocationMetric(convention_id=convention_id, W=metric.W)
        )
    else:
        cid = convention_id if convention_id is not None else "C-METRIC-CUSTOM"
        alloc_metric = AllocationMetric(
            convention_id=cid, W=np.asarray(metric, dtype=np.float64)
        )

    decomp = decompose(snapshot, rcond=rcond)
    n_tau = int(snapshot.S.shape[0])
    if alloc_metric.W.shape != (n_tau, n_tau):
        raise ValueError(
            f"Metric W shape {alloc_metric.W.shape} must match (n_tau, n_tau)=({n_tau}, {n_tau})"
        )

    tau_p = decomp.x_particular[:n_tau]
    n_tau_basis = decomp.null_basis[:n_tau, :]
    if n_tau_basis.shape[1] > 0:
        h_mat = n_tau_basis.T @ alloc_metric.W @ n_tau_basis
        g_vec = n_tau_basis.T @ (alloc_metric.W @ tau_p)
        z_opt = -np.linalg.lstsq(h_mat, g_vec, rcond=rcond)[0]
        tau_opt = tau_p + n_tau_basis @ z_opt
    else:
        tau_opt = tau_p.copy()

    return QuantityRecord(
        values=tau_opt,
        units="N*m",
        allocation_dependence=AllocationDependence.INVARIANT,
        conventions=(alloc_metric.convention_id,),
        hypotheses=(),
        model_provenance=snapshot.model_provenance,
        input_provenance=snapshot.input_provenance,
    )
