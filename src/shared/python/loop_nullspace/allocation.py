"""Prior-conditioned trajectory conic allocation (NSX-23, #12566).

``allocate_trajectory`` picks one representative torque/load history from the
family of per-frame feasible sets (:class:`~.feasible_set.FeasibleSet`) by
minimising a **strictly convex** objective (rule R6): capacity-weighted effort,
a torque-rate smoothness term, an optional reference-tracking term and an
optional gross-power epigraph with separate ``c+`` / ``c-`` costs.  The result
is ``FAMILY_DEPENDENT`` and carries the hypotheses that produced it.

Formulation.  The unknown is ``w = (z_1..z_T, s)`` with ``s`` the gross-power
epigraph variables.  Torques are affine, ``tau = T0 + Ntil z``, so every
quadratic term is ``|A tau - a|^2 = |A Ntil z + (A T0 - a)|^2`` and the whole
problem is a convex QP with second-order-cone rows (exact Coulomb cones),
solved by :func:`~.solvers.solve_socp` (Clarabel) with its KKT residuals and
duals returned.  Cones enter exactly; linearised cone rows of a feasible set
are ignored in favour of the exact blocks.

Strict convexity is a build-time precondition.  The gross-power epigraph is
piecewise linear and contributes no curvature.  Curvature is required on the
*retained* space: the z-directions that change a torque.  Directions that
change no torque (multiplier/``f_g`` gauge) are reported, not penalised; the
optimal torque is unique, the gauge coordinates are not.

Gross power (NSX-00 identity)::

    sum(c+ [p]+ + c- [-p]+) = 1/2 (c+ + c-) sum|p| + 1/2 (c+ - c-) P_net

With uniform ``c+``, ``c-`` and ``P_net`` constant over the family
(``v^T dtau/dz = 0``, i.e. ``J_c v = 0`` with actuated ``S = I``) the ratio
``c-/c+`` is a dead parameter; the reduction is detected, reported and used.

Non-claim: the optimum is a hypothesis-conditioned representative, not an
estimate of the true hand loads.  Inverse optimal control is out of scope
(NSX-71).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from src.shared.python.loop_nullspace.allocation_spec import (
    AllocationResult,
    ConvexityReport,
    GrossPowerSpec,
    NotStrictlyConvexError,
    ObjectiveSpec,
    PowerReduction,
    VarianceTerm,
)
from src.shared.python.loop_nullspace.capacity import CapacityVector
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.decomposition import Decomposition
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.registry import load_registry
from src.shared.python.loop_nullspace.solvers import (
    SolveStatus,
    SOCBlock,
    ConicSettings,
    QuadraticObjective,
    solve_socp,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AllocationResult",
    "ConvexityReport",
    "GrossPowerSpec",
    "NotStrictlyConvexError",
    "ObjectiveSpec",
    "PowerReduction",
    "VarianceTerm",
    "allocate_trajectory",
    "torque_scale_from_capacity",
]

CURVATURE_RTOL = 1e-9
RANK_RTOL = 1e-10
NET_POWER_RTOL = 1e-9
EFFORT_ID = "H-EFFORT-CAPACITY"
RATE_ID = "H-SMOOTH-TORQUE-RATE"
GROSS_ID = "H-GROSS-POWER-COST"
METRIC_CONVENTION = "C-METRIC-CAPACITY"
SOLVER_TOL = 1e-11


def torque_scale_from_capacity(capacity: CapacityVector) -> Array:
    """Symmetric per-joint scale: mean of the positive and negative capacities.

    Convention ``C-METRIC-CAPACITY``; a choice of metric, not physiology.
    """
    if not isinstance(capacity, CapacityVector):
        raise TypeError(f"capacity must be CapacityVector, got {type(capacity)!r}")
    return np.asarray(
        0.5 * (capacity.tau_max_positive_nm + capacity.tau_max_negative_nm), float
    )


# ----------------------------------------------------------------- layout
@dataclass(frozen=True)
class _Layout:
    """Affine torque map ``tau = t0 + ntil z`` over all frames."""

    frames: tuple[FeasibleSet, ...]
    n_a: int
    offsets: np.ndarray
    ntil: sp.csr_matrix
    t0: Array
    omega: Array
    times: Array

    @property
    def t_count(self) -> int:
        return len(self.frames)

    @property
    def n_z_total(self) -> int:
        return int(self.offsets[-1])


def _trapezoid(times: Array) -> Array:
    if times.size == 1:
        return np.ones(1)
    w = np.zeros(times.size)
    half = np.diff(times) / 2.0
    w[:-1] += half
    w[1:] += half
    return w


def _validated_frames(
    feasible_sets: Sequence[FeasibleSet], objective: ObjectiveSpec
) -> tuple[FeasibleSet, ...]:
    if not isinstance(objective, ObjectiveSpec):
        raise TypeError(f"objective must be ObjectiveSpec, got {type(objective)!r}")
    frames = tuple(feasible_sets)
    if not frames:
        raise ValueError("feasible_sets must contain at least one frame")
    if objective.times_s.size != len(frames):
        raise ValueError(
            f"times_s has {objective.times_s.size} entries for {len(frames)} frames"
        )
    n_a = objective.torque_scale.size
    for k, fs in enumerate(frames):
        if not isinstance(fs, FeasibleSet):
            raise TypeError(f"frame {k} must be FeasibleSet, got {type(fs)!r}")
        if fs.decomposition is None:
            raise ValueError(
                f"frame {k} has no decomposition; build it with "
                "FeasibleSet.from_hypotheses (torques are read from x_p + N z)"
            )
        if fs.decomposition.n_actuated != n_a:
            raise ValueError(
                f"frame {k} has {fs.decomposition.n_actuated} actuated torques, "
                f"torque_scale has {n_a}"
            )
    gp = objective.gross_power
    if gp is not None and gp.joint_velocity.shape != (len(frames), n_a):
        raise ValueError(
            f"joint_velocity must have shape ({len(frames)}, {n_a}), "
            f"got {gp.joint_velocity.shape}"
        )
    if gp is not None and gp.c_plus.size != n_a:
        raise ValueError(f"c_plus/c_minus must have {n_a} entries")
    if sum(fs.n_z for fs in frames) == 0:
        raise ValueError(
            "every frame has a zero-dimensional family; nothing to allocate"
        )
    return frames


def _validated_reference(
    reference: Any, objective: ObjectiveSpec, t_count: int
) -> Array | None:
    n_a = objective.torque_scale.size
    if reference is None:
        if objective.reference_weight > 0.0:
            raise ValueError(
                "reference_weight > 0 but no reference was supplied (reference=None)"
            )
        return None
    if objective.reference_weight <= 0.0:
        raise ValueError("a reference was supplied but reference_weight is 0")
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (t_count, n_a):
        raise ValueError(
            f"reference must have shape ({t_count}, {n_a}), got {ref.shape}"
        )
    if not np.all(np.isfinite(ref)):
        raise ValueError("reference must be finite")
    return ref


def _decomposition(fs: FeasibleSet) -> Decomposition:
    """Return the frame's decomposition, rejecting a feasible set without one."""
    if fs.decomposition is None:
        raise ValueError("feasible set carries no decomposition")
    return fs.decomposition


def _layout(frames: tuple[FeasibleSet, ...], objective: ObjectiveSpec) -> _Layout:
    n_a = objective.torque_scale.size
    sizes = [fs.n_z for fs in frames]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    blocks = []
    for k, fs in enumerate(frames):
        dec = _decomposition(fs)
        left = sp.csr_matrix((n_a, int(offsets[k])))
        right = sp.csr_matrix((n_a, int(offsets[-1] - offsets[k + 1])))
        blocks.append(sp.hstack([left, sp.csr_matrix(dec.N[:n_a]), right]))
    t0 = np.concatenate([_decomposition(fs).x_p[:n_a] for fs in frames])
    return _Layout(
        frames,
        n_a,
        offsets,
        sp.vstack(blocks).tocsr(),
        t0,
        _trapezoid(objective.times_s),
        objective.times_s,
    )


# ------------------------------------------------------- quadratic terms
def _scale_diag(lay: _Layout, scale: Array, weight: float) -> sp.csr_matrix:
    root = np.sqrt(weight * np.repeat(lay.omega, lay.n_a)) / np.tile(scale, lay.t_count)
    return sp.diags(root).tocsr()


def _rate_operator(lay: _Layout, scale: Array, weight: float) -> sp.csr_matrix | None:
    if lay.t_count < 2:
        return None
    n_a = lay.n_a
    rows = []
    for k in range(lay.t_count - 1):
        dt = lay.times[k + 1] - lay.times[k]
        sel = sp.lil_matrix((n_a, lay.t_count * n_a))
        for i in range(n_a):
            sel[i, k * n_a + i] = -1.0
            sel[i, (k + 1) * n_a + i] = 1.0
        rows.append(sp.diags(np.sqrt(weight / dt) / scale) @ sel.tocsr())
    return sp.csr_matrix(sp.vstack(rows))


def _variance_operator(lay: _Layout, term: VarianceTerm) -> sp.csr_matrix:
    """Diagonal ``sqrt(weight * coeff)`` on the stacked torques (NSX-92, M1)."""
    if term.coeff.shape != (lay.t_count, lay.n_a):
        raise ValueError(
            f"variance coeff must be {(lay.t_count, lay.n_a)}, got {term.coeff.shape}"
        )
    return sp.diags(np.sqrt(term.weight * term.coeff.reshape(-1))).tocsr()


def _operators(
    lay: _Layout,
    scale: Array,
    weights: dict[str, float],
    variance: VarianceTerm | None = None,
) -> dict[str, sp.csr_matrix]:
    """Residual operators ``A`` of the enabled quadratic terms (weight > 0)."""
    ops: dict[str, sp.csr_matrix] = {}
    if variance is not None:
        ops["variance"] = _variance_operator(lay, variance)
    if weights["effort"] > 0.0:
        ops["effort"] = _scale_diag(lay, scale, weights["effort"])
    if weights["rate"] > 0.0:
        rate = _rate_operator(lay, scale, weights["rate"])
        if rate is not None:
            ops["rate"] = rate
    if weights["reference"] > 0.0:
        ops["reference"] = _scale_diag(lay, scale, weights["reference"])
    return ops


def _objective_operators(lay: _Layout, objective: ObjectiveSpec) -> dict[str, Any]:
    """Operators of every quadratic term the objective enables."""
    weights = _weights(objective)
    return _operators(lay, objective.torque_scale, weights, objective.variance)


def _weights(objective: ObjectiveSpec) -> dict[str, float]:
    return {
        "effort": objective.effort_weight,
        "rate": objective.rate_weight,
        "reference": objective.reference_weight,
    }


def _gram(lay: _Layout, ops: dict[str, sp.csr_matrix]) -> np.ndarray:
    z = lay.n_z_total
    total = np.zeros((z, z))
    for op in ops.values():
        m = op @ lay.ntil
        total += (m.T @ m).toarray()
    return total


def _curvature_rank(gram: np.ndarray) -> tuple[int, float, np.ndarray]:
    """``(rank, largest eigenvalue, eigenvalues descending)`` of a PSD matrix."""
    if gram.size == 0:
        return 0, 0.0, np.zeros(0)
    eig = np.sort(np.linalg.eigvalsh(0.5 * (gram + gram.T)))[::-1]
    lam = float(eig[0])
    if lam <= 0.0:
        return 0, 0.0, eig
    return int(np.sum(eig > CURVATURE_RTOL * lam)), lam, eig


def _required_rank(lay: _Layout) -> int:
    total = 0
    for fs in lay.frames:
        n = _decomposition(fs).N[: lay.n_a]
        if n.size == 0:
            continue
        sv = np.linalg.svd(n, compute_uv=False)
        total += (
            int(np.sum(sv > RANK_RTOL * max(float(sv[0]), 1e-300))) if sv.size else 0
        )
    return total


def _enabled_terms(objective: ObjectiveSpec, has_reference: bool) -> tuple[str, ...]:
    names = []
    if objective.effort_weight > 0.0:
        names.append("effort")
    if objective.rate_weight > 0.0:
        names.append("rate")
    if objective.reference_weight > 0.0 and has_reference:
        names.append("reference")
    if objective.variance is not None:
        names.append("variance")
    if objective.gross_power is not None:
        names.append("gross power (piecewise linear, no curvature)")
    return tuple(names)


def _unit_norm(mat: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(mat))) if mat.size else 0.0
    return mat / peak if peak > 0.0 else mat


def _remedies(lay: _Layout, objective: ObjectiveSpec, base: np.ndarray, need: int):
    """Disabled terms that alone would restore full retained-space curvature."""
    out = []
    for term in ("effort", "reference"):
        if _weights(objective)[term] > 0.0:
            continue
        unit = {"effort": 0.0, "rate": 0.0, "reference": 0.0} | {term: 1.0}
        extra = _gram(lay, _operators(lay, objective.torque_scale, unit))
        rank = _curvature_rank(_unit_norm(base) + _unit_norm(extra))[0]
        if rank >= need:
            out.append(term)
    return out


def _check_convexity(
    lay: _Layout, objective: ObjectiveSpec, ops: dict[str, sp.csr_matrix]
) -> ConvexityReport:
    gram = _gram(lay, ops)
    need = _required_rank(lay)
    rank, lam, eig = _curvature_rank(gram)
    enabled = _enabled_terms(objective, "reference" in ops)
    gauge = lay.n_z_total - need
    if rank < need:
        fix = _remedies(lay, objective, gram, need)
        names = ", ".join(fix) if fix else "effort or reference"
        head = (
            "no quadratic term is enabled"
            if lam == 0.0
            else f"curvature rank {rank} of the required {need}"
        )
        raise NotStrictlyConvexError(
            f"objective is not strictly convex on the retained torque space: "
            f"{head}; enabled terms: {', '.join(enabled) or 'none'}. "
            f"Missing term(s) that would supply the curvature: {names} "
            "(set effort_weight > 0, or reference_weight > 0 with a reference). "
            "The torque-rate and gross-power terms alone leave a flat direction."
        )
    ratio = float(eig[need - 1] / lam) if need else 1.0
    return ConvexityReport(rank, need, gauge, ratio, enabled)


# ------------------------------------------------------- gross power
def _net_power_invariant(lay: _Layout, gp: GrossPowerSpec) -> bool:
    for k in range(lay.t_count):
        rows = slice(k * lay.n_a, (k + 1) * lay.n_a)
        grad = gp.joint_velocity[k] @ lay.ntil[rows].toarray()
        scale = max(1.0, float(np.linalg.norm(gp.joint_velocity[k])))
        if grad.size and float(np.max(np.abs(grad))) > NET_POWER_RTOL * scale:
            return False
    return True


def _classify_power(lay: _Layout, gp: GrossPowerSpec | None) -> PowerReduction | None:
    if gp is None:
        return None
    sym = bool(np.allclose(gp.c_plus, gp.c_minus, rtol=1e-12, atol=0.0))
    uniform = bool(
        np.allclose(gp.c_plus, gp.c_plus[0], rtol=1e-12, atol=0.0)
        and np.allclose(gp.c_minus, gp.c_minus[0], rtol=1e-12, atol=0.0)
    )
    invariant = _net_power_invariant(lay, gp)
    if sym:
        kind = "ABS_POWER_ONLY"
    elif uniform and invariant:
        kind = "UNIFORM_NET_PLUS_ABS"
    else:
        kind = "NONE"
    dead = kind == "UNIFORM_NET_PLUS_ABS"
    return PowerReduction(
        kind, uniform, invariant, dead, kind != "NONE" and gp.allow_reduction
    )


def _epigraph_rows(
    lay: _Layout, gp: GrossPowerSpec, reduced: bool
) -> tuple[sp.csr_matrix, Array]:
    """Rows of ``c+ p - s <= 0`` and ``-c- p - s <= 0`` over ``w = (z, s)``."""
    n_a, t_count = lay.n_a, lay.t_count
    cp, cm = gp.c_plus, gp.c_minus
    if reduced:
        cp = cm = 0.5 * (gp.c_plus + gp.c_minus)
    vel = gp.joint_velocity.reshape(-1)
    coef_up = np.tile(cp, t_count) * vel
    coef_lo = -np.tile(cm, t_count) * vel
    n_s = t_count * n_a
    eye = -sp.identity(n_s, format="csr")
    up = sp.hstack([sp.diags(coef_up) @ lay.ntil, eye])
    lo = sp.hstack([sp.diags(coef_lo) @ lay.ntil, eye])
    rhs = np.concatenate([-coef_up * lay.t0, -coef_lo * lay.t0])
    return sp.vstack([up, lo]).tocsr(), rhs


# ------------------------------------------------------------ assembly
def _embed(mat: Any, start: int, stop: int, total: int) -> sp.csr_matrix:
    m = sp.csr_matrix(mat)
    left = sp.csr_matrix((m.shape[0], start))
    right = sp.csr_matrix((m.shape[0], total - stop))
    return sp.hstack([left, m, right]).tocsr()


@dataclass(frozen=True)
class _Problem:
    n_var: int
    p_mat: sp.csr_matrix
    c_vec: Array
    a_ub: sp.csr_matrix
    b_ub: Array
    cones: tuple[SOCBlock, ...]
    row_labels: tuple[str, ...]
    cone_labels: tuple[str, ...]


def _frame_constraints(
    lay: _Layout, n_var: int
) -> tuple[list[sp.csr_matrix], list[Array], list[str], list[SOCBlock], list[str]]:
    mats: list[sp.csr_matrix] = []
    rhs: list[Array] = []
    labels: list[str] = []
    blocks: list[SOCBlock] = []
    cone_labels: list[str] = []
    for k, fs in enumerate(lay.frames):
        lo, hi = int(lay.offsets[k]), int(lay.offsets[k + 1])
        form = fs.to_socp()
        if form.A_ub.shape[0]:
            mats.append(_embed(form.A_ub, lo, hi, n_var))
            rhs.append(np.asarray(form.b_ub, float))
            kept = [r.label for r in fs.rows if r.kind != "cone_linear"]
            labels += [f"frame{k}:{lab}" for lab in kept]
        for blk, cone in zip(form.cones, fs.cones, strict=True):
            blocks.append(SOCBlock(_embed(blk.A, lo, hi, n_var), blk.b))
            cone_labels.append(f"frame{k}:{cone.label}")
    return mats, rhs, labels, blocks, cone_labels


def _assemble(
    lay: _Layout,
    objective: ObjectiveSpec,
    ops: dict[str, sp.csr_matrix],
    ref: Array | None,
    power: PowerReduction | None,
) -> _Problem:
    z_n = lay.n_z_total
    gp = objective.gross_power
    n_s = lay.t_count * lay.n_a if gp is not None else 0
    n_var = z_n + n_s
    p_zz = sp.csr_matrix((z_n, z_n))
    c_z = np.zeros(z_n)
    for name, op in ops.items():
        m = op @ lay.ntil
        target: Array = np.zeros(op.shape[0])
        if name == "reference" and ref is not None:
            target = op @ ref.reshape(-1)
        p_zz = p_zz + 2.0 * (m.T @ m)
        c_z += 2.0 * (m.T @ (op @ lay.t0 - target))
    p_full = sp.csr_matrix(
        sp.block_diag([p_zz, sp.csr_matrix((n_s, n_s))], format="csr")
    )
    c_vec = np.concatenate([c_z, np.zeros(n_s)])
    mats, rhs, labels, blocks, cone_labels = _frame_constraints(lay, n_var)
    if gp is not None:
        assert power is not None
        rows, b_epi = _epigraph_rows(lay, gp, power.reduced_form_used)
        mats.append(rows)
        rhs.append(b_epi)
        c_vec[z_n:] = gp.weight * np.repeat(lay.omega, lay.n_a)
    a_ub = sp.vstack(mats).tocsr() if mats else sp.csr_matrix((0, n_var))
    b_ub = np.concatenate(rhs) if rhs else np.zeros(0)
    return _Problem(
        n_var,
        p_full,
        c_vec,
        a_ub,
        b_ub,
        tuple(blocks),
        tuple(labels),
        tuple(cone_labels),
    )


# ------------------------------------------------------------ evaluation
def _objective_terms(
    tau: Array,
    objective: ObjectiveSpec,
    lay: _Layout,
    ref: Array | None,
) -> dict[str, float]:
    """Directly evaluated, unreduced objective terms at ``tau`` (frames, n_a)."""
    s = objective.torque_scale
    om = lay.omega[:, None]
    terms = {
        "effort": float(objective.effort_weight * np.sum(om * (tau / s) ** 2)),
        "rate": 0.0,
        "reference": 0.0,
        "gross_power": 0.0,
    }
    if objective.variance is not None:
        vt = objective.variance
        terms["variance"] = float(vt.weight * np.sum(vt.coeff * tau**2))
    if lay.t_count > 1:
        dt = np.diff(lay.times)[:, None]
        terms["rate"] = float(
            objective.rate_weight * np.sum((np.diff(tau, axis=0) / s) ** 2 / dt)
        )
    if ref is not None:
        terms["reference"] = float(
            objective.reference_weight * np.sum(om * ((tau - ref) / s) ** 2)
        )
    gp = objective.gross_power
    if gp is not None:
        p = tau * gp.joint_velocity
        cost = gp.c_plus * np.maximum(p, 0.0) + gp.c_minus * np.maximum(-p, 0.0)
        terms["gross_power"] = float(gp.weight * np.sum(om * cost))
    return terms


def _net_work_offset(
    tau: Array, gp: GrossPowerSpec | None, power: PowerReduction | None, lay: _Layout
) -> float:
    if gp is None or power is None or power.kind != "UNIFORM_NET_PLUS_ABS":
        return 0.0
    p_net = np.sum(tau * gp.joint_velocity, axis=1)
    return float(
        gp.weight * 0.5 * (gp.c_plus[0] - gp.c_minus[0]) * np.sum(lay.omega * p_net)
    )


def _unverified(ids: Sequence[str]) -> tuple[str, ...]:
    registry = load_registry()
    bad = []
    for hid in ids:
        try:
            registry.require_verified_for_display([hid])
        except ValueError:
            bad.append(hid)
    return tuple(bad)


def _hypotheses(lay: _Layout, objective: ObjectiveSpec) -> tuple[str, ...]:
    ids: list[str] = []
    for fs in lay.frames:
        ids += list(fs.hypothesis_ids)
    if objective.effort_weight > 0.0:
        ids.append(EFFORT_ID)
    if objective.rate_weight > 0.0:
        ids.append(RATE_ID)
    if objective.gross_power is not None:
        ids.append(GROSS_ID)
    if objective.variance is not None:
        ids += list(objective.variance.hypothesis_ids)
    return tuple(dict.fromkeys(ids))


def _record(
    lay: _Layout, objective: ObjectiveSpec, tau: Array, hypotheses: tuple[str, ...]
) -> QuantityRecord:
    recs = _decomposition(lay.frames[0]).records
    first = next(iter(recs.values()), None)
    model = (
        first.model_provenance
        if first is not None
        else ModelProvenance("unspecified-decomposition", "0")
    )
    return QuantityRecord(
        values=tau,
        units="N m",
        allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
        model_provenance=model,
        input_provenance=InputProvenance(
            "allocation",
            f"{lay.t_count}-frame trajectory",
            "hypothesis-conditioned representative (NSX-23)",
        ),
        conventions=(METRIC_CONVENTION,),
        hypotheses=hypotheses,
    )


def _duals(sol_z: Array, prob: _Problem) -> tuple[dict[str, float], dict[str, Array]]:
    n_rows = len(prob.row_labels)
    rows = {
        lab: float(v) for lab, v in zip(prob.row_labels, sol_z[:n_rows], strict=True)
    }
    pos = prob.a_ub.shape[0]
    cones: dict[str, Array] = {}
    for lab, blk in zip(prob.cone_labels, prob.cones, strict=True):
        cones[lab] = np.array(sol_z[pos : pos + blk.dim])
        pos += blk.dim
    return rows, cones


# ------------------------------------------------------------------ API
def allocate_trajectory(
    feasible_sets: Sequence[FeasibleSet],
    objective: ObjectiveSpec,
    *,
    reference: Any = None,
) -> AllocationResult:
    """Strictly convex trajectory allocation over per-frame feasible sets.

    Preconditions (``ValueError``/``TypeError`` otherwise): at least one frame;
    every frame a :class:`FeasibleSet` with a decomposition and the objective's
    number of actuated torques; ``times_s`` and gross-power velocities match the
    frame count; ``reference`` (frames x n_a torques) is given iff
    ``reference_weight > 0``.  :class:`NotStrictlyConvexError` is raised when
    the quadratic part has no curvature on the retained torque space and names
    the terms that would supply it.

    Postconditions: ``status`` is a :class:`SolveStatus`.  On ``OPTIMAL`` the
    result holds ``z``, ``x``, ``torque`` (frames x n_a), directly evaluated
    objective terms, Clarabel KKT residuals (scaled), the row and cone duals,
    the detected c+/c- reduction, a convexity report and a ``FAMILY_DEPENDENT``
    :class:`QuantityRecord` naming every hypothesis.  An infeasible family
    returns ``INFEASIBLE`` with no record (never a relaxed answer).
    """
    frames = _validated_frames(feasible_sets, objective)
    ref = _validated_reference(reference, objective, len(frames))
    lay = _layout(frames, objective)
    ops = _objective_operators(lay, objective)
    convexity = _check_convexity(lay, objective, ops)
    power = _classify_power(lay, objective.gross_power)
    prob = _assemble(lay, objective, ops, ref, power)
    t0 = time.perf_counter()
    has_rows = prob.a_ub.shape[0] > 0
    sol = solve_socp(
        QuadraticObjective(prob.c_vec, prob.p_mat),
        None,
        None,
        prob.a_ub if has_rows else None,
        prob.b_ub if has_rows else None,
        prob.cones,
        settings=ConicSettings(tol=SOLVER_TOL),
    )
    elapsed = time.perf_counter() - t0
    if sol.status is not SolveStatus.OPTIMAL or sol.x is None or sol.z is None:
        return AllocationResult(
            sol.status,
            reason=sol.reason or sol.status.name,
            solve_time_s=elapsed,
            iterations=sol.iterations,
            power_reduction=power,
            convexity=convexity,
        )
    z_all = np.asarray(sol.x[: lay.n_z_total])
    z_list = tuple(
        z_all[int(lay.offsets[k]) : int(lay.offsets[k + 1])].copy()
        for k in range(lay.t_count)
    )
    x_list = tuple(
        _decomposition(fs).x(z) for fs, z in zip(frames, z_list, strict=True)
    )
    tau = np.array([x[: lay.n_a] for x in x_list])
    terms = _objective_terms(tau, objective, lay, ref)
    offset = _net_work_offset(tau, objective.gross_power, power, lay)
    if power is not None:
        power = PowerReduction(
            power.kind,
            power.uniform,
            power.net_power_invariant,
            power.ratio_is_dead_parameter,
            power.reduced_form_used,
            offset,
        )
    hyps = _hypotheses(lay, objective)
    rows, cones = _duals(np.asarray(sol.z), prob)
    logger.info(
        "NSX-23 allocation: %d frames, %d iterations, %.3f s, KKT %s",
        lay.t_count,
        sol.iterations,
        elapsed,
        sol.kkt_residuals,
    )
    return AllocationResult(
        status=sol.status,
        times_s=lay.times,
        z=z_list,
        x=x_list,
        torque=tau,
        objective_terms=terms,
        objective_total=float(sum(terms.values())),
        kkt_residuals=sol.kkt_residuals,
        row_duals=rows,
        cone_duals=cones,
        solve_time_s=elapsed,
        iterations=sol.iterations,
        power_reduction=power,
        convexity=convexity,
        record=_record(lay, objective, tau, hyps),
        reference_used=ref is not None,
        unverified_hypothesis_ids=_unverified(hyps),
    )
