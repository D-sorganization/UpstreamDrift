"""Noise-aware null-space allocation: minimum face-angle variance (NSX-92, #12721).

Chooses the representative allocation ``x = x_p + N z`` of every frame that
minimises the linear outcome variance of :class:`~.noise_aware_model.NoiseAwareModel`
under the declared mechanisms, subject to the exact feasible sets (capacity
rows, friction cones, grip rows).

* **Rule V1.**  The allocation changes the variance only through M1
  (signal-dependent noise, ``k_j > 0``) or M2 (geometric preload moving
  ``A_t``).  With neither, the variance is a constant: the solve is skipped, the
  status is ``OBJECTIVE_FLAT_INVARIANT`` and the variance record is
  ``INVARIANT``.  That is the negative control ``H-NOISE-ADDITIVE``.
* **M1 only (convex).**  Kernels are constant, the objective is the diagonal
  torque quadratic ``sum_tj coeff_tj tau_tj^2`` (``coeff = k_j^2 S_in^2 / dt``)
  plus a constant, solved by the NSX-23 conic allocator
  (:func:`~.allocation.allocate_trajectory`) through
  :class:`~.allocation_spec.VarianceTerm`.  The optimum is global (KKT).
* **M2 on (non-convex).**  ``Var`` is polynomial in ``z`` and in general not
  convex.  A damped Gauss-Newton sequential convex program is run from the
  minimum-effort allocation (and optional extra starts).  Each subproblem is a
  convex SOCP over the same feasible sets.  Because ``Var = |r|^2``, the
  Gauss-Newton model gradient equals the exact gradient, so a vanishing step
  with a small subproblem KKT residual certifies a first-order stationary
  point.  It is **not** claimed to be globally optimal; the several local
  optima found are reported.

The variance of the reported allocation is always re-evaluated directly from
the model (not read from the optimiser), next to the minimum-effort reference.

Non-claim: a hypothesis-conditioned representative; the benefit is conditional
on the declared mechanisms and amplitudes (V8).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import scipy.sparse as sp

from src.shared.python.core.contracts import ensure
from src.shared.python.loop_nullspace.allocation import (
    SOLVER_TOL,
    _frame_constraints,
    _hypotheses,
    _layout,
    _objective_terms,
    _operators,
    _validated_frames,
    allocate_trajectory,
)
from src.shared.python.loop_nullspace.allocation_spec import (
    NotStrictlyConvexError,
    ObjectiveSpec,
    VarianceTerm,
)
from src.shared.python.loop_nullspace.evidence import Array, QuantityRecord
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.noise_aware_model import (
    NoiseAwareModel,
    wrap_probability_bound,
)
from src.shared.python.loop_nullspace.noise_aware_spec import (
    NoiseAwareResult,
    NoiseAwareSettings,
    NoiseAwareStatus,
)
from src.shared.python.loop_nullspace.pareto import (
    _bound_cone,
    _eps_block,
    pareto_front,
)
from src.shared.python.loop_nullspace.solvers import (
    ConicSettings,
    KKTResiduals,
    QuadraticObjective,
    SolveStatus,
    SOCBlock,
    solve_socp,
)
from src.shared.python.loop_nullspace.trajectory_system import unverified_ids
from src.shared.python.loop_nullspace.variance_records import (
    Mechanism,
    VarianceDeclaration,
    _record as variance_quantity_record,
)

logger = logging.getLogger(__name__)

__all__ = ["effort_of", "minimise_face_variance", "reference_allocation"]

MU_FLOOR_REL = 1e-9
MU_GROWTH = 10.0
MU_MAX_STEPS = 12


# ---------------------------------------------------------------- helpers
def _effort_spec(times: Array, scale: Array, weight: float = 1.0) -> ObjectiveSpec:
    return ObjectiveSpec(times_s=times, torque_scale=scale, effort_weight=weight)


def effort_of(
    frames: tuple[FeasibleSet, ...], times: Array, scale: Array, torque: Array
) -> float:
    """Capacity-weighted effort ``sum_k w_k |S^-1 tau_k|^2`` of ``torque``.

    Pre: ``torque`` is ``(frames, n_a)`` and finite.  Post: non-negative.
    """
    spec = _effort_spec(times, scale)
    lay = _layout(frames, spec)
    tau = np.asarray(torque, dtype=float)
    if tau.shape != (len(frames), scale.size) or not np.all(np.isfinite(tau)):
        raise ValueError(
            f"torque must be finite with shape {(len(frames), scale.size)}"
        )
    return float(_objective_terms(tau, spec, lay, None)["effort"])


def reference_allocation(feasible_sets: Any, times_s: Any, torque_scale: Any) -> Any:
    """Minimum-effort NSX-23 allocation: the comparison baseline (V4).

    Returns the :class:`~.allocation_spec.AllocationResult`; its status is not
    OPTIMAL when the family is infeasible.
    """
    spec = _effort_spec(np.asarray(times_s, float), np.asarray(torque_scale, float))
    return allocate_trajectory(tuple(feasible_sets), spec)


def _check_declaration(model: NoiseAwareModel, decl: VarianceDeclaration) -> None:
    if not isinstance(decl, VarianceDeclaration):
        raise TypeError("declaration must be a VarianceDeclaration")
    if model.spec.has_signal_law and (
        Mechanism.SIGNAL_DEPENDENT_NOISE not in decl.mechanisms
    ):
        raise ValueError("the model has an M1 signal law the declaration omits")
    if model.preload_active and Mechanism.GEOMETRIC_PRELOAD not in decl.mechanisms:
        raise ValueError("the model has an M2 preload slope the declaration omits")


def _layout_for(
    frames: tuple[FeasibleSet, ...], times: Array, scale: Array, model: NoiseAwareModel
) -> Any:
    lay = _layout(frames, _effort_spec(times, scale))
    if model.n_steps != lay.t_count:
        raise ValueError(
            f"model has {model.n_steps} steps for {lay.t_count} feasible-set frames"
        )
    if model.n_channels != lay.n_a:
        raise ValueError(
            f"model has {model.n_channels} channels for {lay.n_a} actuated torques"
        )
    if model.n_z != lay.n_z_total:
        raise ValueError(
            f"model n_z={model.n_z} does not match the stacked null dimension "
            f"{lay.n_z_total}"
        )
    return lay


def _tau(lay: Any, z: Array) -> Array:
    return np.asarray((lay.t0 + lay.ntil @ z).reshape(lay.t_count, lay.n_a))


def _split(lay: Any, z_all: Array) -> tuple[Array, ...]:
    return tuple(
        z_all[int(lay.offsets[k]) : int(lay.offsets[k + 1])].copy()
        for k in range(lay.t_count)
    )


def _stack(z_frames: Any) -> Array:
    return np.concatenate([np.asarray(z, dtype=float).ravel() for z in z_frames])


# ------------------------------------------------------------ record/result
def _typed_record(
    variance: float,
    decl: VarianceDeclaration,
    units: str,
    *,
    flat: bool,
) -> QuantityRecord:
    ids = () if flat else decl.ids()
    return variance_quantity_record(
        np.array([variance]), f"({units})^2", decl, ids, decl.noise_hypotheses
    )


def _assemble_result(
    status: NoiseAwareStatus,
    lay: Any,
    model: NoiseAwareModel,
    z_all: Array,
    ctx: dict[str, Any],
    **extra: Any,
) -> NoiseAwareResult:
    """Result with every number re-evaluated directly from the model."""
    frames, decl, settings = ctx["frames"], ctx["decl"], ctx["settings"]
    tau = _tau(lay, z_all)
    parts = model.variance_parts(z_all, tau)
    total = float(sum(parts.values()))
    resid, jac = model.residuals(z_all, tau, lay.ntil.toarray())
    grad = float(np.max(np.abs(2.0 * jac.T @ resid)))
    flat = status is NoiseAwareStatus.OBJECTIVE_FLAT_INVARIANT
    hyps = tuple(
        dict.fromkeys((*_frame_ids(frames), *decl.ids(), *decl.noise_hypotheses))
    )
    z_frames = _split(lay, z_all)
    x_frames = tuple(
        fs.decomposition.x(z) for fs, z in zip(frames, z_frames, strict=True)
    )
    ref_tau = ctx["ref_tau"]
    return NoiseAwareResult(
        status=status,
        z=z_frames,
        x=x_frames,
        torque=tau,
        variance=total,
        variance_parts=parts,
        reference_variance=model.variance(ctx["ref_z"], ref_tau),
        reference_torque=ref_tau,
        reference_z=_split(lay, ctx["ref_z"]),
        effort=effort_of(frames, ctx["times"], ctx["scale"], tau),
        reference_effort=effort_of(frames, ctx["times"], ctx["scale"], ref_tau),
        gradient_norm=grad,
        mechanisms=tuple(sorted(m.value for m in decl.mechanisms)),
        hypotheses=hyps,
        unverified_hypothesis_ids=unverified_ids(hyps),
        wrap_probability_bound=wrap_probability_bound(total),
        record=_typed_record(total, decl, settings.outcome_units, flat=flat),
        **extra,
    )


def _frame_ids(frames: tuple[FeasibleSet, ...]) -> tuple[str, ...]:
    ids: list[str] = []
    for fs in frames:
        ids += list(fs.hypothesis_ids)
    return tuple(dict.fromkeys(ids))


def _failure(status: NoiseAwareStatus, reason: str) -> NoiseAwareResult:
    return NoiseAwareResult(status=status, reason=reason)


# ---------------------------------------------------------------- convex M1
def _normalised_variance_term(coeff: Array, scale: Array) -> VarianceTerm:
    """``VarianceTerm`` with unit-order curvature (weight = 1 / max coeff scale^2)."""
    peak = float(np.max(coeff * scale[None, :] ** 2))
    return VarianceTerm(weight=1.0 / peak, coeff=coeff)


def _variance_spec(
    times: Array, scale: Array, coeff: Array, tie: float
) -> ObjectiveSpec:
    term = _normalised_variance_term(coeff, scale)
    return ObjectiveSpec(
        times_s=times, torque_scale=scale, effort_weight=tie, variance=term
    )


def _solve_convex(
    frames: tuple[FeasibleSet, ...],
    lay: Any,
    model: NoiseAwareModel,
    ctx: dict[str, Any],
) -> NoiseAwareResult:
    settings: NoiseAwareSettings = ctx["settings"]
    times, scale = ctx["times"], ctx["scale"]
    coeff = model.signal_coefficients(np.zeros(model.n_z))
    spec = _variance_spec(times, scale, coeff, settings.tie_break_effort_weight)
    try:
        if settings.effort_cap is None:
            res = allocate_trajectory(frames, spec)
            if res.status is not SolveStatus.OPTIMAL:
                return _from_status(res.status, res.reason)
            z_all, kkt, conv = _stack(res.z), res.kkt_residuals, res.convexity
            iters = res.iterations
        else:
            front = pareto_front(
                frames,
                (spec, _effort_spec(times, scale)),
                [settings.effort_cap],
            )
            pt = front.points[0]
            if pt.status is not SolveStatus.OPTIMAL:
                return _from_status(pt.status, pt.reason)
            z_all, kkt, conv, iters = _stack(pt.z), pt.kkt_residuals, None, 0
    except NotStrictlyConvexError as exc:
        raise NotStrictlyConvexError(
            f"{exc} Set NoiseAwareSettings.tie_break_effort_weight > 0 to select "
            "the minimum-effort member of the variance-optimal family explicitly."
        ) from exc
    unique = conv is None or conv.rank >= conv.required_rank
    tie_used = settings.tie_break_effort_weight > 0.0
    return _assemble_result(
        NoiseAwareStatus.OPTIMAL,
        lay,
        model,
        z_all,
        ctx,
        kkt_residuals=kkt,
        iterations=iters,
        convex=True,
        global_optimum_certified=not tie_used,
        unique_torque=bool(unique),
        tie_break_applied=tie_used,
        local_optima=(),
    )


def _from_status(status: SolveStatus, reason: str) -> NoiseAwareResult:
    if status is SolveStatus.INFEASIBLE:
        return _failure(NoiseAwareStatus.INFEASIBLE, reason or "infeasible family")
    return _failure(NoiseAwareStatus.SOLVER_FAILURE, reason or status.name)


# ------------------------------------------------------------ non-convex M2
def _constraint_data(
    lay: Any, effort_cone: SOCBlock | None
) -> tuple[Any, Any, tuple[SOCBlock, ...]]:
    mats, rhs, _, blocks, _ = _frame_constraints(lay, lay.n_z_total)
    if effort_cone is not None:
        blocks = [*blocks, effort_cone]
    a_ub = sp.vstack(mats).tocsr() if mats else None
    b_ub = np.concatenate(rhs) if rhs else None
    return a_ub, b_ub, tuple(blocks)


def _gn_subproblem(
    lay: Any,
    model: NoiseAwareModel,
    z: Array,
    mu: float,
    cons: tuple[Any, Any, tuple[SOCBlock, ...]],
) -> tuple[Array | None, KKTResiduals | None, str]:
    """Convex Gauss-Newton step ``min |r + J d|^2 + mu |d|^2`` over the family."""
    resid, jac = model.residuals(z, _tau(lay, z), lay.ntil.toarray())
    hess = jac.T @ jac + mu * np.eye(z.size)
    sol = solve_socp(
        QuadraticObjective(
            2.0 * (jac.T @ resid) - 2.0 * hess @ z, sp.csr_matrix(2.0 * hess)
        ),
        None,
        None,
        cons[0],
        cons[1],
        cons[2],
        settings=ConicSettings(tol=SOLVER_TOL),
    )
    if sol.status is not SolveStatus.OPTIMAL or sol.x is None:
        return None, None, sol.reason or sol.status.name
    return np.asarray(sol.x[: z.size]), sol.kkt_residuals, ""


def _mu_floor(lay: Any, model: NoiseAwareModel, z: Array) -> float:
    _, jac = model.residuals(z, _tau(lay, z), lay.ntil.toarray())
    return MU_FLOOR_REL * max(1.0, float(np.sum(jac**2)) / z.size)


def _scp(
    lay: Any,
    model: NoiseAwareModel,
    z0: Array,
    cons: tuple[Any, Any, tuple[SOCBlock, ...]],
    settings: NoiseAwareSettings,
) -> tuple[Array, int, float, KKTResiduals | None, str]:
    """Damped Gauss-Newton SCP; returns ``(z, iterations, step, kkt, reason)``."""
    z = z0.copy()
    value = model.variance(z, _tau(lay, z))
    scale_mu = 1.0
    for it in range(1, settings.max_iterations + 1):
        floor = _mu_floor(lay, model, z)
        accepted = False
        for _ in range(MU_MAX_STEPS):
            z_new, _, why = _gn_subproblem(lay, model, z, floor * scale_mu, cons)
            if z_new is None:
                return z, it, math.inf, None, why
            v_new = model.variance(z_new, _tau(lay, z_new))
            if v_new <= value + 1e-15 * (1.0 + value):
                accepted = True
                break
            scale_mu *= MU_GROWTH
        if not accepted:
            return z, it, math.inf, None, "no descent step at the damping limit"
        assert z_new is not None  # accepted implies a candidate step
        step = float(np.linalg.norm(z_new - z))
        z, value = z_new, min(v_new, value)
        scale_mu = max(1.0, scale_mu / MU_GROWTH)
        if step <= settings.step_tol * (1.0 + float(np.linalg.norm(z))):
            return _certify(lay, model, z, cons, settings, it)
    return z, settings.max_iterations, math.inf, None, "iteration limit reached"


def _certify(
    lay: Any,
    model: NoiseAwareModel,
    z: Array,
    cons: tuple[Any, Any, tuple[SOCBlock, ...]],
    settings: NoiseAwareSettings,
    iterations: int,
) -> tuple[Array, int, float, KKTResiduals | None, str]:
    """Re-solve at ``z`` with the floor damping: stationary iff the step is ~0.

    The damping only selects the minimum-norm step among optimal ones; at a
    first-order stationary point the step is zero for every ``mu >= 0``.
    """
    z_try, kkt, why = _gn_subproblem(lay, model, z, _mu_floor(lay, model, z), cons)
    if z_try is None:
        return z, iterations, math.inf, None, why
    step = float(np.linalg.norm(z_try - z))
    return z, iterations, step, kkt, ""


def _solve_nonconvex(
    frames: tuple[FeasibleSet, ...],
    lay: Any,
    model: NoiseAwareModel,
    ctx: dict[str, Any],
) -> NoiseAwareResult:
    settings: NoiseAwareSettings = ctx["settings"]
    cone = None
    if settings.effort_cap is not None:
        ops = _operators(
            lay, ctx["scale"], {"effort": 1.0, "rate": 0.0, "reference": 0.0}
        )
        m_pad, m_off = _bound_cone(lay, ops, None, lay.n_z_total)
        cone = _eps_block(m_pad, m_off, settings.effort_cap, lay.n_z_total)
    cons = _constraint_data(lay, cone)
    starts = [ctx["ref_z"], *[np.asarray(s, float) for s in settings.starts]]
    runs = []
    for z0 in starts:
        if z0.shape != (lay.n_z_total,):
            raise ValueError(f"start must have shape ({lay.n_z_total},)")
        runs.append(_scp(lay, model, z0, cons, settings))
    certified = [
        r
        for r in runs
        if r[4] == ""
        and r[2] <= settings.step_tol * (1.0 + float(np.linalg.norm(r[0])))
        and r[3] is not None
        and r[3].max() <= settings.kkt_tol
    ]
    values = [model.variance(r[0], _tau(lay, r[0])) for r in runs]
    if not certified:
        reasons = "; ".join(r[4] or "stationarity not certified" for r in runs)
        return _failure(NoiseAwareStatus.NOT_CONVERGED, reasons)
    best = min(certified, key=lambda r: model.variance(r[0], _tau(lay, r[0])))
    return _assemble_result(
        NoiseAwareStatus.LOCAL_STATIONARY,
        lay,
        model,
        best[0],
        ctx,
        kkt_residuals=best[3],
        step_norm=best[2],
        iterations=best[1],
        convex=False,
        global_optimum_certified=False,
        local_optima=tuple(sorted(values)),
    )


# ------------------------------------------------------------------- API
def minimise_face_variance(
    feasible_sets: Any,
    times_s: Any,
    torque_scale: Any,
    model: NoiseAwareModel,
    declaration: VarianceDeclaration,
    settings: NoiseAwareSettings | None = None,
) -> NoiseAwareResult:
    """Allocation of minimum linear face-angle variance over the feasible sets.

    Preconditions (``ValueError``/``TypeError``): frames, times and scales as in
    :func:`~.allocation.allocate_trajectory`; the model's steps, channels and
    stacked null dimension match the frames; the declaration names every
    mechanism the model actually has (M1 signal law, M2 preload slope).
    :class:`~.allocation_spec.NotStrictlyConvexError` when the variance has no
    curvature on a retained torque direction and no tie-break was requested.

    Postconditions: ``OBJECTIVE_FLAT_INVARIANT`` (no mechanism couples the
    allocation to the variance) returns the minimum-effort representative with
    an ``INVARIANT`` variance record and ``gradient_norm`` below ``flat_tol``;
    ``OPTIMAL`` (M1 only) is a convex global optimum with solver KKT residuals
    and ``variance <= reference_variance``; ``LOCAL_STATIONARY`` (M2) is a
    certified stationary point only (``global_optimum_certified`` is False) with
    ``variance <= reference_variance`` because the iteration starts at the
    reference.  ``INFEASIBLE`` is propagated, never relaxed.
    """
    settings = settings or NoiseAwareSettings()
    if not isinstance(model, NoiseAwareModel):
        raise TypeError("model must be a NoiseAwareModel")
    _check_declaration(model, declaration)
    times, scale = np.asarray(times_s, float), np.asarray(torque_scale, float)
    frames = _validated_frames(tuple(feasible_sets), _effort_spec(times, scale))
    lay = _layout_for(frames, times, scale, model)
    ref = allocate_trajectory(frames, _effort_spec(times, scale))
    if ref.status is not SolveStatus.OPTIMAL:
        return _from_status(ref.status, ref.reason)
    ref_z = _stack(ref.z)
    ctx: dict[str, Any] = {
        "frames": frames,
        "decl": declaration,
        "settings": settings,
        "times": times,
        "scale": scale,
        "ref_z": ref_z,
        "ref_tau": np.asarray(ref.torque),
    }
    resid, jac = model.residuals(ref_z, ctx["ref_tau"], lay.ntil.toarray())
    sd = math.sqrt(float(resid @ resid))
    flat = (not model.preload_active) and float(
        np.linalg.norm(jac)
    ) <= settings.flat_tol * max(1.0, sd)
    if flat:
        result = _assemble_result(
            NoiseAwareStatus.OBJECTIVE_FLAT_INVARIANT,
            lay,
            model,
            ref_z,
            ctx,
            kkt_residuals=ref.kkt_residuals,
            reason=(
                "no declared mechanism couples the allocation to the variance "
                "(V1); the reported allocation is the minimum-effort "
                "representative and carries no variance claim"
            ),
        )
        ensure(
            (result.gradient_norm or 0.0) <= settings.flat_tol * max(1.0, sd),
            "a flat objective must have a vanishing gradient",
        )
        return result
    if model.preload_active:
        return _solve_nonconvex(frames, lay, model, ctx)
    return _solve_convex(frames, lay, model, ctx)
