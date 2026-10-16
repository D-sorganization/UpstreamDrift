"""Epsilon-constraint Pareto fronts over the trajectory feasible set (NSX-25, #12568).

Two objectives are given as two :class:`~.allocation_spec.ObjectiveSpec`
instances over the same frames; each objective is the weighted sum of the
quadratic terms that spec enables (capacity-weighted effort, torque-rate
smoothness, reference tracking, and, for the *minimised* objective only, the
gross-power epigraph)::

    minimise  J_a(z)   subject to  J_b(z) <= eps,  z in the feasible sets.

``J_b = |M z + m|^2`` is imposed *exactly* as the second-order cone
``(sqrt(eps), M z + m) in SOC`` (no weight sweep, no penalty).  ``J_a`` must be
strictly convex on the retained torque space (the NSX-23 precondition), so each
solve has a unique torque history and, by the standard epsilon-constraint
result, every OPTIMAL point is Pareto optimal in the *torque* objectives.  The
multiplier of the cone gives the front's slope ``d J_a* / d eps``.

This replaces the weight sweep that ``explore_torque_tradeoffs`` calls a Pareto
front: a weighted sum cannot reach the non-convex part of a front and its
weights are not a coordinate on it.

Non-claims: every point is ``FAMILY_DEPENDENT`` (conditional on the feasible-set
hypotheses and on both objectives' priors); a point is Pareto optimal among
admissible torque histories, not an estimate of what an athlete does.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from src.shared.python.loop_nullspace.allocation import (
    SOLVER_TOL,
    _assemble,
    _check_convexity,
    _classify_power,
    _hypotheses,
    _layout,
    _objective_terms,
    _operators,
    _record,
    _validated_frames,
    _validated_reference,
    _weights,
)
from src.shared.python.loop_nullspace.allocation_spec import ObjectiveSpec
from src.shared.python.loop_nullspace.evidence import Array, QuantityRecord
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.solvers import (
    ConicSettings,
    KKTResiduals,
    QuadraticObjective,
    SolveStatus,
    SOCBlock,
    solve_socp,
)
from src.shared.python.loop_nullspace.trajectory_system import unverified_ids

logger = logging.getLogger(__name__)

__all__ = ["ParetoFront", "ParetoPoint", "pareto_front"]

DOMINANCE_RTOL = 1e-8
ACTIVE_RTOL = 1e-6
NON_CLAIM = (
    "Each point is Pareto optimal among admissible torque histories under the "
    "declared hypotheses and both objectives' priors; it is not an estimate of "
    "what an athlete does, and the front changes with the hypotheses."
)


@dataclass(frozen=True, eq=False)
class ParetoPoint:
    """One epsilon-constraint solve.

    On a non-OPTIMAL ``status`` only ``eps``, ``status`` and ``reason`` carry
    data.  ``primary`` is the minimised objective ``J_a``, ``constrained`` the
    bounded one ``J_b`` (``<= eps``), ``active`` whether the bound binds and
    ``slope`` the front's derivative ``d J_a / d eps`` (``None`` when
    ``eps == 0``).  ``allocation_dependence`` is always ``FAMILY_DEPENDENT``
    for an OPTIMAL point and carries ``hypotheses``.
    """

    eps: float
    status: SolveStatus
    reason: str = ""
    primary: float | None = None
    constrained: float | None = None
    active: bool = False
    slope: float | None = None
    z: tuple[Array, ...] = ()
    x: tuple[Array, ...] = ()
    torque: Array | None = None
    kkt_residuals: KKTResiduals | None = None
    record: QuantityRecord | None = None
    hypotheses: tuple[str, ...] = ()
    dominated: bool = False

    @property
    def allocation_dependence(self) -> str:
        """``FAMILY_DEPENDENT`` for a solved point, else ``UNAVAILABLE``."""
        return (
            "FAMILY_DEPENDENT" if self.status is SolveStatus.OPTIMAL else "UNAVAILABLE"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-able dict (arrays as float lists)."""

        def lst(a: Array | None) -> list[float] | None:
            return None if a is None else np.asarray(a, float).ravel().tolist()

        return {
            "eps": self.eps,
            "status": self.status.name,
            "reason": self.reason,
            "primary": self.primary,
            "constrained": self.constrained,
            "active": self.active,
            "slope": self.slope,
            "z": [lst(a) for a in self.z],
            "x": [lst(a) for a in self.x],
            "torque": None if self.torque is None else np.asarray(self.torque).tolist(),
            "allocation_dependence": self.allocation_dependence,
            "hypotheses": list(self.hypotheses),
            "dominated": self.dominated,
        }


@dataclass(frozen=True, eq=False)
class ParetoFront:
    """Epsilon-constraint points in ascending ``eps`` (see :class:`ParetoPoint`)."""

    points: tuple[ParetoPoint, ...]
    primary_terms: tuple[str, ...]
    constrained_terms: tuple[str, ...]
    hypotheses: tuple[str, ...]
    unverified_hypothesis_ids: tuple[str, ...]
    non_claim: str = NON_CLAIM

    def solved(self) -> tuple[ParetoPoint, ...]:
        """The OPTIMAL points."""
        return tuple(p for p in self.points if p.status is SolveStatus.OPTIMAL)

    def nondominated(self) -> tuple[ParetoPoint, ...]:
        """Solved points not dominated by another solved point."""
        return tuple(p for p in self.solved() if not p.dominated)

    def is_monotone(self, *, rtol: float = DOMINANCE_RTOL) -> bool:
        """``J_a*`` is non-increasing and ``J_b`` within ``eps`` along ``eps``."""
        pts = self.solved()
        prim = [p.primary for p in pts]
        if any(v is None for v in prim):
            raise ValueError("solved Pareto points must carry a primary objective")
        floats = [float(v) for v in prim if v is not None]
        ok = all(
            nxt <= cur + rtol * (1.0 + abs(cur))
            for cur, nxt in zip(floats, floats[1:], strict=False)
        )
        return ok and all(
            p.constrained <= p.eps + rtol * (1.0 + p.eps)  # type: ignore[operator]
            for p in pts
        )

    def to_json(self) -> str:
        """Deterministic JSON text."""
        return json.dumps(
            {
                "schema": "loop-nullspace-pareto/1",
                "points": [p.to_dict() for p in self.points],
                "primary_terms": list(self.primary_terms),
                "constrained_terms": list(self.constrained_terms),
                "hypotheses": list(self.hypotheses),
                "unverified_hypothesis_ids": list(self.unverified_hypothesis_ids),
                "non_claim": self.non_claim,
            },
            sort_keys=True,
            allow_nan=False,
        )


# ------------------------------------------------------------ validation
def _two_specs(objectives: Any) -> tuple[ObjectiveSpec, ObjectiveSpec]:
    try:
        a, b = objectives
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "objectives must be (minimised ObjectiveSpec, constrained ObjectiveSpec)"
        ) from exc
    for spec in (a, b):
        if not isinstance(spec, ObjectiveSpec):
            raise TypeError(f"objectives must be ObjectiveSpec, got {type(spec)!r}")
    if b.gross_power is not None:
        raise ValueError(
            "the constrained objective may not include gross power (it is "
            "piecewise linear; only quadratic terms are imposed as a cone)"
        )
    if not (
        a.times_s.shape == b.times_s.shape and np.array_equal(a.times_s, b.times_s)
    ):
        raise ValueError("both objectives must use the same times_s")
    if a.torque_scale.shape != b.torque_scale.shape:
        raise ValueError("both objectives must have the same number of torque scales")
    return a, b


def _eps_grid(eps_grid: Any) -> Array:
    eps = np.asarray(eps_grid, dtype=float)
    if eps.ndim != 1 or eps.size == 0:
        raise ValueError("eps_grid must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(eps)) or np.any(eps < 0.0):
        raise ValueError("eps_grid entries must be finite and non-negative")
    eps = np.sort(eps)
    if np.any(np.diff(eps) <= 0.0):
        raise ValueError("eps_grid entries must be distinct")
    return eps


def _bound_cone(lay: Any, ops: dict[str, Any], ref: Array | None, n_var: int):
    """``(M, m)`` with ``J_b(z) = |M z + m|^2`` padded to ``n_var`` columns."""
    mats, offs = [], []
    for name, op in ops.items():
        target = np.zeros(op.shape[0])
        if name == "reference" and ref is not None:
            target = op @ ref.reshape(-1)
        mats.append(op @ lay.ntil)
        offs.append(op @ lay.t0 - target)
    m_mat = sp.vstack(mats).tocsr()
    pad = sp.csr_matrix((m_mat.shape[0], n_var - m_mat.shape[1]))
    return sp.hstack([m_mat, pad]).tocsr(), np.concatenate(offs)


def _eps_block(m_pad: sp.csr_matrix, m_off: Array, eps: float, n_var: int) -> SOCBlock:
    top = sp.csr_matrix((1, n_var))
    return SOCBlock(
        sp.vstack([top, -m_pad]).tocsc(), np.concatenate([[np.sqrt(eps)], m_off])
    )


def _mark_dominated(points: list[ParetoPoint]) -> list[ParetoPoint]:
    solved = [p for p in points if p.status is SolveStatus.OPTIMAL]

    def le(u: float, v: float) -> bool:
        return u <= v + DOMINANCE_RTOL * (1.0 + abs(v))

    def lt(u: float, v: float) -> bool:
        return u < v - DOMINANCE_RTOL * (1.0 + abs(v))

    out = []
    for p in points:
        dom = p.status is SolveStatus.OPTIMAL and any(
            q is not p
            and le(q.primary, p.primary)  # type: ignore[arg-type]
            and le(q.constrained, p.constrained)  # type: ignore[arg-type]
            and (lt(q.primary, p.primary) or lt(q.constrained, p.constrained))  # type: ignore[arg-type]
            for q in solved
        )
        out.append(_with_dominated(p, dom))
    return out


def _with_dominated(p: ParetoPoint, dominated: bool) -> ParetoPoint:
    from dataclasses import replace

    return replace(p, dominated=dominated)


# ------------------------------------------------------------------- API
def pareto_front(
    feasible_sets: Sequence[FeasibleSet],
    objectives: Sequence[ObjectiveSpec],
    eps_grid: Any,
    *,
    references: Sequence[Any] = (None, None),
) -> ParetoFront:
    """Epsilon-constraint front: minimise ``J_a`` subject to ``J_b <= eps``.

    ``objectives`` is ``(minimised, constrained)``; ``eps_grid`` a set of
    distinct non-negative bounds on ``J_b``; ``references`` the optional
    ``(frames, n_a)`` torque references of the two specs.

    Preconditions (``ValueError``/``TypeError`` otherwise): the NSX-23
    preconditions of :func:`~.allocation.allocate_trajectory` for the minimised
    spec (including strict convexity, :class:`~.allocation_spec.NotStrictlyConvexError`),
    a constrained spec with at least one enabled quadratic term, matching
    ``times_s`` and torque scales, no gross power in the constrained spec.
    Postconditions: points are in ascending ``eps``; an ``eps`` below the
    smallest attainable ``J_b`` is reported ``INFEASIBLE``, never relaxed; on
    OPTIMAL points ``J_b <= eps`` (up to solver tolerance), ``J_a*`` is
    non-increasing in ``eps``, no solved point is dominated and each carries a
    ``FAMILY_DEPENDENT`` record naming every hypothesis of both objectives.
    """
    primary, constrained = _two_specs(objectives)
    frames = _validated_frames(feasible_sets, primary)
    if len(references) != 2:
        raise ValueError("references must be a pair (ref_primary, ref_constrained)")
    ref_a = _validated_reference(references[0], primary, len(frames))
    ref_b = _validated_reference(references[1], constrained, len(frames))
    eps_values = _eps_grid(eps_grid)
    lay = _layout(frames, primary)
    ops_a = _operators(lay, primary.torque_scale, _weights(primary), primary.variance)
    _check_convexity(lay, primary, ops_a)
    ops_b = _operators(
        lay, constrained.torque_scale, _weights(constrained), constrained.variance
    )
    if not ops_b:
        raise ValueError(
            "the constrained objective has no enabled quadratic term (rate needs "
            "at least two frames)"
        )
    power = _classify_power(lay, primary.gross_power)
    prob = _assemble(lay, primary, ops_a, ref_a, power)
    m_pad, m_off = _bound_cone(lay, ops_b, ref_b, prob.n_var)
    hyps = tuple(
        dict.fromkeys((*_hypotheses(lay, primary), *_hypotheses(lay, constrained)))
    )
    points = [
        _solve_point(
            float(e),
            prob,
            lay,
            (primary, constrained),
            (ref_a, ref_b),
            (m_pad, m_off),
            hyps,
        )
        for e in eps_values
    ]
    return ParetoFront(
        points=tuple(_mark_dominated(points)),
        primary_terms=_term_names(primary, ref_a),
        constrained_terms=_term_names(constrained, ref_b),
        hypotheses=hyps,
        unverified_hypothesis_ids=unverified_ids(hyps),
    )


def _term_names(spec: ObjectiveSpec, ref: Array | None) -> tuple[str, ...]:
    names = []
    if spec.effort_weight > 0:
        names.append("effort")
    if spec.rate_weight > 0:
        names.append("rate")
    if spec.reference_weight > 0 and ref is not None:
        names.append("reference")
    if spec.variance is not None:
        names.append("variance")
    if spec.gross_power is not None:
        names.append("gross_power")
    return tuple(names)


def _solve_point(
    eps: float,
    prob: Any,
    lay: Any,
    specs: tuple[ObjectiveSpec, ObjectiveSpec],
    refs: tuple[Array | None, Array | None],
    cone_data: tuple[sp.csr_matrix, Array],
    hyps: tuple[str, ...],
) -> ParetoPoint:
    block = _eps_block(cone_data[0], cone_data[1], eps, prob.n_var)
    has_rows = prob.a_ub.shape[0] > 0
    sol = solve_socp(
        QuadraticObjective(prob.c_vec, prob.p_mat),
        None,
        None,
        prob.a_ub if has_rows else None,
        prob.b_ub if has_rows else None,
        (*prob.cones, block),
        settings=ConicSettings(tol=SOLVER_TOL),
    )
    if sol.status is not SolveStatus.OPTIMAL or sol.x is None or sol.z is None:
        return ParetoPoint(eps, sol.status, reason=sol.reason or sol.status.name)
    z_all = np.asarray(sol.x[: lay.n_z_total])
    z_list = tuple(
        z_all[int(lay.offsets[k]) : int(lay.offsets[k + 1])].copy()
        for k in range(lay.t_count)
    )
    x_list = tuple(
        fs.decomposition.x(z) for fs, z in zip(lay.frames, z_list, strict=True)
    )
    tau = np.array([x[: lay.n_a] for x in x_list])
    j_a = sum(_objective_terms(tau, specs[0], lay, refs[0]).values())
    j_b = sum(_objective_terms(tau, specs[1], lay, refs[1]).values())
    w0 = float(np.asarray(sol.z)[-block.dim])
    slope = -w0 / (2.0 * np.sqrt(eps)) if eps > 0.0 else None
    logger.info("NSX-25 pareto eps=%.6g: J_a=%.6g J_b=%.6g", eps, j_a, j_b)
    return ParetoPoint(
        eps=eps,
        status=SolveStatus.OPTIMAL,
        primary=float(j_a),
        constrained=float(j_b),
        active=bool(j_b >= eps - ACTIVE_RTOL * (1.0 + eps)),
        slope=None if slope is None else float(slope),
        z=z_list,
        x=x_list,
        torque=tau,
        kkt_residuals=sol.kkt_residuals,
        record=_record(lay, specs[0], tau, hyps),
        hypotheses=hyps,
    )
