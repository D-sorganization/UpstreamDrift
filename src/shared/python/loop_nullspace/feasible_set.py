"""Declared feasible set in null-space coordinates (NSX-21, #12564).

The decomposition of one frame gives ``x = x_p + N z`` with
``x = (tau, lambda, f_g)``.  A :class:`FeasibleSet` is the set of ``z`` that
satisfies the constraints implied by a *declared* hypothesis set:

* **polyhedral rows** ``g . z <= h``: torque capacity (``H-TORQUE-CAPACITY``),
  the unilateral normal force and a ``k``-facet linearisation of every
  friction cone (``H-FRICTION-CONE``), and hand-contact squeeze floors
  (``H-GRIP-PRELOAD``);
* **conic blocks** ``b - A z`` in a second-order cone: the exact Coulomb cones.

Every row and block carries the id of the hypothesis it came from, so an
envelope, an allocation or a probe can name what it rests on.  The linearised
cone is an outer polygon (contains the exact cone, relative radius error
``1 / cos(pi/k) - 1``) or an inner polygon (apothem scaled by ``cos(pi/k)``,
relative shortfall ``1 - cos(pi/k)``); the facet count and the bound are
recorded on the set.  The exact cone is used by :meth:`FeasibleSet.to_socp`
and by the emptiness test.

Non-claims: the set is conditional on the declared hypotheses, torque
capacities are population priors, and a row that rests on an ASSUMED capacity
slot or on an unverified hypothesis is flagged (``flagged_rows``), never
silently presented as measured.  No objective is chosen here.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from src.shared.python.loop_nullspace.capacity import CapacityVector
from src.shared.python.loop_nullspace.decomposition import Decomposition
from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.ground_contact import (
    GroundObservation,
    observation_rows,
)
from src.shared.python.loop_nullspace.feasible_rows import (
    CONE_ID,
    DEFAULT_CONE_FACETS,
    FLAG_ASSUMED,
    FLAG_UNVERIFIED,
    GRF_ID,
    GRIP_ID,
    TORQUE_ID,
    ConeBlock,
    ContactCone,
    FeasibleRow,
    GripPreload,
    LPForm,
    SOCPForm,
    cone_error_bound,
)
from src.shared.python.loop_nullspace.registry import (
    HypothesisSet,
    load_registry,
)
from src.shared.python.loop_nullspace.solvers import (
    SolveStatus,
    SOCBlock,
    solve_lp,
    solve_socp,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ContactCone",
    "ConeBlock",
    "FeasibleRow",
    "FeasibleSet",
    "GripPreload",
    "LPForm",
    "SOCPForm",
    "cone_error_bound",
]

_UNSOUND_STATUSES = ("UNVERIFIED", "REMOVED")
_SLACK_TOL = 1e-9
_CONST_TOL = 1e-9


# ------------------------------------------------------------ feasibility
def _soc_ok(b: Array) -> bool:
    return bool(b[0] >= float(np.linalg.norm(b[1:])) - _CONST_TOL)


def _feasible(
    n_z: int, rows: Sequence[FeasibleRow], cones: Sequence[ConeBlock]
) -> bool:
    """Exact feasibility of the given items; raises on a numerical failure."""
    if n_z == 0:
        return all(r.h >= -_CONST_TOL for r in rows) and all(
            _soc_ok(c.b) for c in cones
        )
    g = np.array([r.g for r in rows]) if rows else None
    h = np.array([r.h for r in rows]) if rows else None
    zero = np.zeros(n_z)
    if cones:
        blocks = tuple(SOCBlock(c.A, c.b) for c in cones)
        res: Any = solve_socp(zero, None, None, g, h, blocks)
    else:
        res = solve_lp(zero, None, None, g, h, [(None, None)] * n_z)
    if res.status is SolveStatus.OPTIMAL:
        return True
    if res.status is SolveStatus.INFEASIBLE:
        return False
    raise RuntimeError(
        f"feasibility check inconclusive: {res.status.name} ({res.reason})"
    )


def _elastic_candidates(n_z: int, rows: Sequence[FeasibleRow]) -> list[FeasibleRow]:
    """Rows with positive slack in the elastic LP ``min sum s, G z - s <= h``."""
    m = len(rows)
    g = np.array([r.g for r in rows])
    a_ub = np.hstack([g, -np.eye(m)])
    c = np.concatenate([np.zeros(n_z), np.ones(m)])
    bounds = [(None, None)] * n_z + [(0.0, None)] * m
    res = solve_lp(c, None, None, a_ub, np.array([r.h for r in rows]), bounds)
    if res.status is not SolveStatus.OPTIMAL or res.x is None:
        return list(rows)
    return [r for r, s in zip(rows, res.x[n_z:], strict=True) if s > _SLACK_TOL]


def _deletion_filter(items: list[Any], infeasible: Any) -> list[Any]:
    """Inclusion-minimal infeasible subset of ``items`` (``infeasible`` monotone)."""
    kept = list(items)
    for item in list(items):
        trial = [i for i in kept if i is not item]
        if infeasible(trial):
            kept = trial
    return kept


# ------------------------------------------------------------ feasible set
@dataclass(frozen=True, eq=False)
class FeasibleSet:
    """Admissible ``z`` for one frame under declared hypotheses.

    Build with :meth:`from_hypotheses` (or :meth:`from_rows` for raw rows).
    Immutable; rows are read-only.  ``is_empty`` and the conflict queries use
    the *exact* cones, whereas ``to_lp`` uses the declared linearisation.
    """

    n_z: int
    rows: tuple[FeasibleRow, ...]
    cones: tuple[ConeBlock, ...] = ()
    hypothesis_ids: tuple[str, ...] = ()
    cone_facets: int | None = None
    cone_linearization: str | None = None
    cone_error_bound: float | None = None
    assumed_slots: tuple[str, ...] = ()
    unverified_hypothesis_ids: tuple[str, ...] = ()
    decomposition: Decomposition | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.n_z, bool) or not isinstance(self.n_z, int) or self.n_z < 0:
            raise ValueError(f"n_z must be a non-negative integer, got {self.n_z!r}")
        labels = [r.label for r in self.rows] + [c.label for c in self.cones]
        if len(set(labels)) != len(labels):
            dup = sorted({x for x in labels if labels.count(x) > 1})
            raise ValueError(f"duplicate row labels: {dup}")
        for r in self.rows:
            if r.g.shape != (self.n_z,):
                raise ValueError(
                    f"row {r.label!r} has length {r.g.size}, n_z={self.n_z}"
                )
        for c in self.cones:
            if c.A.shape[1] != self.n_z:
                raise ValueError(f"cone {c.label!r} has {c.A.shape[1]} columns")

    # ---------------------------------------------------------------- build
    @classmethod
    def from_rows(
        cls,
        n_z: int,
        rows: Sequence[FeasibleRow],
        cones: Sequence[ConeBlock] = (),
    ) -> FeasibleSet:
        """Wrap explicit rows; ``hypothesis_ids`` is taken from the items."""
        ids = tuple(
            dict.fromkeys(
                [r.hypothesis_id for r in rows] + [c.hypothesis_id for c in cones]
            )
        )
        return cls(n_z, tuple(rows), tuple(cones), ids)

    @classmethod
    def from_hypotheses(
        cls,
        decomposition: Decomposition,
        hypothesis_set: HypothesisSet,
        *,
        capacity: CapacityVector | None = None,
        cones: Sequence[ContactCone] = (),
        grip_preloads: Sequence[GripPreload] = (),
        cone_facets: int = DEFAULT_CONE_FACETS,
        linearization: str = "outer",
        observation: GroundObservation | None = None,
    ) -> FeasibleSet:
        """Express the declared constraints in the coordinates ``z`` of ``decomposition``.

        Preconditions: ``decomposition`` and ``hypothesis_set`` have the right
        types (``TypeError``); ``capacity`` is required when
        ``H-TORQUE-CAPACITY`` is declared and its actuated order matches the
        decomposition's torques; cone and grip data are accepted only when
        ``H-FRICTION-CONE`` / ``H-GRIP-PRELOAD`` is declared ("not declared"
        ``ValueError``); ``cone_facets >= 3``; an ``observation`` (ground-reaction
        data, NSX-60) only with ``H-GRF-OBSERVATION`` declared.
        Postconditions: every row and block names a declared hypothesis; rows
        resting on ASSUMED capacity slots or unverified hypotheses carry flags.
        """
        if not isinstance(decomposition, Decomposition):
            raise TypeError(
                f"decomposition must be Decomposition, got {type(decomposition)!r}"
            )
        if not isinstance(hypothesis_set, HypothesisSet):
            raise TypeError(
                f"hypothesis_set must be HypothesisSet, got {type(hypothesis_set)!r}"
            )
        declared = hypothesis_set.hypothesis_ids()
        _check_declared(declared, capacity, cones, grip_preloads, observation)
        err = cone_error_bound(cone_facets, linearization)
        unverified = _unverified_ids(hypothesis_set)
        rows: list[FeasibleRow] = []
        blocks: list[ConeBlock] = []
        if TORQUE_ID in declared:
            rows += _torque_rows(decomposition, capacity)
        if CONE_ID in declared:
            for cc in cones:
                r, b = _cone_items(decomposition, cc, cone_facets, linearization)
                rows += r
                blocks.append(b)
        if GRIP_ID in declared:
            rows += [_grip_row(decomposition, g) for g in grip_preloads]
        if GRF_ID in declared and observation is not None:
            rows += observation_rows(decomposition, observation)
        rows = [_with_unverified(r, unverified) for r in rows]
        blocks = [_with_unverified_block(b, unverified) for b in blocks]
        return cls(
            n_z=decomposition.nullity,
            rows=tuple(rows),
            cones=tuple(blocks),
            hypothesis_ids=tuple(declared),
            cone_facets=cone_facets if blocks else None,
            cone_linearization=linearization if blocks else None,
            cone_error_bound=err if blocks else None,
            assumed_slots=capacity.assumed_slots if capacity is not None else (),
            unverified_hypothesis_ids=tuple(i for i in declared if i in unverified),
            decomposition=decomposition,
        )

    # ------------------------------------------------------------- queries
    def flagged_rows(self) -> tuple[str, ...]:
        """Labels of rows and blocks resting on ASSUMED or unverified bases."""
        items: list[Any] = [*self.rows, *self.cones]
        return tuple(i.label for i in items if i.flags)

    @property
    def rests_on_unverified(self) -> bool:
        """True when any row rests on an ASSUMED capacity or unverified hypothesis."""
        return bool(self.flagged_rows())

    def to_lp(self) -> LPForm:
        """Polyhedral form (cones as the declared linearisation)."""
        return self._lp_form(self.rows)

    def to_socp(self) -> SOCPForm:
        """Polyhedral rows without linearised cones, plus the exact cone blocks."""
        keep = [r for r in self.rows if r.kind != "cone_linear"]
        form = self._lp_form(keep)
        blocks = tuple(SOCBlock(c.A, c.b) for c in self.cones)
        return SOCPForm(form.A_ub, form.b_ub, blocks, self.n_z)

    def _lp_form(self, rows: Sequence[FeasibleRow]) -> LPForm:
        g = np.array([r.g for r in rows]) if rows else np.zeros((0, self.n_z))
        h = np.array([r.h for r in rows])
        return LPForm(sp.csr_matrix(g), h, self.n_z)

    def is_empty(self) -> bool:
        """True when no ``z`` satisfies every row and exact cone.

        Reports emptiness; never raises for an infeasible set.  Raises
        ``RuntimeError`` only if the solver is inconclusive.
        """
        return not _feasible(self.n_z, self.rows, self.cones)

    def _infeasible(self, items: Sequence[Any]) -> bool:
        rows = [i for i in items if isinstance(i, FeasibleRow)]
        blocks = [i for i in items if isinstance(i, ConeBlock)]
        return not _feasible(self.n_z, rows, blocks)

    def _all_items(self) -> list[Any]:
        items: list[Any] = [*self.rows, *self.cones]
        return [
            i
            for i in items
            if not (isinstance(i, FeasibleRow) and i.kind == "cone_linear")
        ]

    def conflict_set(self) -> tuple[str, ...]:
        """Inclusion-minimal set of hypothesis ids whose rows are jointly infeasible.

        Empty tuple when the set is non-empty.  Minimal at hypothesis level:
        dropping any returned id (all its rows) leaves a feasible system.
        Linearised cone rows are excluded (the exact cone is tested).
        """
        if not self.is_empty():
            return ()
        items = self._all_items()
        groups = list(dict.fromkeys(i.hypothesis_id for i in items))

        def bad(ids: list[str]) -> bool:
            return self._infeasible([i for i in items if i.hypothesis_id in ids])

        return tuple(_deletion_filter(groups, bad))

    def conflict_rows(self) -> tuple[str, ...]:
        """Labels of an inclusion-minimal infeasible subsystem (row-level IIS).

        Found by the elastic filter (rows with positive slack) narrowed by a
        deletion filter, inside the rows of :meth:`conflict_set`.  Empty when
        the set is non-empty.
        """
        ids = self.conflict_set()
        if not ids:
            return ()
        items = [i for i in self._all_items() if i.hypothesis_id in ids]
        poly = [i for i in items if isinstance(i, FeasibleRow)]
        if not any(isinstance(i, ConeBlock) for i in items) and self.n_z:
            cand = _elastic_candidates(self.n_z, poly)
            if self._infeasible(cand):
                items = cand
        return tuple(i.label for i in _deletion_filter(items, self._infeasible))


# --------------------------------------------------------------- builders
def _check_declared(
    declared: Sequence[str],
    capacity: CapacityVector | None,
    cones: Sequence[ContactCone],
    grips: Sequence[GripPreload],
    observation: GroundObservation | None = None,
) -> None:
    if TORQUE_ID in declared and capacity is None:
        raise ValueError(f"{TORQUE_ID} is declared but no capacity was supplied")
    if TORQUE_ID not in declared and capacity is not None:
        raise ValueError(f"capacity supplied but {TORQUE_ID} not declared")
    if cones and CONE_ID not in declared:
        raise ValueError(f"cones supplied but {CONE_ID} not declared")
    if grips and GRIP_ID not in declared:
        raise ValueError(f"grip preloads supplied but {GRIP_ID} not declared")
    if observation is not None and GRF_ID not in declared:
        raise ValueError(f"observation supplied but {GRF_ID} not declared")


def _unverified_ids(hypothesis_set: HypothesisSet) -> frozenset[str]:
    """Ids that fail the same gate as ``Registry.require_verified_for_display``."""
    bad = set()
    reg = load_registry()
    for h in hypothesis_set.hypotheses:
        if h.verification_status in _UNSOUND_STATUSES:
            bad.add(h.id)
            continue
        try:
            reg.require_verified_for_display([h.id])
        except ValueError:
            bad.add(h.id)
    return frozenset(bad)


def _with_unverified(r: FeasibleRow, unverified: frozenset[str]) -> FeasibleRow:
    if r.hypothesis_id not in unverified:
        return r
    return FeasibleRow(
        r.g, r.h, r.label, r.hypothesis_id, r.kind, (*r.flags, FLAG_UNVERIFIED)
    )


def _with_unverified_block(b: ConeBlock, unverified: frozenset[str]) -> ConeBlock:
    if b.hypothesis_id not in unverified:
        return b
    return ConeBlock(b.A, b.b, b.label, b.hypothesis_id, (*b.flags, FLAG_UNVERIFIED))


def _affine(dec: Decomposition, selector: Array) -> tuple[Array, float]:
    """``selector . x = row . z + const`` for ``x = x_p + N z``."""
    return selector @ dec.N, float(selector @ dec.x_p)


def _torque_rows(dec: Decomposition, cap: CapacityVector | None) -> list[FeasibleRow]:
    if cap is None or len(cap.actuated_names) != dec.n_actuated:
        raise ValueError(
            f"capacity covers {0 if cap is None else len(cap.actuated_names)} actuated "
            f"coordinates, decomposition has {dec.n_actuated}"
        )
    out: list[FeasibleRow] = []
    for i, name in enumerate(cap.actuated_names):
        row, const = dec.N[i], float(dec.x_p[i])
        flags = (FLAG_ASSUMED,) if name in cap.assumed_slots else ()
        out.append(
            FeasibleRow(
                row,
                float(cap.tau_max_positive_nm[i]) - const,
                f"{TORQUE_ID}:{name}:upper",
                TORQUE_ID,
                "torque",
                flags,
            )
        )
        out.append(
            FeasibleRow(
                -row,
                float(cap.tau_max_negative_nm[i]) + const,
                f"{TORQUE_ID}:{name}:lower",
                TORQUE_ID,
                "torque",
                flags,
            )
        )
    return out


def _cone_items(
    dec: Decomposition, cc: ContactCone, k: int, linearization: str
) -> tuple[list[FeasibleRow], ConeBlock]:
    idx = list(cc.cone.indices)
    if max(idx) >= dec.dim_x:
        raise ValueError(f"cone {cc.name!r} index exceeds dim_x={dec.dim_x}")
    mu = cc.cone.mu
    frame = cc.cone.frame
    loc = frame.T @ dec.N[idx]  # (3, n_z): tangent, tangent, normal
    loc0 = frame.T @ dec.x_p[idx]
    base = f"{CONE_ID}:{cc.name}"
    rows = [
        FeasibleRow(-loc[2], float(loc0[2]), f"{base}:normal", CONE_ID, "cone_normal")
    ]
    scale = 1.0 if linearization == "outer" else math.cos(math.pi / k)
    for j in range(k):
        d = np.array([math.cos(2 * math.pi * j / k), math.sin(2 * math.pi * j / k)])
        rows.append(
            FeasibleRow(
                d @ loc[:2] - scale * mu * loc[2],
                float(scale * mu * loc0[2] - d @ loc0[:2]),
                f"{base}:facet{j}",
                CONE_ID,
                "cone_linear",
            )
        )
    block = ConeBlock(
        -np.vstack([mu * loc[2], loc[:2]]),
        np.concatenate([[mu * loc0[2]], loc0[:2]]),
        f"{base}:soc",
        CONE_ID,
    )
    return rows, block


def _grip_row(dec: Decomposition, grip: GripPreload) -> FeasibleRow:
    if grip.selector.shape != (dec.dim_x,):
        raise ValueError(
            f"grip {grip.name!r} selector has length {grip.selector.size}, "
            f"dim_x={dec.dim_x}"
        )
    row, const = _affine(dec, grip.selector)
    return FeasibleRow(
        -row,
        const + grip.offset - grip.minimum_squeeze_n,
        f"{GRIP_ID}:{grip.name}:squeeze",
        GRIP_ID,
        "grip",
    )
