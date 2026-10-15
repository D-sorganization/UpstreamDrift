"""Static torque and hand-split envelopes over the feasible set (NSX-22, #12565).

For every frame the :class:`~.feasible_set.FeasibleSet` is a polyhedron in the
null-space coordinates ``z``.  A *quantity* is an affine functional of ``z``
(a joint torque, a hand-split coordinate, any selected entry of
``x = x_p + N z``): ``q(z) = const + coeff . z``.  The envelope of ``q`` is the
pair ``[min q, max q]`` over the set, found by two linear programs per
quantity per frame on one persistent :class:`LPSession` (objective swaps reuse
the simplex basis; the basis also carries over to the next frame when the
dimensions match).

Conventions and non-claims:

* Status per bound is ``BOUNDED``, ``UNBOUNDED`` or ``INFEASIBLE``.  An
  unbounded or infeasible bound is **NaN**, never a large number.
* The series label is ``STATIC``: every frame is solved on its own and
  trajectory coupling is not represented.  When smoothness priors or a
  trajectory model are active, a static envelope over-approximates the
  trajectory-feasible range (NSX-25 owns the coupled case).
* The sign class (``POSITIVE``/``NEGATIVE``/``ZERO``/``INDETERMINATE``) is
  decided with the explicit absolute tolerance
  ``SIGN_ATOL_RTOL * scale``, where ``scale`` is the quantity's characteristic
  scale (``C-SCALE-FORCE-MOMENT``) and ``SIGN_ATOL_RTOL`` equals the NSX-00
  scaled balance threshold; never a bare ``0``.  A bound that only touches
  zero is ``INDETERMINATE``.
* The Chebyshev margin is the radius of the largest ball inside the set in the
  declared metric ``{z : z^T W z <= r^2}`` (``W`` diagonal, default identity).
  It is a margin to the *nearest* declared row, not a probability.
* Cones enter through the declared linearisation of the set
  (:meth:`FeasibleSet.to_lp`); an outer polygon gives bounds that contain the
  exact-cone bounds, an inner polygon bounds contained by them.
* Each bound lists the rows active at its optimum (row slack within
  ``ACTIVE_RTOL``) and the hypotheses they came from, and whether any of
  those rows rests on an ASSUMED capacity or an unverified hypothesis.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import scipy.sparse as sp

from src.shared.python.loop_nullspace.decomposition import BALANCE_TOL
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
    readonly_float,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.solvers import (
    Basis,
    LPResult,
    LPSession,
    SolveStatus,
    solve_lp,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ACTIVE_RTOL",
    "SIGN_ATOL_RTOL",
    "BoundStatus",
    "ChebyshevResult",
    "EnvelopeQuantity",
    "EnvelopeSeries",
    "SignClass",
    "chebyshev_margin",
    "classify_sign",
    "joint_torque_quantities",
    "static_envelopes",
    "x_functional_quantities",
]

SERIES_LABEL = "STATIC"
SIGN_ATOL_RTOL = BALANCE_TOL
ACTIVE_RTOL = 1e-8
STATIC_CONVENTION = (
    "STATIC: per-frame feasible sets; trajectory coupling is not represented"
)

QuantityFactory = Callable[[FeasibleSet], Sequence["EnvelopeQuantity"]]


class BoundStatus(Enum):
    """Termination status of one bound."""

    BOUNDED = "bounded"
    UNBOUNDED = "unbounded"
    INFEASIBLE = "infeasible"


class SignClass(Enum):
    """Sign determinacy of a quantity over its envelope."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    INDETERMINATE = "indeterminate"
    ZERO = "zero"


def classify_sign(lower: float, upper: float, *, scale: float) -> SignClass:
    """Sign class of a quantity known to lie in ``[lower, upper]``.

    A NaN bound is unknown (unbounded side).  Decisions use
    ``atol = SIGN_ATOL_RTOL * scale``: ``ZERO`` when both bounds are within
    ``atol`` of 0, ``POSITIVE`` when ``lower > atol``, ``NEGATIVE`` when
    ``upper < -atol``, otherwise ``INDETERMINATE`` (including a bound that
    merely touches zero).  Raises ``ValueError`` for a non-positive ``scale``
    or ``lower > upper + atol``.
    """
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be positive and finite, got {scale!r}")
    atol = SIGN_ATOL_RTOL * scale
    lo_known, hi_known = not math.isnan(lower), not math.isnan(upper)
    if lo_known and hi_known and lower > upper + atol:
        raise ValueError(f"lower {lower!r} exceeds upper {upper!r}")
    if lo_known and hi_known and abs(lower) <= atol and abs(upper) <= atol:
        return SignClass.ZERO
    if lo_known and lower > atol:
        return SignClass.POSITIVE
    if hi_known and upper < -atol:
        return SignClass.NEGATIVE
    return SignClass.INDETERMINATE


# --------------------------------------------------------------- quantities
@dataclass(frozen=True, eq=False)
class EnvelopeQuantity:
    """Affine functional ``q(z) = const + coeff . z`` with units and a scale.

    ``scale`` is the characteristic magnitude (force or moment scale) that
    sets the sign tolerance.  Raises ``ValueError`` for an empty name or units,
    non-finite data, or a non-positive scale.
    """

    name: str
    coeff: Array
    const: float
    units: str
    scale: float

    def __post_init__(self) -> None:
        if not self.name or not self.units:
            raise ValueError("EnvelopeQuantity needs a name and units")
        coeff = readonly_float(self.coeff, f"coeff of {self.name}")
        if coeff.ndim != 1:
            raise ValueError(f"coeff of {self.name!r} must be 1-D")
        if not math.isfinite(self.const):
            raise ValueError(f"const of {self.name!r} must be finite")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(f"scale of {self.name!r} must be positive and finite")
        object.__setattr__(self, "coeff", coeff)
        object.__setattr__(self, "const", float(self.const))
        object.__setattr__(self, "scale", float(self.scale))


def x_functional_quantities(
    names: Sequence[str],
    selectors: Any,
    *,
    offsets: Sequence[float] | None = None,
    units: str,
    scale: float,
) -> QuantityFactory:
    """Factory of quantities ``s_k . x + offset_k`` for ``x = x_p + N z``.

    ``selectors`` is ``(len(names), dim_x)``.  Joint torques are rows of the
    identity; a hand-split coordinate uses its row of ``Phi`` composed with the
    affine closing-hand map.  The factory reads the frame's decomposition, so it
    raises ``ValueError`` for a set built without one.
    """
    sel = readonly_float(selectors, "selectors")
    if sel.ndim != 2 or sel.shape[0] != len(names):
        raise ValueError("selectors must be (len(names), dim_x)")
    off = np.zeros(len(names)) if offsets is None else np.asarray(offsets, float)
    if off.shape != (len(names),):
        raise ValueError("offsets must have one entry per name")

    def build(fs: FeasibleSet) -> list[EnvelopeQuantity]:
        dec = _require_decomposition(fs)
        if sel.shape[1] != dec.dim_x:
            raise ValueError(
                f"selectors have {sel.shape[1]} columns, dim_x={dec.dim_x}"
            )
        return [
            EnvelopeQuantity(
                n, sel[k] @ dec.N, float(sel[k] @ dec.x_p + off[k]), units, scale
            )
            for k, n in enumerate(names)
        ]

    return build


def joint_torque_quantities(names: Sequence[str] | None = None) -> QuantityFactory:
    """Factory of the actuated joint torques ``tau_i`` (N m, moment scale)."""

    def build(fs: FeasibleSet) -> list[EnvelopeQuantity]:
        dec = _require_decomposition(fs)
        n_a = dec.n_actuated
        labels = (
            tuple(names)
            if names is not None
            else tuple(f"tau_{i + 1}" for i in range(n_a))
        )
        if len(labels) != n_a:
            raise ValueError(f"{len(labels)} names for {n_a} actuated torques")
        return [
            EnvelopeQuantity(lab, dec.N[i], float(dec.x_p[i]), "N m", dec.scales.moment)
            for i, lab in enumerate(labels)
        ]

    return build


def _require_decomposition(fs: FeasibleSet) -> Any:
    if not isinstance(fs, FeasibleSet):
        raise TypeError(f"expected FeasibleSet, got {type(fs).__name__}")
    if fs.decomposition is None:
        raise ValueError("this FeasibleSet carries no decomposition")
    return fs.decomposition


# ---------------------------------------------------------------- Chebyshev
@dataclass(frozen=True, eq=False)
class ChebyshevResult:
    """Largest inscribed ball: ``radius`` and ``center`` (NaN unless BOUNDED)."""

    radius: float
    center: Array
    status: BoundStatus
    metric_label: str


def _metric_weights(weights: Sequence[float] | None, n: int) -> tuple[Array, str]:
    if weights is None:
        return np.ones(n), "euclidean"
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"metric_weights must have length {n}, got {w.shape}")
    if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("metric_weights must be positive and finite")
    return w, "diag(" + ",".join(f"{x:g}" for x in w) + ")"


def chebyshev_margin(
    feasible_set: FeasibleSet, *, metric_weights: Sequence[float] | None = None
) -> ChebyshevResult:
    """Radius of the largest ball in ``{z : z^T W z <= r^2}`` inside the set.

    ``W = diag(metric_weights)`` (default identity).  Solves
    ``max r : g_i z + r sqrt(g_i W^-1 g_i) <= h_i``.  ``INFEASIBLE`` when the
    set is empty, ``UNBOUNDED`` (radius NaN) when balls of any size fit.  A
    radius of 0 means the set has no interior in this metric.  ``n_z = 0``
    gives radius 0 for a non-empty set.
    """
    if not isinstance(feasible_set, FeasibleSet):
        raise TypeError(f"expected FeasibleSet, got {type(feasible_set).__name__}")
    n = feasible_set.n_z
    w, label = _metric_weights(metric_weights, n)
    form = feasible_set.to_lp()
    g = form.A_ub.toarray() if form.A_ub.shape[0] else np.zeros((0, n))
    norms = np.sqrt((g**2 / w).sum(axis=1)) if n else np.zeros(g.shape[0])
    nan_center = np.full(n, np.nan)
    if n == 0:
        ok = bool(np.all(form.b_ub >= -BALANCE_TOL))
        status = BoundStatus.BOUNDED if ok else BoundStatus.INFEASIBLE
        return ChebyshevResult(0.0 if ok else math.nan, nan_center, status, label)
    a = sp.csr_matrix(np.hstack([g, norms[:, None]])) if g.shape[0] else None
    cost = np.zeros(n + 1)
    cost[-1] = -1.0
    res = solve_lp(
        cost,
        None,
        None,
        a,
        form.b_ub if g.shape[0] else None,
        [(None, None)] * n + [(0.0, None)],
    )
    return _chebyshev_result(res, n, label)


def _chebyshev_result(res: LPResult, n: int, label: str) -> ChebyshevResult:
    nan_center = np.full(n, np.nan)
    if res.status is SolveStatus.OPTIMAL and res.x is not None:
        center = _ro(res.x[:n])
        return ChebyshevResult(float(res.x[n]), center, BoundStatus.BOUNDED, label)
    if res.status is SolveStatus.UNBOUNDED:
        return ChebyshevResult(math.nan, nan_center, BoundStatus.UNBOUNDED, label)
    if res.status is SolveStatus.INFEASIBLE:
        return ChebyshevResult(math.nan, nan_center, BoundStatus.INFEASIBLE, label)
    raise RuntimeError(f"Chebyshev LP inconclusive: {res.status.name} ({res.reason})")


def _ro(values: Any) -> Array:
    """Read-only float copy that keeps NaN (unbounded / infeasible markers)."""
    arr = np.array(values, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


# ------------------------------------------------------------ per-bound data
@dataclass(frozen=True)
class _Bound:
    value: float
    status: BoundStatus
    rows: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    unverified: bool = False
    iterations: int = 0


def _nan_bound(status: BoundStatus, iters: int = 0) -> _Bound:
    return _Bound(math.nan, status, iterations=iters)


def _binding(
    fs: FeasibleSet, x: Array
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """Rows active at ``x``, their hypothesis ids and whether any is flagged."""
    labels, hyps, flagged = [], [], False
    for r in fs.rows:
        slack = r.h - float(r.g @ x)
        if slack <= ACTIVE_RTOL * (1.0 + abs(r.h)):
            labels.append(r.label)
            hyps.append(r.hypothesis_id)
            flagged = flagged or bool(r.flags)
    return tuple(labels), tuple(dict.fromkeys(hyps)), flagged


class _FrameSolver:
    """One persistent LP model of one frame plus the cross-frame basis."""

    def __init__(self, fs: FeasibleSet, *, reuse: bool) -> None:
        self.fs = fs
        self.reuse = reuse
        form = fs.to_lp()
        has_rows = form.A_ub.shape[0] > 0
        self.session = LPSession(
            None,
            None,
            form.A_ub if has_rows else None,
            form.b_ub if has_rows else None,
            [(None, None)] * fs.n_z,
        )
        self.iterations = 0

    def close(self) -> None:
        self.session.close()

    def _solve(self, c: Array, warm: Basis | None = None) -> LPResult:
        if not self.reuse:
            self.session.reset()
        res = self.session.solve(c, warm=warm)
        self.iterations += res.iterations
        return res

    def feasibility(self, warm: Basis | None) -> tuple[LPResult, bool]:
        """Solve with zero cost; falls back to a cold start on a bad basis."""
        zero = np.zeros(self.fs.n_z)
        try:
            res = self._solve(zero, warm if self.reuse else None)
        except ValueError as exc:  # basis from a different model shape
            logger.debug("warm basis rejected, solving cold: %s", exc)
            self.session.reset()
            res = self._solve(zero)
        if res.status is SolveStatus.INFEASIBLE:
            return res, False
        if res.status is not SolveStatus.OPTIMAL:
            raise RuntimeError(
                f"feasibility solve inconclusive: {res.status.name} ({res.reason})"
            )
        return res, True

    def bound(self, q: EnvelopeQuantity, sign: float) -> _Bound:
        """``sign=+1`` minimises ``q`` (lower bound); ``-1`` gives the upper."""
        res = self._solve(sign * q.coeff)
        if res.status is SolveStatus.UNBOUNDED:
            return _nan_bound(BoundStatus.UNBOUNDED, res.iterations)
        if res.status is not SolveStatus.OPTIMAL or res.x is None:
            raise RuntimeError(
                f"bound LP for {q.name!r} inconclusive: {res.status.name} "
                f"({res.reason})"
            )
        value = q.const + sign * float(res.objective or 0.0)
        rows, hyps, flagged = _binding(self.fs, res.x)
        return _Bound(value, BoundStatus.BOUNDED, rows, hyps, flagged, res.iterations)


def _constant_bound(q: EnvelopeQuantity) -> _Bound:
    return _Bound(q.const, BoundStatus.BOUNDED)


def _solve_point(
    fs: FeasibleSet, qs: Sequence[EnvelopeQuantity]
) -> list[tuple[_Bound, _Bound]]:
    """``n_z = 0``: the set is the single point ``z = ()`` or empty."""
    if all(r.h >= -BALANCE_TOL for r in fs.rows):
        return [(_constant_bound(q), _constant_bound(q)) for q in qs]
    bad = _nan_bound(BoundStatus.INFEASIBLE)
    return [(bad, bad) for _ in qs]


def _solve_frame(
    fs: FeasibleSet, qs: Sequence[EnvelopeQuantity], warm: Basis | None, reuse: bool
) -> tuple[list[tuple[_Bound, _Bound]], Basis | None, int]:
    if fs.n_z == 0:
        return _solve_point(fs, qs), warm, 0
    solver = _FrameSolver(fs, reuse=reuse)
    try:
        feas, ok = solver.feasibility(warm)
        if not ok:
            bad = _nan_bound(BoundStatus.INFEASIBLE)
            return [(bad, bad) for _ in qs], None, solver.iterations
        out = []
        for q in qs:
            if not np.any(q.coeff):
                out.append((_constant_bound(q), _constant_bound(q)))
            else:
                out.append((solver.bound(q, +1.0), solver.bound(q, -1.0)))
        return out, feas.basis, solver.iterations
    finally:
        solver.close()


# ------------------------------------------------------------------- series
@dataclass(frozen=True, eq=False)
class EnvelopeSeries:
    """Per-frame envelopes; every table is indexed ``[frame][quantity]``.

    ``lower``/``upper`` are NaN wherever the status is not ``BOUNDED``.
    ``sign_class`` is ``None`` for an infeasible frame.  ``binding_*`` list the
    active row labels at each bound, ``binding_hypotheses_*`` the hypotheses
    they came from, and ``*_rests_on_unverified`` whether any of those rows
    rests on an ASSUMED capacity or unverified hypothesis.
    """

    names: tuple[str, ...]
    units: tuple[str, ...]
    lower: Array
    upper: Array
    status_lower: tuple[tuple[BoundStatus, ...], ...]
    status_upper: tuple[tuple[BoundStatus, ...], ...]
    sign_class: tuple[tuple[SignClass | None, ...], ...]
    binding_lower: tuple[tuple[tuple[str, ...], ...], ...]
    binding_upper: tuple[tuple[tuple[str, ...], ...], ...]
    binding_hypotheses_lower: tuple[tuple[tuple[str, ...], ...], ...]
    binding_hypotheses_upper: tuple[tuple[tuple[str, ...], ...], ...]
    lower_rests_on_unverified: tuple[tuple[bool, ...], ...]
    upper_rests_on_unverified: tuple[tuple[bool, ...], ...]
    frame_rests_on_unverified: tuple[bool, ...]
    frame_hypothesis_ids: tuple[tuple[str, ...], ...]
    chebyshev_margin: Array
    chebyshev_center: tuple[Array, ...]
    chebyshev_status: tuple[BoundStatus, ...]
    chebyshev_metric: str
    cone_error_bound: Array
    simplex_iterations: int
    warm_started: bool
    label: str = SERIES_LABEL
    non_claims: tuple[str, ...] = field(
        default=(
            "static per-frame sets: over-approximates trajectory-feasible ranges",
            "conditional on the declared hypotheses; capacities are priors",
        )
    )

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.lower.shape[0])

    def width(self) -> Array:
        """``upper - lower`` (NaN where either bound is not BOUNDED)."""
        return np.asarray(self.upper - self.lower)

    def records(
        self, model: ModelProvenance, source: InputProvenance
    ) -> dict[str, QuantityRecord]:
        """One FAMILY_DEPENDENT record per quantity, values ``(frames, 2)``.

        A quantity with any non-BOUNDED bound becomes UNAVAILABLE with the
        reason stated (the finite bounds stay in :attr:`lower`/:attr:`upper`).
        """
        out = {}
        for j, name in enumerate(self.names):
            out[name] = self._record(j, name, model, source)
        return out

    def _record(
        self, j: int, name: str, model: ModelProvenance, source: InputProvenance
    ) -> QuantityRecord:
        vals = np.stack([self.lower[:, j], self.upper[:, j]], axis=1)
        common = {
            "units": self.units[j],
            "model_provenance": model,
            "input_provenance": source,
            "conventions": (STATIC_CONVENTION,),
        }
        bad = sorted(
            {
                s.name
                for t in range(self.n_frames)
                for s in (self.status_lower[t][j], self.status_upper[t][j])
                if s is not BoundStatus.BOUNDED
            }
        )
        if bad:
            return QuantityRecord(
                values=np.full(vals.shape, np.nan),
                allocation_dependence=AllocationDependence.UNAVAILABLE,
                reason=f"{name}: non-bounded envelope in some frames ({', '.join(bad)})",
                **common,
            )
        hyps: dict[str, None] = {}
        for t in range(self.n_frames):
            hyps.update(dict.fromkeys(self.frame_hypothesis_ids[t]))
            hyps.update(dict.fromkeys(self.binding_hypotheses_lower[t][j]))
            hyps.update(dict.fromkeys(self.binding_hypotheses_upper[t][j]))
        return QuantityRecord(
            values=vals,
            allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
            hypotheses=tuple(hyps) or ("static feasible set without declared rows",),
            **common,
        )


def _resolve_quantities(
    fs: FeasibleSet,
    quantities: Sequence[EnvelopeQuantity] | QuantityFactory,
) -> list[EnvelopeQuantity]:
    qs = list(quantities(fs) if callable(quantities) else quantities)
    if not qs:
        raise ValueError("at least one quantity is required")
    for q in qs:
        if not isinstance(q, EnvelopeQuantity):
            raise TypeError(f"quantities must be EnvelopeQuantity, got {type(q)!r}")
        if q.coeff.shape != (fs.n_z,):
            raise ValueError(
                f"quantity {q.name!r} has {q.coeff.size} columns, set has n_z={fs.n_z}"
            )
    names = [q.name for q in qs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate quantity names: {names}")
    return qs


def _usable(basis: Basis | None, fs: FeasibleSet) -> Basis | None:
    if basis is None:
        return None
    if basis.col_status.size != fs.n_z or basis.row_status.size != len(fs.rows):
        return None
    return basis


def static_envelopes(
    feasible_sets: Sequence[FeasibleSet],
    quantities: Sequence[EnvelopeQuantity] | QuantityFactory,
    *,
    chebyshev_weights: Sequence[float] | None = None,
    warm_start: bool = True,
) -> EnvelopeSeries:
    """Min and max of each quantity over each frame's feasible set.

    ``quantities`` is a sequence of :class:`EnvelopeQuantity` (same ``z``
    dimension in every frame) or a factory such as
    :func:`joint_torque_quantities` evaluated per frame.  Two LPs per quantity
    per frame; ``warm_start`` reuses the simplex basis between objectives and
    from the previous frame (it changes iteration counts, not the bounds).

    Preconditions: non-empty ``feasible_sets`` of :class:`FeasibleSet`
    (``ValueError``/``TypeError``), the same quantity names in every frame,
    and coefficient length equal to ``n_z``.  Postconditions: bounds are NaN
    unless BOUNDED; an empty frame is reported ``INFEASIBLE``, never raised;
    the series is labelled ``STATIC``.  Raises ``RuntimeError`` only when the
    solver is numerically inconclusive.
    """
    frames = list(feasible_sets)
    if not frames:
        raise ValueError("at least one feasible set is required")
    for fs in frames:
        if not isinstance(fs, FeasibleSet):
            raise TypeError(f"feasible_sets must hold FeasibleSet, got {type(fs)!r}")
    per_frame = [_resolve_quantities(fs, quantities) for fs in frames]
    names = tuple(q.name for q in per_frame[0])
    if any(tuple(q.name for q in qs) != names for qs in per_frame):
        raise ValueError("quantity names differ between frames")
    rows: list[Any] = []
    warm: Basis | None = None
    total = 0
    for fs, qs in zip(frames, per_frame, strict=True):
        bounds, basis, iters = _solve_frame(fs, qs, _usable(warm, fs), warm_start)
        total += iters
        warm = basis if basis is not None else warm
        cheb = chebyshev_margin(fs, metric_weights=chebyshev_weights)
        rows.append((bounds, cheb))
    return _assemble(
        frames, per_frame, names, rows, total, warm_start, chebyshev_weights
    )


def _assemble(
    frames: Sequence[FeasibleSet],
    per_frame: Sequence[Sequence[EnvelopeQuantity]],
    names: tuple[str, ...],
    rows: Sequence[Any],
    iterations: int,
    warm_start: bool,
    weights: Sequence[float] | None,
) -> EnvelopeSeries:
    def col(side: int, attr: str) -> tuple[Any, ...]:
        return tuple(tuple(getattr(b[side], attr) for b in bs) for bs, _ in rows)

    lower = np.array([[b[0].value for b in bs] for bs, _ in rows])
    upper = np.array([[b[1].value for b in bs] for bs, _ in rows])
    sign = tuple(
        tuple(
            None
            if b[0].status is BoundStatus.INFEASIBLE
            else classify_sign(b[0].value, b[1].value, scale=q.scale)
            for b, q in zip(bs, qs, strict=True)
        )
        for (bs, _), qs in zip(rows, per_frame, strict=True)
    )
    chebs = [c for _, c in rows]
    cone_err = [
        fs.cone_error_bound if fs.cone_error_bound is not None else math.nan
        for fs in frames
    ]
    return EnvelopeSeries(
        names=names,
        units=tuple(q.units for q in per_frame[0]),
        lower=_ro(lower),
        upper=_ro(upper),
        status_lower=col(0, "status"),
        status_upper=col(1, "status"),
        sign_class=sign,
        binding_lower=col(0, "rows"),
        binding_upper=col(1, "rows"),
        binding_hypotheses_lower=col(0, "hypotheses"),
        binding_hypotheses_upper=col(1, "hypotheses"),
        lower_rests_on_unverified=col(0, "unverified"),
        upper_rests_on_unverified=col(1, "unverified"),
        frame_rests_on_unverified=tuple(fs.rests_on_unverified for fs in frames),
        frame_hypothesis_ids=tuple(fs.hypothesis_ids for fs in frames),
        chebyshev_margin=_ro([c.radius for c in chebs]),
        chebyshev_center=tuple(c.center for c in chebs),
        chebyshev_status=tuple(c.status for c in chebs),
        chebyshev_metric=chebs[0].metric_label,
        cone_error_bound=_ro(cone_err),
        simplex_iterations=iterations,
        warm_started=warm_start,
    )
