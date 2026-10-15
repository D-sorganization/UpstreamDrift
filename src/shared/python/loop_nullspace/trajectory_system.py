"""Trajectory-coupled constraint system and solver adapter (NSX-25, #12568).

The per-frame feasible sets of NSX-21 are polyhedra (plus exact second-order
cones) in the null-space coordinates ``z_k``.  A *trajectory-coupled* set is
their product, stacked in ``z = (z_1, ..., z_T)``, intersected with affine
links between frames.  The only link implemented is a hard rate bound on an
affine quantity ``q``::

    | q_{k+1}(z_{k+1}) - q_k(z_k) | <= delta_k

(the NSX-00 ``trajectory_coupled`` oracle).  A hard rate bound is the
constraint form of the ``H-SMOOTH-TORQUE-RATE`` prior and carries that
hypothesis id; it is a modelling choice, not physiology.

:class:`StackedSystem` is a dense, solver-independent description of the
stacked set (``G z <= h`` plus cone blocks ``b_j - A_j z`` in a second-order
cone).  Its :attr:`~StackedSystem.digest` binds a certificate to the exact data
it was computed from, and a plain-numpy verifier needs nothing else.
:class:`SystemSolver` adapts it to HiGHS (no cones: one persistent model with
basis reuse) or Clarabel (exact cones) and returns the solver-independent
dual vector ``d >= 0`` with stationarity ``c + [G; A]^T d = 0`` (the
convention used by the certificates).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import scipy.sparse as sp

from src.shared.python.loop_nullspace.envelopes import (
    EnvelopeQuantity,
    _resolve_quantities,
)
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.registry import load_registry
from src.shared.python.loop_nullspace.solvers import (
    LPSession,
    SolveStatus,
    SOCBlock,
    solve_lp,
    solve_socp,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SMOOTH_ID",
    "ConeData",
    "MinOutcome",
    "QuantitySpec",
    "RateBound",
    "StackedSystem",
    "SystemSolver",
    "family_record",
    "select_quantity",
    "stack_system",
    "unverified_ids",
]

SMOOTH_ID = "H-SMOOTH-TORQUE-RATE"
CONE_MODES = ("exact", "linearized")
SOLVER_TOL = 1e-11

QuantitySpec = Any  # EnvelopeQuantity | Sequence[EnvelopeQuantity] | factory


def select_quantity(
    spec: QuantitySpec, name: str | None, fs: FeasibleSet
) -> EnvelopeQuantity:
    """The affine quantity ``spec`` defines on ``fs`` (factory, list or single).

    A factory or list returning several quantities needs ``name``.  Raises
    ``TypeError`` for a non-affine object and ``ValueError`` for a missing or
    unknown name or a coefficient length different from ``fs.n_z``.
    """
    if isinstance(spec, EnvelopeQuantity):
        spec = [spec]
    elif not (callable(spec) or isinstance(spec, (list, tuple))):
        raise TypeError(
            "quantity must be an EnvelopeQuantity, a sequence of them or a "
            f"factory (affine functionals only), got {type(spec)!r}"
        )
    qs = _resolve_quantities(fs, spec)
    if name is None:
        if len(qs) != 1:
            raise ValueError(
                f"quantity_name is required: candidates {[q.name for q in qs]}"
            )
        return qs[0]
    for q in qs:
        if q.name == name:
            return q
    raise ValueError(f"unknown quantity {name!r}; candidates {[q.name for q in qs]}")


@dataclass(frozen=True, eq=False)
class RateBound:
    """Hard bound ``|q_{k+1} - q_k| <= delta_k`` on an affine quantity.

    ``delta`` is a positive scalar or an array of ``T - 1`` positive entries.
    Raises ``ValueError`` for a non-positive or non-finite ``delta``.
    """

    quantity: QuantitySpec
    delta: Any
    quantity_name: str | None = None
    hypothesis_id: str = SMOOTH_ID

    def __post_init__(self) -> None:
        d = np.array(self.delta, dtype=float, copy=True)
        if d.ndim > 1 or d.size == 0 or not np.all(np.isfinite(d)) or np.any(d <= 0):
            raise ValueError(f"delta must be positive and finite, got {self.delta!r}")
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id must be non-empty")
        d.setflags(write=False)
        object.__setattr__(self, "delta", d)

    def describe(self) -> dict[str, Any]:
        """JSON-able description (the quantity itself is not serialised)."""
        return {
            "kind": "rate_bound",
            "quantity_name": self.quantity_name,
            "delta": [float(v) for v in np.atleast_1d(self.delta)],
            "hypothesis_id": self.hypothesis_id,
        }


@dataclass(frozen=True, eq=False)
class ConeData:
    """Dense exact cone ``b - A z`` in ``{(t, u): t >= ||u||}``."""

    A: Array
    b: Array
    label: str
    hypothesis_id: str


@dataclass(frozen=True, eq=False)
class StackedSystem:
    """``G z <= h`` and cone blocks over ``z = (z_1..z_T)`` (dense, read-only)."""

    n_var: int
    offsets: tuple[int, ...]
    G: Array
    h: Array
    row_labels: tuple[str, ...]
    row_hypotheses: tuple[str, ...]
    cones: tuple[ConeData, ...]
    hypothesis_ids: tuple[str, ...]
    cone_mode: str
    digest: str = field(default="")

    @property
    def n_frames(self) -> int:
        """Number of stacked frames."""
        return len(self.offsets) - 1

    @property
    def n_dual(self) -> int:
        """Length of the dual vector: polyhedral rows plus all cone rows."""
        return int(self.G.shape[0] + sum(c.b.size for c in self.cones))

    def embed(self, frame: int, vec: Any) -> Array:
        """Place a per-frame coefficient vector at the frame's block."""
        lo, hi = self.offsets[frame], self.offsets[frame + 1]
        out = np.zeros(self.n_var)
        out[lo:hi] = np.asarray(vec, dtype=float)
        return out

    def split(self, x: Any) -> tuple[Array, ...]:
        """Per-frame ``z_k`` of a stacked vector."""
        arr = np.asarray(x, dtype=float)
        return tuple(
            arr[self.offsets[k] : self.offsets[k + 1]].copy()
            for k in range(self.n_frames)
        )

    def stacked_matrix(self) -> tuple[Array, Array]:
        """``(A, b)`` with polyhedral rows first, then every cone block."""
        mats = [self.G] + [c.A for c in self.cones]
        rhs = [self.h] + [c.b for c in self.cones]
        return np.vstack(mats), np.concatenate(rhs)

    def with_row(self, g: Any, h: float, label: str) -> StackedSystem:
        """A copy with one extra polyhedral row (digest recomputed)."""
        sysd = replace(
            self,
            G=np.vstack([self.G, np.asarray(g, float)[None, :]]),
            h=np.append(self.h, float(h)),
            row_labels=(*self.row_labels, label),
            row_hypotheses=(*self.row_hypotheses, "PROBE"),
            digest="",
        )
        return _sealed(sysd)


def _digest(s: StackedSystem) -> str:
    sha = hashlib.sha256()
    sha.update(f"{s.cone_mode}|{s.n_var}|{s.offsets}".encode())
    for arr in (s.G, s.h):
        sha.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
        sha.update(str(arr.shape).encode())
    for c in s.cones:
        for arr in (c.A, c.b):
            sha.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
            sha.update(str(arr.shape).encode())
    return sha.hexdigest()


def _sealed(s: StackedSystem) -> StackedSystem:
    for arr in (s.G, s.h):
        arr.setflags(write=False)
    return replace(s, digest=_digest(s))


# --------------------------------------------------------------- assembly
def _validated_frames(feasible_sets: Sequence[FeasibleSet]) -> tuple[FeasibleSet, ...]:
    frames = tuple(feasible_sets)
    if not frames:
        raise ValueError("at least one feasible set is required")
    for k, fs in enumerate(frames):
        if not isinstance(fs, FeasibleSet):
            raise TypeError(f"frame {k} must be FeasibleSet, got {type(fs)!r}")
    return frames


def _frame_blocks(
    frames: tuple[FeasibleSet, ...], offsets: Array, mode: str
) -> tuple[Array, Array, list[str], list[str], list[ConeData]]:
    n_var = int(offsets[-1])
    mats: list[Array] = []
    rhs: list[Array] = []
    labels: list[str] = []
    hyps: list[str] = []
    cones: list[ConeData] = []
    for k, fs in enumerate(frames):
        lo, hi = int(offsets[k]), int(offsets[k + 1])
        rows = (
            list(fs.rows)
            if mode == "linearized"
            else [r for r in fs.rows if r.kind != "cone_linear"]
        )
        if rows:
            block = np.zeros((len(rows), n_var))
            block[:, lo:hi] = np.array([r.g for r in rows])
            mats.append(block)
            rhs.append(np.array([r.h for r in rows]))
            labels += [f"frame{k}:{r.label}" for r in rows]
            hyps += [r.hypothesis_id for r in rows]
        if mode == "exact":
            for cone in fs.cones:
                a = np.zeros((cone.A.shape[0], n_var))
                a[:, lo:hi] = cone.A
                cones.append(
                    ConeData(
                        a,
                        np.array(cone.b),
                        f"frame{k}:{cone.label}",
                        cone.hypothesis_id,
                    )
                )
    g = np.vstack(mats) if mats else np.zeros((0, n_var))
    h = np.concatenate(rhs) if rhs else np.zeros(0)
    return g, h, labels, hyps, cones


def _coupling_rows(
    frames: tuple[FeasibleSet, ...], offsets: Array, coupling: Sequence[RateBound]
) -> tuple[Array, Array, list[str], list[str]]:
    n_var = int(offsets[-1])
    rows: list[Array] = []
    rhs: list[float] = []
    labels: list[str] = []
    hyps: list[str] = []
    steps = len(frames) - 1
    for idx, rb in enumerate(coupling):
        if not isinstance(rb, RateBound):
            raise TypeError(f"coupling[{idx}] must be RateBound, got {type(rb)!r}")
        delta = np.atleast_1d(rb.delta)
        if delta.size not in (1, steps):
            raise ValueError(
                f"delta has {delta.size} entries, expected 1 or {steps} (T - 1)"
            )
        qs = [select_quantity(rb.quantity, rb.quantity_name, fs) for fs in frames]
        tag = rb.quantity_name or qs[0].name
        for k in range(steps):
            d = float(delta[0] if delta.size == 1 else delta[k])
            row = np.zeros(n_var)
            row[offsets[k + 1] : offsets[k + 2]] = qs[k + 1].coeff
            row[offsets[k] : offsets[k + 1]] -= qs[k].coeff
            shift = qs[k + 1].const - qs[k].const
            for sign, side in ((1.0, "upper"), (-1.0, "lower")):
                rows.append(sign * row)
                rhs.append(d - sign * shift)
                labels.append(f"coupling{idx}:rate:{tag}:{k}->{k + 1}:{side}")
                hyps.append(rb.hypothesis_id)
    if not rows:
        return np.zeros((0, n_var)), np.zeros(0), [], []
    return np.array(rows), np.array(rhs), labels, hyps


def stack_system(
    feasible_sets: Sequence[FeasibleSet],
    coupling: Sequence[RateBound] = (),
    *,
    cones: str = "exact",
) -> StackedSystem:
    """Stack the per-frame sets and the rate couplings into one system.

    ``cones="exact"`` keeps the Coulomb cones as second-order-cone blocks and
    drops their linearised rows; ``"linearized"`` keeps the declared polygon
    rows (as NSX-22 static envelopes do) and no cone blocks.
    Preconditions: non-empty ``feasible_sets`` of :class:`FeasibleSet`
    (``ValueError``/``TypeError``), every coupling a :class:`RateBound` whose
    ``delta`` has 1 or ``T - 1`` entries.  Postcondition: the digest binds the
    exact numbers; the row order is frames first, then couplings.
    """
    if cones not in CONE_MODES:
        raise ValueError(f"cones must be one of {CONE_MODES}, got {cones!r}")
    frames = _validated_frames(feasible_sets)
    offsets = np.concatenate([[0], np.cumsum([fs.n_z for fs in frames])]).astype(int)
    g, h, labels, hyps, cone_list = _frame_blocks(frames, offsets, cones)
    cg, ch, cl, chy = _coupling_rows(frames, offsets, tuple(coupling))
    ids = dict.fromkeys(h_id for fs in frames for h_id in fs.hypothesis_ids)
    ids.update(dict.fromkeys(chy))
    system = StackedSystem(
        n_var=int(offsets[-1]),
        offsets=tuple(int(v) for v in offsets),
        G=np.vstack([g, cg]) if cg.size else g,
        h=np.concatenate([h, ch]),
        row_labels=tuple(labels + cl),
        row_hypotheses=tuple(hyps + chy),
        cones=tuple(cone_list),
        hypothesis_ids=tuple(ids),
        cone_mode=cones,
    )
    return _sealed(system)


# ---------------------------------------------------------------- solving
@dataclass(frozen=True, eq=False)
class MinOutcome:
    """Result of minimising ``c . z`` over a :class:`StackedSystem`.

    ``dual`` is ``d >= 0`` over ``[rows; cone rows]`` with
    ``c + [G; A]^T d = 0`` (OPTIMAL) or the Farkas vector with
    ``[G; A]^T d = 0``, ``[h; b] . d < 0`` (INFEASIBLE).
    """

    status: SolveStatus
    x: Array | None = None
    objective: float | None = None
    dual: Array | None = None
    iterations: int = 0
    reason: str = ""


class SystemSolver:
    """Minimise affine functionals over one :class:`StackedSystem`.

    Without cones one persistent HiGHS model is reused (objective swaps, basis
    reuse); with exact cones each solve is a Clarabel SOCP.  Use as a context
    manager.  Raises ``RuntimeError`` on a numerically inconclusive solve.
    """

    def __init__(self, system: StackedSystem) -> None:
        self.system = system
        self.iterations = 0
        self._session: LPSession | None = None
        if system.n_var and not system.cones:
            has = system.G.shape[0] > 0
            self._session = LPSession(
                None,
                None,
                sp.csr_matrix(system.G) if has else None,
                system.h if has else None,
                [(None, None)] * system.n_var,
            )

    def __enter__(self) -> SystemSolver:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the HiGHS model, if any."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def feasibility(self) -> MinOutcome:
        """Is the system non-empty?  OPTIMAL (with a point) or INFEASIBLE."""
        return self.minimize(np.zeros(self.system.n_var))

    def minimize(self, c: Any) -> MinOutcome:
        """Minimise ``c . z``; statuses OPTIMAL, INFEASIBLE or UNBOUNDED."""
        cc = np.asarray(c, dtype=float)
        s = self.system
        if cc.shape != (s.n_var,):
            raise ValueError(f"c must have shape ({s.n_var},), got {cc.shape}")
        if s.n_var == 0:
            return _point_outcome(s)
        out = self._lp(cc) if self._session is not None else self._conic(cc)
        self.iterations += out.iterations
        if out.status in (
            SolveStatus.OPTIMAL,
            SolveStatus.INFEASIBLE,
            SolveStatus.UNBOUNDED,
        ):
            return out
        raise RuntimeError(f"solve inconclusive: {out.status.name} ({out.reason})")

    # -------------------------------------------------------------- LP
    def _lp(self, c: Array) -> MinOutcome:
        assert self._session is not None
        res = self._session.solve(c)
        if res.status is SolveStatus.OPTIMAL and res.x is not None:
            return MinOutcome(
                SolveStatus.OPTIMAL,
                np.array(res.x),
                float(res.objective or 0.0),
                _lp_dual(self.system, c, res.row_dual),
                res.iterations,
            )
        if res.status is SolveStatus.INFEASIBLE:
            return MinOutcome(
                SolveStatus.INFEASIBLE,
                dual=_farkas_lp(self.system),
                iterations=res.iterations,
            )
        return MinOutcome(res.status, iterations=res.iterations, reason=res.reason)

    # ----------------------------------------------------------- conic
    def _conic(self, c: Array) -> MinOutcome:
        s = self.system
        has = s.G.shape[0] > 0
        blocks = [SOCBlock(sp.csc_matrix(cd.A), cd.b) for cd in s.cones]
        res = solve_socp(
            c,
            None,
            None,
            sp.csr_matrix(s.G) if has else None,
            s.h if has else None,
            blocks,
            tol=SOLVER_TOL,
        )
        if res.status is SolveStatus.OPTIMAL and res.x is not None:
            return MinOutcome(
                SolveStatus.OPTIMAL,
                np.array(res.x),
                float(c @ res.x),
                None if res.z is None else np.array(res.z),
                res.iterations,
            )
        if res.status is SolveStatus.INFEASIBLE:
            ray = None if res.dual_ray is None else np.array(res.dual_ray)
            return MinOutcome(
                SolveStatus.INFEASIBLE,
                dual=ray,
                iterations=res.iterations,
                reason=res.reason,
            )
        return MinOutcome(res.status, iterations=res.iterations, reason=res.reason)


def _point_outcome(s: StackedSystem) -> MinOutcome:
    """``n_var == 0``: the set is the single point ``z = ()`` or empty."""
    bad = np.flatnonzero(s.h < -1e-9)
    if bad.size == 0:
        return MinOutcome(SolveStatus.OPTIMAL, np.zeros(0), 0.0, np.zeros(s.n_dual))
    dual = np.zeros(s.n_dual)
    dual[int(bad[0])] = 1.0
    return MinOutcome(SolveStatus.INFEASIBLE, dual=dual)


def _lp_dual(s: StackedSystem, c: Array, row_dual: Any) -> Array | None:
    """Multipliers ``mu >= 0`` with ``c + G^T mu = 0`` from HiGHS row duals."""
    if row_dual is None:
        return None
    y = np.asarray(row_dual, dtype=float)
    best = None
    for cand in (np.maximum(-y, 0.0), np.maximum(y, 0.0)):
        res = float(np.max(np.abs(c + s.G.T @ cand))) if c.size else 0.0
        if best is None or res < best[0]:
            best = (res, cand)
    assert best is not None
    return np.array(best[1])


def _farkas_lp(s: StackedSystem) -> Array | None:
    """``mu >= 0`` with ``G^T mu = 0``, ``sum mu = 1`` and ``h . mu`` minimal."""
    m = s.G.shape[0]
    if m == 0:
        return None
    a_eq = sp.csr_matrix(np.vstack([s.G.T, np.ones((1, m))]))
    b_eq = np.concatenate([np.zeros(s.n_var), [1.0]])
    res = solve_lp(s.h, a_eq, b_eq, None, None, [(0.0, None)] * m)
    if res.status is not SolveStatus.OPTIMAL or res.x is None:
        return None
    return np.array(res.x)


# --------------------------------------------------------------- evidence
def unverified_ids(ids: Sequence[str]) -> tuple[str, ...]:
    """Hypothesis ids that fail the registry's display gate (or are unknown)."""
    registry = load_registry()
    bad = []
    for hid in ids:
        try:
            registry.require_verified_for_display([hid])
        except (ValueError, KeyError):
            bad.append(hid)
    return tuple(bad)


def family_record(
    frames: Sequence[FeasibleSet],
    values: Any,
    units: str,
    hypotheses: Sequence[str],
    *,
    source: str,
    note: str,
    conventions: Sequence[str] = (),
    unavailable: str | None = None,
) -> QuantityRecord:
    """A ``FAMILY_DEPENDENT`` (or ``UNAVAILABLE``) record for a probe output."""
    model = ModelProvenance("unspecified-decomposition", "0")
    for fs in frames:
        dec = fs.decomposition
        first = next(iter(dec.records.values()), None) if dec is not None else None
        if first is not None:
            model = first.model_provenance
            break
    vals = np.asarray(values, dtype=float)
    common = {
        "units": units,
        "model_provenance": model,
        "input_provenance": InputProvenance("probe", source, note),
        "conventions": tuple(conventions),
    }
    if unavailable is not None:
        return QuantityRecord(
            values=np.full(vals.shape, np.nan),
            allocation_dependence=AllocationDependence.UNAVAILABLE,
            reason=unavailable,
            **common,
        )
    return QuantityRecord(
        values=vals,
        allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
        hypotheses=tuple(hypotheses) or ("feasible set without declared rows",),
        **common,
    )
