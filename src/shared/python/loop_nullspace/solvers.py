"""Convex solver backends for the loop null-space package (NSX-20, #12563).

Routing (epic #12593, rule R11):

* Polyhedral sets (LPs) use ``highspy`` through the ``Highs`` object directly,
  because ``scipy.optimize.linprog`` has no warm start.  :class:`LPSession`
  keeps one model alive, swaps the objective and re-solves from the previous
  basis, which is the access pattern of envelopes (two LPs per quantity per
  frame on one constraint set).
* Second-order cones use Clarabel (:func:`solve_socp`).
* There is never a silent fallback.  A missing backend raises ``ImportError``
  naming the ``loop-nullspace`` extra; :func:`available_backends` lets a caller
  turn that into an explicit "unavailable" state of its own.

Problem forms
-------------
LP, with stacked row order ``[A_eq; A_ub]``::

    min  c^T x   s.t.  A_eq x = b_eq,  A_ub x <= b_ub,  lo <= x <= hi

SOCP, Clarabel's native form with the cones stacked in the order
``[A_eq (zero cone); A_ub (nonnegative); SOC blocks in the order given]``::

    min  c^T x   s.t.  A x + s = b,  s in K

so a :class:`SOCBlock` ``(A, b)`` asserts that ``b - A x`` lies in the
second-order cone ``{(t, u): t >= ||u||_2}`` (first row is ``t``).

Certificates (what each solver really provides)
-----------------------------------------------
* HiGHS (simplex, presolve off): a dual ray for ``INFEASIBLE`` and a primal ray
  for ``UNBOUNDED`` *when the simplex terminates with one*; if HiGHS reports
  the status without a ray the field is ``None`` and ``reason`` says so.  The
  dual ray is in the stacked ``[A_eq; A_ub]`` row order and is **sign
  convention free for the caller**: it certifies infeasibility if
  ``y^T A x`` over the variable box does not intersect ``y^T r`` over the row
  box (the test-suite checks exactly this).
* Clarabel: for ``INFEASIBLE`` the final ``z`` is a Farkas certificate
  (``A^T z = 0``, ``b^T z < 0``, ``z`` in the dual cone) and for
  ``UNBOUNDED`` the final ``x`` is a recession direction (``A x + s = 0``,
  ``s`` in ``K``, ``c^T x < 0``), both up to the solver tolerance.  Only the
  strict statuses count: the "almost" statuses map to ``NUMERICAL``.

Imports of ``highspy`` and ``clarabel`` are lazy, so importing this module
needs only numpy and scipy.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from src.shared.python.core.contracts import StateError

logger = logging.getLogger(__name__)

__all__ = [
    "Basis",
    "ConicResult",
    "KKTResiduals",
    "LPResult",
    "LPSession",
    "SOCBlock",
    "SolveStatus",
    "available_backends",
    "lp_kkt_residuals",
    "solve_lp",
    "solve_socp",
]

_EXTRA_HINT = (
    "install the 'loop-nullspace' extra: pip install 'upstream-drift[loop-nullspace]'"
)


class SolveStatus(enum.Enum):
    """Solver-independent termination status."""

    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    NUMERICAL = "numerical"


def _import_backend(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError as exc:
        raise ImportError(
            f"The '{name}' package is required for this solver backend; {_EXTRA_HINT}"
        ) from exc


def available_backends() -> dict[str, bool]:
    """Report which optional backends can be imported (no fallbacks exist)."""
    out = {}
    for name in ("highspy", "clarabel"):
        try:
            _import_backend(name)
            out[name] = True
        except ImportError:
            out[name] = False
    return out


# ------------------------------------------------------------ value objects
def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class Basis:
    """A simplex basis: HiGHS status codes per column and per row.

    Codes (HiGHS ``HighsBasisStatus``): 0 lower, 1 basic, 2 upper, 3 zero,
    4 nonbasic.  Rows are in stacked ``[A_eq; A_ub]`` order.
    """

    col_status: np.ndarray
    row_status: np.ndarray

    def __post_init__(self) -> None:
        for name in ("col_status", "row_status"):
            arr = np.asarray(getattr(self, name))
            if arr.ndim != 1:
                raise ValueError(f"basis {name} must be 1-D")
            object.__setattr__(self, name, _readonly(arr.astype(np.int8)))


@dataclass(frozen=True)
class KKTResiduals:
    """Scaled optimality residuals, each dimensionless.

    ``primal``: largest row/bound violation over ``1 + ||finite bounds||_inf``.
    ``dual``: stationarity ``||c - A^T y - z||_inf`` and multiplier sign
    violations over ``1 + ||c||_inf``.
    ``complementarity``: ``|multiplier| * slack`` over ``1 + |c^T x|``.
    """

    primal: float
    dual: float
    complementarity: float

    def max(self) -> float:
        """Largest of the three residuals."""
        return max(self.primal, self.dual, self.complementarity)


@dataclass(frozen=True)
class LPResult:
    """Outcome of an LP solve.

    Postconditions: ``x``/``row_dual``/``col_dual``/``basis``/``kkt_residuals``
    are populated iff ``status`` is OPTIMAL; ``dual_ray`` only for INFEASIBLE
    and ``primal_ray`` only for UNBOUNDED (and then only if HiGHS supplied it).
    """

    status: SolveStatus
    x: np.ndarray | None = None
    objective: float | None = None
    basis: Basis | None = None
    kkt_residuals: KKTResiduals | None = None
    iterations: int = 0
    row_dual: np.ndarray | None = None
    col_dual: np.ndarray | None = None
    dual_ray: np.ndarray | None = None
    primal_ray: np.ndarray | None = None
    reason: str = ""


@dataclass(frozen=True)
class ConicResult:
    """Outcome of a conic solve (see module docstring for conventions)."""

    status: SolveStatus
    x: np.ndarray | None = None
    z: np.ndarray | None = None
    s: np.ndarray | None = None
    objective: float | None = None
    kkt_residuals: KKTResiduals | None = None
    iterations: int = 0
    dual_ray: np.ndarray | None = None
    primal_ray: np.ndarray | None = None
    reason: str = ""


@dataclass(frozen=True)
class SOCBlock:
    """One second-order cone constraint ``b - A x in {(t, u): t >= ||u||}``."""

    A: Any
    b: Any
    dim: int = field(init=False)

    def __post_init__(self) -> None:
        a = _as_sparse(self.A, "SOCBlock.A")
        b = _as_vector(self.b, "SOCBlock.b")
        if a.shape[0] != b.size:
            raise ValueError(f"SOCBlock.A has {a.shape[0]} rows but b has {b.size}")
        if a.shape[0] < 2:
            raise ValueError("SOCBlock dimension must be at least 2 (t plus one u)")
        object.__setattr__(self, "A", a.tocsc())
        object.__setattr__(self, "b", _readonly(b))
        object.__setattr__(self, "dim", int(a.shape[0]))


# ----------------------------------------------------------- input checking
def _as_vector(v: Any, name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _as_sparse(a: Any, name: str) -> sp.csr_matrix:
    if not sp.issparse(a):
        a = np.asarray(a, dtype=float)
        if a.ndim != 2:
            raise ValueError(f"{name} must be 2-D, got shape {a.shape}")
    mat = sp.csr_matrix(a).astype(float)
    if not np.all(np.isfinite(mat.data)):
        raise ValueError(f"{name} must be finite")
    return mat


def _rows(
    a: Any, b: Any, n: int, a_name: str, b_name: str
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Validate an optional ``(A, b)`` pair; ``None`` means zero rows."""
    if a is None:
        if b is not None and np.size(b) > 0:
            raise ValueError(f"{b_name} given without {a_name}")
        return sp.csr_matrix((0, n)), np.zeros(0)
    if b is None:
        raise ValueError(f"{a_name} given but {b_name} is None")
    mat = _as_sparse(a, a_name)
    vec = _as_vector(b, b_name)
    if mat.shape[1] != n:
        raise ValueError(f"{a_name} has {mat.shape[1]} columns, expected {n}")
    if mat.shape[0] != vec.size:
        raise ValueError(
            f"{a_name} has {mat.shape[0]} rows but {b_name} has {vec.size}"
        )
    return mat, vec


def _bounds(bounds: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lo, hi)`` with ``None``/``+-inf`` meaning unbounded."""
    if bounds is None:
        raise ValueError("bounds must be given explicitly (use (None, None) for free)")
    if len(bounds) != n:
        raise ValueError(f"bounds has {len(bounds)} entries but c has {n} variables")
    lo = np.empty(n)
    hi = np.empty(n)
    for i, pair in enumerate(bounds):
        if len(pair) != 2:
            raise ValueError(f"bounds[{i}] must be a (lower, upper) pair")
        lo[i] = -np.inf if pair[0] is None else float(pair[0])
        hi[i] = np.inf if pair[1] is None else float(pair[1])
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
        raise ValueError("bounds must not contain NaN")
    if np.any(lo == np.inf) or np.any(hi == -np.inf):
        raise ValueError("bounds: lower must be < +inf and upper > -inf")
    if np.any(lo > hi):
        raise ValueError("bounds: lower exceeds upper for some variable")
    return lo, hi


# ----------------------------------------------------------- KKT residuals
def _sign_violation(mult: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Multiplier sign violation for a ``lo <= . <= hi`` constraint (min form)."""
    v = np.where(np.isinf(hi) & np.isfinite(lo), np.maximum(0.0, -mult), 0.0)
    w = np.where(np.isinf(lo) & np.isfinite(hi), np.maximum(0.0, mult), 0.0)
    u = np.where(np.isinf(lo) & np.isinf(hi), np.abs(mult), 0.0)
    both = np.concatenate([v, w, u])
    return float(both.max()) if both.size else 0.0


def _complementarity(
    mult: np.ndarray, val: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> float:
    with np.errstate(invalid="ignore"):
        d_lo = np.where(np.isfinite(lo), val - lo, 0.0)
        d_hi = np.where(np.isfinite(hi), hi - val, 0.0)
    comp = np.where(mult > 0.0, mult * np.abs(d_lo), -mult * np.abs(d_hi))
    return float(np.abs(comp).max()) if comp.size else 0.0


def lp_kkt_residuals(
    c: Any,
    A_eq: Any,
    b_eq: Any,
    A_ub: Any,
    b_ub: Any,
    bounds: Any,
    *,
    primal_dual: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> KKTResiduals:
    """Scaled KKT residuals of a primal/dual pair for the LP in the module docs.

    ``primal_dual`` is ``(x, row_dual, col_dual)``.

    Multipliers follow HiGHS: ``c - A^T row_dual = col_dual`` and, for a
    minimisation, a multiplier is nonnegative at a lower bound and nonpositive
    at an upper bound.  Rows are stacked ``[A_eq; A_ub]``.
    """
    cc = _as_vector(c, "c")
    n = cc.size
    a_eq, be = _rows(A_eq, b_eq, n, "A_eq", "b_eq")
    a_ub, bu = _rows(A_ub, b_ub, n, "A_ub", "b_ub")
    lo, hi = _bounds(bounds, n)
    big_a = sp.vstack([a_eq, a_ub]).tocsr()
    row_lo = np.concatenate([be, np.full(bu.size, -np.inf)])
    row_hi = np.concatenate([be, bu])
    if len(primal_dual) != 3:
        raise ValueError("primal_dual must be (x, row_dual, col_dual)")
    xx, y, z = (np.asarray(part, dtype=float) for part in primal_dual)
    ax = big_a @ xx
    fin = np.concatenate([row_lo, row_hi, lo, hi])
    fin = fin[np.isfinite(fin)]
    p_scale = 1.0 + (float(np.abs(fin).max()) if fin.size else 0.0)
    viol = np.concatenate([row_lo - ax, ax - row_hi, lo - xx, xx - hi, np.zeros(1)])
    primal = float(max(0.0, viol.max())) / p_scale
    stat = float(np.abs(cc - big_a.T @ y - z).max()) if n else 0.0
    d_scale = 1.0 + float(np.abs(cc).max() if n else 0.0)
    dual = (
        max(stat, _sign_violation(y, row_lo, row_hi), _sign_violation(z, lo, hi))
        / d_scale
    )
    comp = max(
        _complementarity(y, ax, row_lo, row_hi), _complementarity(z, xx, lo, hi)
    ) / (1.0 + abs(float(cc @ xx)))
    return KKTResiduals(primal=primal, dual=dual, complementarity=comp)


# --------------------------------------------------------------- LP session
_HIGHS_BASIS_LOWER = 0


class LPSession:
    """A persistent HiGHS model with objective swaps and basis reuse.

    The constraint set ``(A_eq, b_eq, A_ub, b_ub, bounds)`` is fixed at
    construction.  Each :meth:`solve` replaces the objective and re-solves
    from the basis HiGHS retained from the previous solve (or from ``warm``),
    so a sequence of nearby objectives costs few simplex iterations.  Presolve
    is off and the dual simplex is used: this is required for basis reuse and
    for the solver to return infeasibility/unboundedness rays.

    Use as a context manager or call :meth:`close`.  Not thread safe.
    """

    def __init__(
        self,
        A_eq: Any,
        b_eq: Any,
        A_ub: Any,
        b_ub: Any,
        bounds: Any,
        *,
        max_iterations: int | None = None,
    ) -> None:
        """Build the model.  Raises ``ValueError`` on inconsistent shapes."""
        n = len(bounds) if bounds is not None else 0
        self._lo, self._hi = _bounds(bounds, n)
        self._n = n
        a_eq, be = _rows(A_eq, b_eq, n, "A_eq", "b_eq")
        a_ub, bu = _rows(A_ub, b_ub, n, "A_ub", "b_ub")
        self._a = sp.vstack([a_eq, a_ub]).tocsc()
        self._row_lo = np.concatenate([be, np.full(bu.size, -np.inf)])
        self._row_hi = np.concatenate([be, bu])
        self._b_eq, self._b_ub = be, bu
        self._a_eq, self._a_ub = a_eq, a_ub
        if max_iterations is not None and max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self._hs = _import_backend("highspy")
        self._highs: Any = self._hs.Highs()
        self._configure(max_iterations)
        self._load_model()

    @property
    def num_vars(self) -> int:
        """Number of columns."""
        return self._n

    @property
    def num_rows(self) -> int:
        """Number of stacked rows ``[A_eq; A_ub]``."""
        return int(self._a.shape[0])

    def _configure(self, max_iterations: int | None) -> None:
        h = self._highs
        h.setOptionValue("output_flag", False)
        h.setOptionValue("presolve", "off")
        h.setOptionValue("solver", "simplex")
        h.setOptionValue("simplex_strategy", 1)  # dual simplex
        h.setOptionValue("parallel", "off")
        if max_iterations is not None:
            h.setOptionValue("simplex_iteration_limit", int(max_iterations))

    def _load_model(self) -> None:
        hs = self._hs
        lp = hs.HighsLp()
        lp.num_col_ = self._n
        lp.num_row_ = self.num_rows
        lp.col_cost_ = np.zeros(self._n)
        lp.col_lower_ = self._lo
        lp.col_upper_ = self._hi
        lp.row_lower_ = self._row_lo
        lp.row_upper_ = self._row_hi
        lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
        lp.a_matrix_.start_ = self._a.indptr.astype(np.int32)
        lp.a_matrix_.index_ = self._a.indices.astype(np.int32)
        lp.a_matrix_.value_ = self._a.data.astype(float)
        self._highs.passModel(lp)
        self._closed = False

    def __enter__(self) -> LPSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the HiGHS model; further solves raise ``StateError``."""
        if not self._closed:
            self._highs.clear()
            self._closed = True

    def reset(self) -> None:
        """Discard the retained basis so the next solve is a cold start."""
        self._require_open()
        self._highs.clearSolver()

    def _require_open(self) -> None:
        if self._closed:
            raise StateError("LPSession is closed", current_state="closed")

    def _set_basis(self, basis: Basis) -> None:
        if basis.col_status.size != self._n or basis.row_status.size != self.num_rows:
            raise ValueError(
                f"warm basis shape ({basis.col_status.size}, {basis.row_status.size})"
                f" does not match the model ({self._n}, {self.num_rows})"
            )
        hs = self._hs
        hb = hs.HighsBasis()
        hb.col_status = [hs.HighsBasisStatus(int(s)) for s in basis.col_status]
        hb.row_status = [hs.HighsBasisStatus(int(s)) for s in basis.row_status]
        hb.valid = True
        if self._highs.setBasis(hb) != hs.HighsStatus.kOk:
            raise ValueError("HiGHS rejected the warm basis (not a valid basis)")

    def solve(self, c: Any, *, warm: Basis | None = None) -> LPResult:
        """Minimise ``c^T x`` over the session's constraint set.

        Preconditions: ``c`` is finite with length ``num_vars``; ``warm`` (if
        given) matches the model dimensions.  Without ``warm`` the basis left
        by the previous solve is reused.  Postcondition: see :class:`LPResult`.
        """
        self._require_open()
        cc = _as_vector(c, "c")
        if cc.size != self._n:
            raise ValueError(f"c has {cc.size} entries, expected {self._n}")
        if warm is not None:
            self._set_basis(warm)
        h = self._highs
        h.changeColsCost(self._n, np.arange(self._n, dtype=np.int32), cc)
        h.run()
        return self._collect(cc)

    def _collect(self, c: np.ndarray) -> LPResult:
        h, hs = self._highs, self._hs
        ms = h.getModelStatus()
        iters = int(h.getInfo().simplex_iteration_count)
        if ms == hs.HighsModelStatus.kOptimal:
            return self._optimal(c, iters)
        if ms == hs.HighsModelStatus.kInfeasible:
            has, ray = h.getDualRay()[1:]
            return LPResult(
                SolveStatus.INFEASIBLE,
                iterations=iters,
                dual_ray=_readonly(ray) if has else None,
                reason="" if has else "HiGHS reported infeasible without a dual ray",
            )
        if ms == hs.HighsModelStatus.kUnbounded:
            has, ray = h.getPrimalRay()[1:]
            return LPResult(
                SolveStatus.UNBOUNDED,
                iterations=iters,
                primal_ray=_readonly(ray) if has else None,
                reason="" if has else "HiGHS reported unbounded without a primal ray",
            )
        return LPResult(
            SolveStatus.NUMERICAL, iterations=iters, reason=h.modelStatusToString(ms)
        )

    def _optimal(self, c: np.ndarray, iters: int) -> LPResult:
        h = self._highs
        sol = h.getSolution()
        x = np.array(sol.col_value)
        y = np.array(sol.row_dual)
        z = np.array(sol.col_dual)
        hb = h.getBasis()
        basis = Basis(
            np.array([int(s.value) for s in hb.col_status]),
            np.array([int(s.value) for s in hb.row_status]),
        )
        kkt = lp_kkt_residuals(
            c,
            self._a_eq if self._a_eq.shape[0] else None,
            self._b_eq if self._a_eq.shape[0] else None,
            self._a_ub if self._a_ub.shape[0] else None,
            self._b_ub if self._a_ub.shape[0] else None,
            list(zip(self._lo, self._hi, strict=True)),
            primal_dual=(x, y, z),
        )
        return LPResult(
            SolveStatus.OPTIMAL,
            x=_readonly(x),
            objective=float(c @ x),
            basis=basis,
            kkt_residuals=kkt,
            iterations=iters,
            row_dual=_readonly(y),
            col_dual=_readonly(z),
        )


def solve_lp(
    c: Any,
    A_eq: Any,
    b_eq: Any,
    A_ub: Any,
    b_ub: Any,
    bounds: Any,
    *,
    warm: Basis | None = None,
    max_iterations: int | None = None,
) -> LPResult:
    """Solve one LP with HiGHS; see the module docstring for the problem form.

    Preconditions: all data finite, shapes consistent, ``bounds`` one
    ``(lower, upper)`` pair per variable (``None`` = unbounded), ``warm`` (a
    :class:`Basis` from an earlier solve of the *same* constraint set) matches
    the model.  Postcondition: ``status`` is one of :class:`SolveStatus` and the
    certificate for INFEASIBLE/UNBOUNDED is returned, never dropped.
    For repeated solves on one constraint set use :class:`LPSession`.
    """
    cc = _as_vector(c, "c")
    with LPSession(
        A_eq,
        b_eq,
        A_ub,
        b_ub,
        _checked_bounds(bounds, cc.size),
        max_iterations=max_iterations,
    ) as session:
        return session.solve(cc, warm=warm)


def _checked_bounds(bounds: Any, n: int) -> Any:
    _bounds(bounds, n)  # raises with the c-vs-bounds message on mismatch
    return bounds


# --------------------------------------------------------------------- SOCP
_CLARABEL_STATUS = {
    "Solved": SolveStatus.OPTIMAL,
    "PrimalInfeasible": SolveStatus.INFEASIBLE,
    "DualInfeasible": SolveStatus.UNBOUNDED,
}


def _bound_rows(lo: np.ndarray, hi: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    """Finite variable bounds as ``G x <= h`` rows."""
    n = lo.size
    eye = sp.identity(n, format="csr")
    fl = np.isfinite(lo)
    fh = np.isfinite(hi)
    g = sp.vstack([-eye[fl], eye[fh]]).tocsr()
    h = np.concatenate([-lo[fl], hi[fh]])
    return g, h


def _soc_violation(v: np.ndarray, dims: Sequence[int]) -> float:
    worst = 0.0
    pos = 0
    for d in dims:
        t, u = v[pos], v[pos + 1 : pos + d]
        worst = max(worst, float(np.linalg.norm(u)) - float(t))
        pos += d
    return worst


def solve_socp(
    c: Any,
    A_eq: Any,
    b_eq: Any,
    A_ub: Any,
    b_ub: Any,
    cones: Sequence[SOCBlock],
    *,
    bounds: Any = None,
    max_iter: int | None = None,
    P: Any = None,
    tol: float | None = None,
) -> ConicResult:
    """Solve a (convex quadratic) objective over polyhedral and SOC constraints.

    The objective is ``1/2 x^T P x + c^T x``; ``P=None`` means linear (NSX-20
    behaviour, unchanged).  ``P`` must be symmetric positive semidefinite of
    shape ``(n, n)`` (symmetry is checked here, semidefiniteness by Clarabel).
    ``tol``, when given, replaces Clarabel's absolute/relative gap and
    feasibility tolerances (default solver tolerances otherwise).

    Rows are stacked ``[A_eq; A_ub; finite bounds; SOC blocks]`` with
    ``A x + s = b``, ``s`` in zero/nonnegative/second-order cones.
    Preconditions: finite data, consistent shapes, every ``SOCBlock`` has
    ``num_vars`` columns.  ``bounds=None`` leaves variables free.
    Postcondition: on OPTIMAL ``x``, ``z``, ``s`` and ``kkt_residuals`` are set;
    on INFEASIBLE ``dual_ray`` holds Clarabel's Farkas ``z``; on UNBOUNDED
    ``primal_ray`` holds its recession direction ``x``.  Clarabel's "almost"
    and failure statuses map to NUMERICAL with the solver status as ``reason``.
    """
    cl = _import_backend("clarabel")
    cc = _as_vector(c, "c")
    n = cc.size
    p_mat = _quadratic(P, n)
    a_eq, be = _rows(A_eq, b_eq, n, "A_eq", "b_eq")
    a_ub, bu = _rows(A_ub, b_ub, n, "A_ub", "b_ub")
    if bounds is None:
        g, h = sp.csr_matrix((0, n)), np.zeros(0)
    else:
        g, h = _bound_rows(*_bounds(bounds, n))
    blocks = list(cones)
    for blk in blocks:
        if blk.A.shape[1] != n:
            raise ValueError(f"SOCBlock has {blk.A.shape[1]} columns, expected {n}")
    mats = [a_eq, a_ub, g] + [blk.A for blk in blocks]
    rhs = [be, bu, h] + [blk.b for blk in blocks]
    big_a = sp.vstack(mats).tocsc()
    big_b = np.concatenate(rhs)
    cone_list = []
    if be.size:
        cone_list.append(cl.ZeroConeT(be.size))
    if bu.size + h.size:
        cone_list.append(cl.NonnegativeConeT(bu.size + h.size))
    cone_list += [cl.SecondOrderConeT(blk.dim) for blk in blocks]
    settings = cl.DefaultSettings()
    settings.verbose = False
    if max_iter is not None:
        settings.max_iter = int(max_iter)
    if tol is not None:
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError(f"tol must be positive and finite, got {tol!r}")
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = float(tol)
    solver = cl.DefaultSolver(
        sp.triu(p_mat).tocsc(), cc, big_a, big_b, cone_list, settings
    )
    sol = solver.solve()
    return _conic_result(
        sol,
        cc,
        big_a,
        big_b,
        n_zero=be.size,
        n_nonneg=bu.size + h.size,
        blocks=blocks,
        p_mat=p_mat,
    )


def _quadratic(P: Any, n: int) -> sp.csr_matrix:
    """Validated symmetric ``n x n`` quadratic term (zero when ``P`` is None)."""
    if P is None:
        return sp.csr_matrix((n, n))
    mat = _as_sparse(P, "P")
    if mat.shape != (n, n):
        raise ValueError(f"P must be ({n}, {n}), got {mat.shape}")
    asym = abs(mat - mat.T)
    scale = max(1.0, float(abs(mat).max()) if mat.nnz else 1.0)
    if asym.nnz and float(asym.max()) > 1e-12 * scale:
        raise ValueError("P must be symmetric")
    return mat


def _conic_result(
    sol: Any,
    c: np.ndarray,
    a: sp.csc_matrix,
    b: np.ndarray,
    *,
    n_zero: int,
    n_nonneg: int,
    blocks: Sequence[SOCBlock],
    p_mat: sp.csr_matrix,
) -> ConicResult:
    name = str(sol.status)
    status = _CLARABEL_STATUS.get(name, SolveStatus.NUMERICAL)
    iters = int(sol.iterations)
    x, z, s = np.array(sol.x), np.array(sol.z), np.array(sol.s)
    if status is SolveStatus.INFEASIBLE:
        return ConicResult(status, iterations=iters, dual_ray=_readonly(z))
    if status is SolveStatus.UNBOUNDED:
        return ConicResult(status, iterations=iters, primal_ray=_readonly(x))
    if status is SolveStatus.NUMERICAL:
        return ConicResult(status, iterations=iters, reason=name)
    dims = [blk.dim for blk in blocks]
    off = n_zero + n_nonneg
    pscale = 1.0 + float(np.abs(b).max() if b.size else 0.0)
    dscale = 1.0 + float(np.abs(c).max() if c.size else 0.0)
    primal = float(np.abs(a @ x + s - b).max() if b.size else 0.0)
    primal = max(
        primal,
        float(max(0.0, -s[n_zero:off].min())) if n_nonneg else 0.0,
        max(0.0, _soc_violation(s[off:], dims)),
    )
    dual = float(np.abs(p_mat @ x + c + a.T @ z).max()) if c.size else 0.0
    dual = max(
        dual,
        float(max(0.0, -z[n_zero:off].min())) if n_nonneg else 0.0,
        max(0.0, _soc_violation(z[off:], dims)),
    )
    objective = float(c @ x + 0.5 * x @ (p_mat @ x))
    comp = abs(float(s @ z)) / (1.0 + abs(objective))
    return ConicResult(
        status,
        x=_readonly(x),
        z=_readonly(z),
        s=_readonly(s),
        objective=objective,
        kkt_residuals=KKTResiduals(primal / pscale, dual / dscale, comp),
        iterations=iters,
    )
