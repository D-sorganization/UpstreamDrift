"""Convex-QP front end shared by the legacy allocator shims (NSX-24, #12567).

``ContactForceAllocator`` and ``redistribute_forces``/``redistribute_trajectory``
used to run their own SLSQP.  Both are convex (a PSD quadratic over linear
rows and exact second-order cones), so they now go through
:func:`~.solvers.solve_socp` (Clarabel).  This module is the single place that
turns "quadratic + equalities + inequalities + box + Coulomb cones" into that
call, so the shims contain no solver code (DRY, ADR-0054).

Backend behaviour is inherited unchanged from :mod:`.solvers`: a missing
``clarabel`` raises ``ImportError`` naming the ``loop-nullspace`` extra.
There is no fallback to another optimizer, because that would change the
reported optimum without saying so (rule R11).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp

from src.shared.python.loop_nullspace.solvers import (
    ConicResult,
    SOCBlock,
    solve_socp,
)

__all__ = ["friction_cone_rows", "solve_qp"]


def _fixed_rows(
    lo: np.ndarray, hi: np.ndarray, n: int
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Split ``lo <= x <= hi`` into equality rows (lo == hi) and a free box."""
    fixed = np.isfinite(lo) & (lo == hi)
    rows = sp.identity(n, format="csr")[np.flatnonzero(fixed)]
    box_lo = np.where(fixed, -np.inf, lo)
    box_hi = np.where(fixed, np.inf, hi)
    return rows, lo[fixed], np.column_stack([box_lo, box_hi])


def _as_rows(a: Any, b: Any, n: int) -> tuple[sp.csr_matrix, np.ndarray]:
    if a is None:
        return sp.csr_matrix((0, n)), np.zeros(0)
    source: Any = a if sp.issparse(a) else np.asarray(a, dtype=float)
    mat = sp.csr_matrix(source)
    return mat, np.asarray(b, dtype=float)


def solve_qp(
    P: Any,
    c: Any,
    *,
    A_eq: Any = None,
    b_eq: Any = None,
    A_ub: Any = None,
    b_ub: Any = None,
    lo: Any = None,
    hi: Any = None,
    cones: Sequence[SOCBlock] = (),
    tol: float | None = None,
    max_iter: int | None = None,
) -> ConicResult:
    """Minimise ``1/2 x^T P x + c^T x`` over rows, a box and Coulomb cones.

    ``lo``/``hi`` may contain ``+-inf``; entries with ``lo == hi`` are fixed by
    equality rows (not by a degenerate pair of inequalities, which an
    interior-point method handles badly).  Preconditions: ``P`` symmetric PSD,
    finite data (checked in :func:`~.solvers.solve_socp`).  Postcondition:
    ``x`` is populated iff the status is OPTIMAL; infeasible or numerically
    failed problems are returned as such and never repaired.
    """
    cc = np.asarray(c, dtype=float)
    n = cc.size
    a_eq, be = _as_rows(A_eq, b_eq, n)
    lo_v = np.full(n, -np.inf) if lo is None else np.asarray(lo, dtype=float)
    hi_v = np.full(n, np.inf) if hi is None else np.asarray(hi, dtype=float)
    if lo_v.shape != (n,) or hi_v.shape != (n,):
        raise ValueError(f"lo and hi must have shape ({n},)")
    fixed_a, fixed_b, box = _fixed_rows(lo_v, hi_v, n)
    if fixed_a.shape[0]:
        a_eq = sp.vstack([a_eq, fixed_a]).tocsr()
        be = np.concatenate([be, fixed_b])
    pairs = [(None if np.isinf(a) else a, None if np.isinf(b) else b) for a, b in box]
    a_ub, bu = _as_rows(A_ub, b_ub, n)
    return solve_socp(
        cc,
        a_eq if a_eq.shape[0] else None,
        be if be.size else None,
        a_ub if a_ub.shape[0] else None,
        bu if bu.size else None,
        list(cones),
        bounds=pairs,
        max_iter=max_iter,
        P=P,
        tol=tol,
    )


def friction_cone_rows(
    n: int, indices: tuple[int, int, int], frame: Any, mu: float
) -> tuple[SOCBlock, sp.csr_matrix, np.ndarray]:
    """Exact Coulomb cone ``mu*fn >= |ft|`` plus the unilateral row ``fn >= 0``.

    ``frame`` has columns (tangent, tangent, normal) in the force coordinates;
    ``indices`` are the three force entries of one contact.  Returns the SOC
    block, and the ``(A_ub, b_ub)`` rows ``-fn <= 0``.  The unilateral row is
    returned separately because it is not implied by the cone when ``mu == 0``.
    """
    f = np.asarray(frame, dtype=float)
    if f.shape != (3, 3):
        raise ValueError(f"frame must be (3, 3), got {f.shape}")
    if not np.isfinite(mu) or mu < 0.0:
        raise ValueError("mu must be finite and non-negative")
    local = np.zeros((3, n))
    local[:, list(indices)] = f.T
    cone_a = np.vstack([-mu * local[2], -local[0], -local[1]])
    block = SOCBlock(A=cone_a, b=np.zeros(3))
    return block, sp.csr_matrix(-local[2:3]), np.zeros(1)
