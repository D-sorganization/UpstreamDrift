"""Trajectory-coupled envelopes (NSX-25, #12568).

The same machinery as the NSX-22 static envelopes -- two LPs per quantity per
frame on one persistent HiGHS model with basis reuse -- on the stacked,
rate-coupled system of :mod:`.trajectory_system`.  A coupled range is always
inside the static range of the same sets, and equals it without coupling.

Cones enter through the declared linearisation of each frame (as in the static
envelopes); use :func:`~.probes.certify` for exact-cone yes/no answers.
Non-claim: ranges are conditional on the declared hypotheses and the stated
rate bounds.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.envelopes import (
    BoundStatus,
    EnvelopeQuantity,
    SignClass,
    _resolve_quantities,
    classify_sign,
)
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.solvers import SolveStatus
from src.shared.python.loop_nullspace.trajectory_system import (
    RateBound,
    StackedSystem,
    SystemSolver,
    stack_system,
)

__all__ = ["TrajectoryEnvelope", "trajectory_envelope"]


@dataclass(frozen=True, eq=False)
class TrajectoryEnvelope:
    """Per-frame ranges over the trajectory-coupled set (``[frame][quantity]``).

    ``lower``/``upper`` are NaN unless ``BOUNDED``.  ``label`` is
    ``TRAJECTORY_COUPLED``; with no coupling it equals the static envelope.
    """

    names: tuple[str, ...]
    units: tuple[str, ...]
    lower: Array
    upper: Array
    status_lower: tuple[tuple[BoundStatus, ...], ...]
    status_upper: tuple[tuple[BoundStatus, ...], ...]
    sign_class: tuple[tuple[SignClass | None, ...], ...]
    hypotheses: tuple[str, ...]
    coupling: tuple[dict[str, Any], ...]
    cone_error_bound: float
    simplex_iterations: int
    label: str = "TRAJECTORY_COUPLED"
    non_claims: tuple[str, ...] = (
        "conditional on the declared hypotheses and the stated rate bounds",
        "cones enter through the declared linearisation of each frame",
    )

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.lower.shape[0])

    def is_subset_of(self, other: Any, *, atol: float = 1e-9) -> bool:
        """True when every bounded range lies inside ``other``'s (same indexing)."""
        lo, up = np.asarray(other.lower), np.asarray(other.upper)
        ok = np.isnan(self.lower) | np.isnan(lo) | (self.lower >= lo - atol)
        ok &= np.isnan(self.upper) | np.isnan(up) | (self.upper <= up + atol)
        return bool(np.all(ok))

    def strictly_narrower_somewhere(self, other: Any, *, atol: float = 1e-9) -> bool:
        """True when some bound is tighter than ``other``'s by more than ``atol``."""
        lo, up = np.asarray(other.lower), np.asarray(other.upper)
        with np.errstate(invalid="ignore"):
            return bool(
                np.any(self.lower > lo + atol) or np.any(self.upper < up - atol)
            )

    def records(
        self, model: ModelProvenance, source: InputProvenance
    ) -> dict[str, QuantityRecord]:
        """One FAMILY_DEPENDENT record per quantity, values ``(frames, 2)``."""
        out = {}
        for j, name in enumerate(self.names):
            vals = np.stack([self.lower[:, j], self.upper[:, j]], axis=1)
            common: dict[str, Any] = {
                "units": self.units[j],
                "model_provenance": model,
                "input_provenance": source,
                "conventions": (self.label,),
            }
            if np.isnan(vals).any():
                out[name] = QuantityRecord(
                    values=np.full(vals.shape, np.nan),
                    allocation_dependence=AllocationDependence.UNAVAILABLE,
                    reason=f"{name}: non-bounded range in some frames",
                    **common,
                )
            else:
                out[name] = QuantityRecord(
                    values=vals,
                    allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
                    hypotheses=self.hypotheses
                    or ("feasible set without declared rows",),
                    **common,
                )
        return out


def trajectory_envelope(
    feasible_sets: Sequence[FeasibleSet],
    quantities: Sequence[EnvelopeQuantity] | Callable[[FeasibleSet], Any],
    *,
    coupling: Sequence[RateBound] = (),
) -> TrajectoryEnvelope:
    """Min and max of each quantity per frame over the coupled set.

    The same machinery as :func:`~.envelopes.static_envelopes`, on the stacked
    system: two LPs per quantity per frame on one persistent HiGHS model with
    basis reuse.  Cones enter through the declared linearisation (like the
    static envelopes), so a coupled range is always inside the static range of
    the same sets.  An empty coupled set gives ``INFEASIBLE`` everywhere.

    Preconditions: as :func:`~.envelopes.static_envelopes` plus valid
    ``coupling``.  Postcondition: ``lower <= upper`` wherever BOUNDED.
    """
    frames = tuple(feasible_sets)
    system = stack_system(frames, coupling, cones="linearized")
    per_frame = [_resolve_quantities(fs, quantities) for fs in frames]
    names = tuple(q.name for q in per_frame[0])
    if any(tuple(q.name for q in qs) != names for qs in per_frame):
        raise ValueError("quantity names differ between frames")
    t_count, n_q = len(frames), len(names)
    lower = np.full((t_count, n_q), np.nan)
    upper = np.full((t_count, n_q), np.nan)
    st_lo = [[BoundStatus.BOUNDED] * n_q for _ in range(t_count)]
    st_up = [[BoundStatus.BOUNDED] * n_q for _ in range(t_count)]
    iters = 0
    with SystemSolver(system) as solver:
        feas = solver.feasibility()
        empty = feas.status is SolveStatus.INFEASIBLE
        for k in range(t_count):
            for j, q in enumerate(per_frame[k]):
                if empty:
                    st_lo[k][j] = st_up[k][j] = BoundStatus.INFEASIBLE
                    continue
                lower[k, j], st_lo[k][j] = _bound(solver, system, k, q, +1.0)
                upper[k, j], st_up[k][j] = _bound(solver, system, k, q, -1.0)
        iters = solver.iterations
    sign = tuple(
        tuple(
            None
            if st_lo[k][j] is BoundStatus.INFEASIBLE
            else classify_sign(lower[k, j], upper[k, j], scale=q.scale)
            for j, q in enumerate(per_frame[k])
        )
        for k in range(t_count)
    )
    cone_err = max(
        (fs.cone_error_bound for fs in frames if fs.cone_error_bound is not None),
        default=math.nan,
    )
    return TrajectoryEnvelope(
        names=names,
        units=tuple(q.units for q in per_frame[0]),
        lower=lower,
        upper=upper,
        status_lower=tuple(tuple(r) for r in st_lo),
        status_upper=tuple(tuple(r) for r in st_up),
        sign_class=sign,
        hypotheses=tuple(dict.fromkeys(system.hypothesis_ids)),
        coupling=tuple(rb.describe() for rb in coupling),
        cone_error_bound=cone_err,
        simplex_iterations=iters,
    )


def _bound(
    solver: SystemSolver,
    system: StackedSystem,
    k: int,
    q: EnvelopeQuantity,
    sign: float,
) -> tuple[float, BoundStatus]:
    """``sign=+1`` minimises ``q`` (lower bound); ``-1`` gives the upper."""
    if not np.any(q.coeff):
        return q.const, BoundStatus.BOUNDED
    out = solver.minimize(sign * system.embed(k, q.coeff))
    if out.status is SolveStatus.UNBOUNDED:
        return math.nan, BoundStatus.UNBOUNDED
    if out.status is not SolveStatus.OPTIMAL:
        return math.nan, BoundStatus.INFEASIBLE
    return q.const + sign * float(out.objective or 0.0), BoundStatus.BOUNDED
