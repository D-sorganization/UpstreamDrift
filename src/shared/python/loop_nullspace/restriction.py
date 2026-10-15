"""Reaction-restriction sensitivity on a fixed welded trajectory (NSX-26, #12569).

The recorded trajectory is welded: ``J_c v = 0`` and ``J_c a + Jdot_v = 0`` for
all six closure rows.  *Restricting* the reaction keeps the kinematics and the
generalised load ``b = M a + h`` and removes closure-wrench components the
hand-club joint is allowed to transmit.  In the canonical grip frame (rows
``fx fy fz tx ty tz``, first axis the shaft ``u``):

* ``WELD6``       transmits all six components;
* ``SHAFT_FREE5`` cannot transmit the twist moment about the shaft (row ``tx``);
* ``BALL3``       transmits the three forces only.

Dropping a row fixes that multiplier at zero, so the admissible reactions, and
every feasible set built on them, shrink along ``WELD6 -> SHAFT_FREE5 ->
BALL3``; envelopes of the restricted kinds lie inside the ``WELD6`` envelope
and a restricted set can be empty.

What this is and is not:

* It is a restriction of the reaction set **on the welded kinematics**.  It is
  not a different closure model: a true alternative grip changes the
  kinematics and must be re-derived (NSX-72).  It is not evidence about the
  true grip mechanics.
* A trajectory recorded under a weld can be **dynamically inconsistent** under
  a less restrictive joint: the unactuated rows may no longer balance
  (:class:`~.decomposition.BalanceInconsistentError`) or the declared priors
  may admit no allocation.  Both are findings, reported per frame as
  :attr:`FrameOutcome.BALANCE_INCONSISTENT` (with the scaled residual) and
  :attr:`FrameOutcome.INFEASIBLE` (with the conflict set).  The balance
  tolerance and the priors are never relaxed to avoid them.
* The shaft axis is not stored in the snapshot.  ``restrict`` checks that the
  bare-club centre of mass (or an explicit ``shaft_axis``) lies on the grip x
  axis before it calls row ``tx`` a shaft twist.

Every output carries the label :data:`RESTRICTION_LABEL`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.admission import AdmittedTrajectory
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
)
from src.shared.python.loop_nullspace.decomposition import (
    AllocationMetric,
    BalanceInconsistentError,
    CharacteristicScales,
    Decomposition,
    WeightsSource,
    decompose,
)
from src.shared.python.loop_nullspace.envelopes import (
    EnvelopeQuantity,
    QuantityFactory,
    static_envelopes,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.restriction_report import (
    NON_CLAIMS,
    RESTRICTION_LABEL,
    ContainmentViolation,
    FrameOutcome,
    KindFrame,
    KindOutcome,
    RestrictionReport,
    containment_violations,
)

__all__ = [
    "NON_CLAIMS",
    "RESTRICTION_LABEL",
    "SHAFT_AXIS_ATOL",
    "ContainmentViolation",
    "FrameOutcome",
    "KindFrame",
    "QuantityBuilder",
    "RestrictionReport",
    "containment_violations",
    "restrict",
    "restriction_rows",
    "restriction_sensitivity",
    "snapshot_quantities",
]

#: Largest component of the shaft direction off the grip x axis that is still
#: read as "the shaft is the grip x axis" (a gross-frame detector: the club
#: centre of mass of a real club sits a few centimetres off its shaft axis).
SHAFT_AXIS_ATOL = 0.1

QuantityBuilder = Callable[
    [FeasibleSet, ClosedLoopSnapshot], Sequence[EnvelopeQuantity]
]
FeasibleSetBuilder = Callable[[Decomposition, ClosedLoopSnapshot], FeasibleSet]

_COMPONENTS: dict[ClosureKind, tuple[str, ...]] = {
    ClosureKind.WELD6: ("fx", "fy", "fz", "tx", "ty", "tz"),
    ClosureKind.SHAFT_FREE5: ("fx", "fy", "fz", "ty", "tz"),
    ClosureKind.BALL3: ("fx", "fy", "fz"),
}
_TWIST = "tx"


def _as_kind(kind: ClosureKind | str) -> ClosureKind:
    if isinstance(kind, ClosureKind):
        return kind
    if isinstance(kind, str):
        try:
            return ClosureKind(kind)
        except ValueError as exc:
            raise ValueError(f"unknown closure kind {kind!r}") from exc
    raise TypeError(f"closure kind must be ClosureKind or str, got {type(kind)!r}")


def restriction_rows(
    source: ClosureKind | str, target: ClosureKind | str
) -> tuple[int, ...]:
    """Row indices of ``source`` that ``target`` keeps (``J_c[rows]``).

    Admissible restrictions are the nesting ``WELD6 -> SHAFT_FREE5 -> BALL3``
    (and the identity): every component of ``target`` must be a component of
    ``source``.  Planar kinds have no shaft-frame components and are refused.
    Raises ``ValueError`` ("not an admissible restriction") otherwise.
    """
    src, tgt = _as_kind(source), _as_kind(target)
    have, want = _COMPONENTS.get(src), _COMPONENTS.get(tgt)
    if have is None or want is None or not set(want) <= set(have):
        raise ValueError(
            f"{src.value} -> {tgt.value} is not an admissible restriction; "
            "admissible: WELD6 -> SHAFT_FREE5 -> BALL3 (rows of the grip frame)"
        )
    return tuple(have.index(name) for name in want)


def _shaft_direction(snapshot: ClosedLoopSnapshot, axis: ArrayLike | None) -> Any:
    if axis is None:
        vec = np.asarray(snapshot.bare_club.com, dtype=float)
        origin = "the bare-club centre of mass"
    else:
        vec = np.asarray(axis, dtype=float)
        origin = "shaft_axis"
    if vec.shape != (3,) or not np.isfinite(vec).all() or np.linalg.norm(vec) <= 0:
        raise ValueError(f"shaft_axis must be a finite nonzero 3-vector ({origin})")
    return vec / np.linalg.norm(vec), origin


def _require_shaft_on_x(
    snapshot: ClosedLoopSnapshot, axis: ArrayLike | None, atol: float
) -> None:
    direction, origin = _shaft_direction(snapshot, axis)
    off_axis = float(np.linalg.norm(direction[1:]))
    if off_axis > atol:
        raise ValueError(
            f"the shaft axis ({origin}) has a component {off_axis:.3g} off the "
            "grip x axis; row 'tx' is then not the twist about the shaft and "
            "SHAFT_FREE5 would drop the wrong component"
        )


def restrict(
    snapshot: ClosedLoopSnapshot,
    kind: ClosureKind | str,
    *,
    shaft_axis: ArrayLike | None = None,
    shaft_atol: float = SHAFT_AXIS_ATOL,
) -> ClosedLoopSnapshot:
    """Snapshot with the closure rows restricted to ``kind`` (same kinematics).

    Selects ``J_c[rows]`` and ``Jdot_v[rows]`` in the canonical grip frame and
    sets ``topology.closure_kind``; ``q, v, a, M, h, S, J_g`` and the loop
    support are unchanged, and the restriction is recorded in the input
    provenance note with :data:`RESTRICTION_LABEL`.

    Preconditions: ``snapshot`` is a :class:`ClosedLoopSnapshot` (``TypeError``);
    ``kind`` is an admissible restriction of the snapshot's kind
    (:func:`restriction_rows`); when the twist row alone is dropped from the moments the shaft axis
    (``shaft_axis`` in grip axes, else the bare-club centre of mass) lies within
    ``shaft_atol`` of the grip x axis.  Postcondition: the result satisfies the
    NSX-10 contracts and every closure level of the input that held on all six
    rows still holds on the kept rows.
    """
    if not isinstance(snapshot, ClosedLoopSnapshot):
        raise TypeError(f"restrict needs a ClosedLoopSnapshot, got {type(snapshot)!r}")
    target = _as_kind(kind)
    source = snapshot.topology.closure_kind
    rows = restriction_rows(source, target)
    kept = _COMPONENTS[target]
    drops_only_twist = _TWIST in _COMPONENTS[source] and _TWIST not in kept
    if drops_only_twist and any(c.startswith("t") for c in kept):
        _require_shaft_on_x(snapshot, shaft_axis, shaft_atol)
    index = list(rows)
    prev = snapshot.input_provenance.note
    note = f"{RESTRICTION_LABEL}: {source.value}->{target.value} rows {list(rows)}"
    return replace(
        snapshot,
        J_c=snapshot.J_c[index],
        Jdot_v=snapshot.Jdot_v[index],
        topology=replace(snapshot.topology, closure_kind=target),
        input_provenance=replace(
            snapshot.input_provenance, note=f"{prev}; {note}" if prev else note
        ),
    )


def snapshot_quantities(factory: QuantityFactory) -> QuantityBuilder:
    """Lift an envelope factory ``fs -> quantities`` to a snapshot-aware builder."""

    def build(fs: FeasibleSet, _snapshot: ClosedLoopSnapshot) -> Sequence[Any]:
        return factory(fs)

    return build


def _frames(trajectory: Any) -> tuple[ClosedLoopSnapshot, ...]:
    if not isinstance(trajectory, AdmittedTrajectory):
        raise TypeError(
            "restriction_sensitivity requires an AdmittedTrajectory (build one "
            "with AdmittedTrajectory.from_report(admit(...))), got "
            f"{type(trajectory).__name__}"
        )
    return trajectory.snapshots


def _check_kinds(
    kinds: Sequence[ClosureKind | str], frames: Sequence[ClosedLoopSnapshot]
) -> tuple[ClosureKind, ...]:
    resolved = tuple(_as_kind(k) for k in kinds)
    if not resolved:
        raise ValueError("kinds must name at least the reference closure kind")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"kinds must be unique, got {[k.value for k in resolved]}")
    for snap in frames:
        if snap.topology.closure_kind is not resolved[0]:
            raise ValueError(
                f"the first kind {resolved[0].value} must be the closure kind of "
                f"every frame, got {snap.topology.closure_kind.value}"
            )
    for kind in resolved[1:]:
        restriction_rows(resolved[0], kind)
    return resolved


def _frame_metric(weights: WeightsSource | None, snap: ClosedLoopSnapshot) -> Any:
    if weights is None:
        return AllocationMetric.identity(snap.topology.n_actuated)
    if callable(weights) and not isinstance(weights, (AllocationMetric, np.ndarray)):
        return weights(snap)
    return weights


def _decompose_kind(
    snaps: Sequence[ClosedLoopSnapshot],
    weights: WeightsSource | None,
    scales: CharacteristicScales | None,
) -> list[Decomposition | BalanceInconsistentError]:
    out: list[Decomposition | BalanceInconsistentError] = []
    for snap in snaps:
        try:
            out.append(
                decompose(snap, weights=_frame_metric(weights, snap), scales=scales)
            )
        except BalanceInconsistentError as err:
            out.append(err)
    return out


def _solve_kind(
    kind: ClosureKind,
    snaps: Sequence[ClosedLoopSnapshot],
    decs: Sequence[Decomposition | BalanceInconsistentError],
    quantities: Sequence[QuantityBuilder],
    feasible_set_of: FeasibleSetBuilder,
) -> KindOutcome:
    sets: list[FeasibleSet] = []
    table: dict[int, list[EnvelopeQuantity]] = {}
    where: list[int] = []
    for i, (snap, dec) in enumerate(zip(snaps, decs, strict=True)):
        if isinstance(dec, BalanceInconsistentError):
            continue
        fs = feasible_set_of(dec, snap)
        table[id(fs)] = [q for build in quantities for q in build(fs, snap)]
        sets.append(fs)
        where.append(i)
    series = static_envelopes(sets, lambda fs: table[id(fs)]) if sets else None
    return KindOutcome.assemble(kind, decs, sets, where, series, table)


def restriction_sensitivity(
    trajectory: AdmittedTrajectory,
    kinds: Sequence[ClosureKind | str],
    quantities: Sequence[QuantityBuilder],
    *,
    feasible_set_of: FeasibleSetBuilder,
    weights: WeightsSource | None = None,
    scales: CharacteristicScales | None = None,
    shaft_axis: ArrayLike | None = None,
) -> RestrictionReport:
    """Envelopes and feasibility per closure kind on one fixed welded trajectory.

    ``kinds[0]`` is the reference (the trajectory's own kind, ``WELD6``); every
    other kind must be an admissible restriction of it.  ``quantities`` are
    snapshot-aware builders (:func:`snapshot_quantities` lifts an envelope
    factory); quantity names must agree across kinds.  ``feasible_set_of(dec,
    snapshot)`` builds the declared feasible set of a decomposed frame: the
    priors live there and are identical for every kind (this function never
    changes them).  ``weights`` selects the particular solution only (default
    ``C-METRIC-IDENTITY``).

    Preconditions: ``trajectory`` is an :class:`AdmittedTrajectory`
    (``TypeError``); ``kinds`` are unique, non-empty and the first equals every
    frame's kind; ``quantities`` is non-empty; ``feasible_set_of`` is callable.
    Postconditions: no input is mutated; a frame whose unactuated rows cannot be
    balanced is reported ``BALANCE_INCONSISTENT`` with its scaled residual
    (``BALANCE_TOL`` unchanged); an empty set is ``INFEASIBLE`` with its
    conflict set; the report is labelled :data:`RESTRICTION_LABEL`.
    """
    frames = _frames(trajectory)
    resolved = _check_kinds(kinds, frames)
    builders = list(quantities)
    if not builders:
        raise ValueError("quantities must contain at least one builder")
    if not callable(feasible_set_of):
        raise TypeError("feasible_set_of must be callable (decomposition, snapshot)")
    outcomes = []
    for kind in resolved:
        snaps = [restrict(s, kind, shaft_axis=shaft_axis) for s in frames]
        decs = _decompose_kind(snaps, weights, scales)
        outcomes.append(_solve_kind(kind, snaps, decs, builders, feasible_set_of))
    first = frames[0]
    return RestrictionReport.from_outcomes(
        outcomes,
        n_frames=len(frames),
        model=first.model_provenance,
        source=first.input_provenance,
    )
