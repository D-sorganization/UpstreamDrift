"""Result types and comparison statistics for reaction restriction (NSX-26).

Split from :mod:`restriction` to keep both small.  Everything here is
data-only plus pure comparisons of per-kind envelope tables; the solving lives
in :mod:`restriction`.  Every table is indexed ``[frame][quantity]`` over *all*
frames of the trajectory: a frame without an envelope (empty set, or unactuated
rows that do not balance) is NaN with a ``None`` or ``INFEASIBLE`` status.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.contracts import ClosureKind
from src.shared.python.loop_nullspace.decomposition import (
    BalanceInconsistentError,
    Decomposition,
)
from src.shared.python.loop_nullspace.envelopes import (
    BoundStatus,
    EnvelopeSeries,
    SignClass,
)
from src.shared.python.loop_nullspace.evidence import (
    Array,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet

__all__ = [
    "NEST_RTOL",
    "NON_CLAIMS",
    "RESTRICTION_LABEL",
    "AlignedSeries",
    "ContainmentViolation",
    "FrameOutcome",
    "KindFrame",
    "KindOutcome",
    "RestrictionReport",
    "WidthRatio",
    "containment_violations",
]

RESTRICTION_LABEL = "REACTION_RESTRICTION_SENSITIVITY"
#: Containment of restricted envelopes is checked to ``NEST_RTOL * (scale + |bound|)``,
#: the LP solution accuracy; it is a check tolerance, not a balance tolerance.
NEST_RTOL = 1e-6
_NARROWER = 1e-6
NON_CLAIMS = (
    "not evidence about the true grip mechanics",
    "a restriction of the reaction set on the welded kinematics, not a different "
    "closure model (alternative closures re-derive the kinematics: NSX-72)",
    "a weld-consistent trajectory may be dynamically inconsistent under a less "
    "restrictive joint: balance residuals and empty sets are findings, never repaired",
    "conditional on the declared hypotheses; capacities are priors",
)


class FrameOutcome(Enum):
    """What one closure kind makes of one frame."""

    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    BALANCE_INCONSISTENT = "BALANCE_INCONSISTENT"


@dataclass(frozen=True)
class KindFrame:
    """Outcome of one frame under one kind.

    ``conflict_set`` is the inclusion-minimal hypothesis set whose rows are
    jointly infeasible (``INFEASIBLE`` only); ``balance_residual_scaled`` is the
    scaled residual of the unactuated rows (``BALANCE_INCONSISTENT`` only, else
    NaN).  ``nullity`` and ``hypothesis_ids`` are those of the decomposition.
    """

    kind: ClosureKind
    index: int
    outcome: FrameOutcome
    conflict_set: tuple[str, ...] = ()
    balance_residual_scaled: float = math.nan
    nullity: int | None = None
    hypothesis_ids: tuple[str, ...] = ()
    rests_on_unverified: bool = False


@dataclass(frozen=True, eq=False)
class AlignedSeries:
    """Envelope bounds of every frame (NaN where there is none)."""

    names: tuple[str, ...]
    units: tuple[str, ...]
    scales: tuple[float, ...]
    lower: Array
    upper: Array
    status_lower: tuple[tuple[BoundStatus | None, ...], ...]
    status_upper: tuple[tuple[BoundStatus | None, ...], ...]
    sign_class: tuple[tuple[SignClass | None, ...], ...]

    def width(self) -> Array:
        """``upper - lower`` (NaN where either bound is not BOUNDED)."""
        return np.asarray(self.upper - self.lower)


@dataclass(frozen=True)
class WidthRatio:
    """Restricted over reference envelope width on frames bounded in both."""

    frames: int
    median: float
    minimum: float
    maximum: float
    narrower_frames: int


@dataclass(frozen=True)
class ContainmentViolation:
    """A restricted envelope that is not inside the reference envelope."""

    frame: int
    quantity: str
    excess: float
    detail: str
    kind: str = ""


def _tol(scale: float, bound: float) -> float:
    return NEST_RTOL * (scale + abs(bound))


def _side_violation(
    ref_value: float, value: float, ref_ok: bool, ok: bool, lower: bool, tol: float
) -> float | None:
    """Excess of a restricted bound outside the reference bound (None if inside)."""
    if not ref_ok:
        return None
    if not ok:
        return math.inf
    excess = (ref_value - value) if lower else (value - ref_value)
    return float(excess) if excess > tol else None


def containment_violations(
    reference: AlignedSeries, restricted: AlignedSeries
) -> tuple[ContainmentViolation, ...]:
    """Frames and quantities where ``restricted`` is not inside ``reference``.

    A restricted set must be empty whenever the reference set is empty
    (``"<feasibility>"``); otherwise each bounded reference bound must contain
    the restricted bound to ``NEST_RTOL * (scale + |bound|)``.  Frames where
    the reference has no envelope (balance-inconsistent) are not compared.
    """
    if reference.names != restricted.names:
        raise ValueError("series must carry the same quantities")
    out: list[ContainmentViolation] = []
    bounded = BoundStatus.BOUNDED
    for t in range(reference.lower.shape[0]):
        ref_row, res_row = reference.status_lower[t], restricted.status_lower[t]
        if ref_row and ref_row[0] is BoundStatus.INFEASIBLE and res_row[0] is bounded:
            out.append(
                ContainmentViolation(t, "<feasibility>", math.inf, "reference is empty")
            )
            continue
        for j, name in enumerate(reference.names):
            for lower in (True, False):
                st_ref = (reference.status_lower if lower else reference.status_upper)[
                    t
                ][j]
                st_res = (
                    restricted.status_lower if lower else restricted.status_upper
                )[t][j]
                if st_res is None or st_res is BoundStatus.INFEASIBLE:
                    continue  # no envelope, or empty: empty is contained
                arr_ref = reference.lower if lower else reference.upper
                arr_res = restricted.lower if lower else restricted.upper
                excess = _side_violation(
                    float(arr_ref[t, j]),
                    float(arr_res[t, j]),
                    st_ref is bounded,
                    st_res is bounded,
                    lower,
                    _tol(reference.scales[j], float(arr_ref[t, j])),
                )
                if excess is not None:
                    side = "lower" if lower else "upper"
                    out.append(
                        ContainmentViolation(t, name, excess, f"{side} bound outside")
                    )
    return tuple(out)


def _stats(ref: AlignedSeries, other: AlignedSeries, j: int) -> WidthRatio:
    w_ref, w_other = ref.width()[:, j], other.width()[:, j]
    floor = 1e-9 * ref.scales[j]
    both = np.isfinite(w_ref) & np.isfinite(w_other) & (w_ref > floor)
    if not both.any():
        return WidthRatio(0, math.nan, math.nan, math.nan, 0)
    ratio = w_other[both] / w_ref[both]
    return WidthRatio(
        int(both.sum()),
        float(np.median(ratio)),
        float(ratio.min()),
        float(ratio.max()),
        int(np.sum(ratio < 1.0 - _NARROWER)),
    )


@dataclass(frozen=True, eq=False)
class KindOutcome:
    """Per-frame outcomes, aligned envelope table and raw series of one kind."""

    kind: ClosureKind
    frames: tuple[KindFrame, ...]
    aligned: AlignedSeries
    raw: EnvelopeSeries | None
    raw_index: tuple[int, ...]

    @classmethod
    def assemble(
        cls,
        kind: ClosureKind,
        decs: list[Decomposition | BalanceInconsistentError],
        sets: list[FeasibleSet],
        where: list[int],
        series: EnvelopeSeries | None,
        table: dict[int, list[Any]],
    ) -> KindOutcome:
        """Combine decompositions, feasible sets and their envelope series."""
        n = len(decs)
        names = series.names if series else ()
        units = series.units if series else ()
        scales = tuple(q.scale for q in table[id(sets[0])]) if sets else ()
        q = len(names)
        lower, upper = np.full((n, q), np.nan), np.full((n, q), np.nan)
        s_lo = [[None] * q for _ in range(n)]
        s_hi = [[None] * q for _ in range(n)]
        sign: list[list[SignClass | None]] = [[None] * q for _ in range(n)]
        frames = [None] * n
        for i, dec in enumerate(decs):
            if isinstance(dec, BalanceInconsistentError):
                frames[i] = KindFrame(
                    kind,
                    i,
                    FrameOutcome.BALANCE_INCONSISTENT,
                    balance_residual_scaled=dec.residual_scaled,
                )
        for k, i in enumerate(where):
            assert series is not None
            lower[i], upper[i] = series.lower[k], series.upper[k]
            s_lo[i], s_hi[i] = (
                list(series.status_lower[k]),
                list(series.status_upper[k]),
            )
            sign[i] = list(series.sign_class[k])
            frames[i] = _decomposed_frame(kind, i, decs[i], sets[k], s_lo[i])
        aligned = AlignedSeries(
            names,
            units,
            scales,
            lower,
            upper,
            tuple(map(tuple, s_lo)),
            tuple(map(tuple, s_hi)),
            tuple(map(tuple, sign)),
        )
        return cls(kind, tuple(frames), aligned, series, tuple(where))  # type: ignore[arg-type]


def _decomposed_frame(
    kind: ClosureKind,
    index: int,
    dec: Decomposition,
    fs: FeasibleSet,
    status_row: list[BoundStatus | None],
) -> KindFrame:
    empty = bool(status_row) and status_row[0] is BoundStatus.INFEASIBLE
    return KindFrame(
        kind,
        index,
        FrameOutcome.INFEASIBLE if empty else FrameOutcome.FEASIBLE,
        conflict_set=fs.conflict_set() if empty else (),
        nullity=dec.nullity,
        hypothesis_ids=tuple(fs.hypothesis_ids),
        rests_on_unverified=bool(fs.rests_on_unverified),
    )


def _float(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


@dataclass(frozen=True, eq=False)
class RestrictionReport:
    """Envelopes and feasibility of one welded trajectory under several kinds.

    ``kinds[0]`` is the reference.  Labelled :data:`RESTRICTION_LABEL`; the
    :attr:`non_claims` say what it is not.  ``admitted`` records that the
    trajectory came through admission (NSX-14).
    """

    kinds: tuple[ClosureKind, ...]
    outcomes: dict[ClosureKind, KindOutcome]
    n_frames: int
    model: ModelProvenance
    source: InputProvenance
    admitted: bool = True
    label: str = RESTRICTION_LABEL
    non_claims: tuple[str, ...] = NON_CLAIMS

    @classmethod
    def from_outcomes(
        cls,
        outcomes: list[KindOutcome],
        *,
        n_frames: int,
        model: ModelProvenance,
        source: InputProvenance,
    ) -> RestrictionReport:
        """Build from per-kind outcomes (the first is the reference)."""
        template = next((o.aligned for o in outcomes if o.aligned.names), None)
        fixed = []
        for o in outcomes:
            if template is not None and not o.aligned.names:
                o = replace(o, aligned=_empty_like(template, n_frames))
            elif template is not None and o.aligned.names != template.names:
                raise ValueError(
                    "quantity names differ between closure kinds: "
                    f"{o.aligned.names} vs {template.names}"
                )
            fixed.append(o)
        return cls(
            tuple(o.kind for o in fixed),
            {o.kind: o for o in fixed},
            n_frames,
            model,
            source,
        )

    # ------------------------------------------------------------ lookups
    @property
    def reference(self) -> ClosureKind:
        """The reference kind (``kinds[0]``)."""
        return self.kinds[0]

    def series(self, kind: ClosureKind) -> AlignedSeries:
        """Aligned envelope table of ``kind``."""
        return self.outcomes[kind].aligned

    def frame(self, kind: ClosureKind, index: int) -> KindFrame:
        """Outcome of frame ``index`` under ``kind``."""
        return self.outcomes[kind].frames[index]

    def _count(self, outcome: FrameOutcome) -> dict[ClosureKind, int]:
        return {
            k: sum(f.outcome is outcome for f in self.outcomes[k].frames)
            for k in self.kinds
        }

    def feasible_counts(self) -> dict[ClosureKind, int]:
        """Frames with a non-empty feasible set, per kind."""
        return self._count(FrameOutcome.FEASIBLE)

    def infeasible_counts(self) -> dict[ClosureKind, int]:
        """Frames whose declared set is empty, per kind."""
        return self._count(FrameOutcome.INFEASIBLE)

    def balance_inconsistent_counts(self) -> dict[ClosureKind, int]:
        """Frames whose unactuated rows cannot be balanced, per kind."""
        return self._count(FrameOutcome.BALANCE_INCONSISTENT)

    def conflict_hypotheses(self, kind: ClosureKind) -> dict[str, int]:
        """How often each hypothesis id sits in a conflict set of ``kind``."""
        out: dict[str, int] = {}
        for f in self.outcomes[kind].frames:
            for h in f.conflict_set:
                out[h] = out.get(h, 0) + 1
        return out

    def lost_frames(self, kind: ClosureKind) -> tuple[int, ...]:
        """Frames feasible under the reference but not under ``kind``."""
        ref = self.outcomes[self.reference].frames
        return tuple(
            f.index
            for f, r in zip(self.outcomes[kind].frames, ref, strict=True)
            if r.outcome is FrameOutcome.FEASIBLE
            and f.outcome is not FrameOutcome.FEASIBLE
        )

    # ------------------------------------------------------- comparisons
    def _index(self, name: str) -> int:
        names = self.series(self.reference).names
        if name not in names:
            raise ValueError(f"unknown quantity {name!r}; have {names}")
        return names.index(name)

    def width_ratio(self, kind: ClosureKind, name: str) -> WidthRatio:
        """Width of ``kind`` over the reference width for quantity ``name``."""
        return _stats(self.series(self.reference), self.series(kind), self._index(name))

    def sign_class_changes(self, kind: ClosureKind, name: str) -> dict[str, int]:
        """Count of ``REF->KIND`` sign-class changes on frames bounded in both."""
        j = self._index(name)
        ref, other = self.series(self.reference), self.series(kind)
        out: dict[str, int] = {}
        for t in range(self.n_frames):
            a, b = ref.sign_class[t][j], other.sign_class[t][j]
            if a is not None and b is not None and a is not b:
                key = f"{a.name}->{b.name}"
                out[key] = out.get(key, 0) + 1
        return out

    def nesting_violations(self) -> tuple[ContainmentViolation, ...]:
        """Containment failures between every ordered pair of admissible kinds."""
        from src.shared.python.loop_nullspace.restriction import (  # noqa: PLC0415
            restriction_rows,
        )

        found: list[ContainmentViolation] = []
        for i, outer in enumerate(self.kinds):
            for inner in self.kinds[i + 1 :]:
                try:
                    restriction_rows(outer, inner)
                except ValueError:
                    continue
                bad = containment_violations(self.series(outer), self.series(inner))
                found += [replace(v, kind=f"{outer.value}>={inner.value}") for v in bad]
        return tuple(found)

    # ----------------------------------------------------------- outputs
    def records(
        self, model: ModelProvenance, source: InputProvenance
    ) -> dict[ClosureKind, dict[str, QuantityRecord]]:
        """One labelled record per kind and quantity (envelope bounds per frame).

        A kind with any non-bounded frame has ``UNAVAILABLE`` records with the
        reason stated; the finite bounds stay in :meth:`series`.
        """
        out: dict[ClosureKind, dict[str, QuantityRecord]] = {}
        for kind in self.kinds:
            o = self.outcomes[kind]
            out[kind] = _kind_records(o, model, source)
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible summary (NaN becomes ``None``)."""
        ref = self.series(self.reference)
        kinds: dict[str, Any] = {}
        for kind in self.kinds:
            width = {n: _ratio_dict(self.width_ratio(kind, n)) for n in ref.names}
            kinds[kind.value] = {
                "feasible": self.feasible_counts()[kind],
                "infeasible": self.infeasible_counts()[kind],
                "balance_inconsistent": self.balance_inconsistent_counts()[kind],
                "conflict_hypotheses": self.conflict_hypotheses(kind),
                "lost_frames": list(self.lost_frames(kind)),
                "width_ratio": width,
                "sign_class_changes": {
                    n: self.sign_class_changes(kind, n) for n in ref.names
                },
            }
        return {
            "label": self.label,
            "reference": self.reference.value,
            "admitted": self.admitted,
            "n_frames": self.n_frames,
            "quantities": list(ref.names),
            "kinds": kinds,
            "nesting_violations": len(self.nesting_violations()),
            "non_claims": list(self.non_claims),
        }


def _ratio_dict(r: WidthRatio) -> dict[str, Any]:
    return {
        "frames": r.frames,
        "median": _float(r.median),
        "min": _float(r.minimum),
        "max": _float(r.maximum),
        "narrower_frames": r.narrower_frames,
    }


def _empty_like(template: AlignedSeries, n: int) -> AlignedSeries:
    q = len(template.names)
    nan = np.full((n, q), np.nan)
    none = tuple((None,) * q for _ in range(n))
    return AlignedSeries(
        template.names,
        template.units,
        template.scales,
        nan,
        nan.copy(),
        none,
        none,
        none,
    )


def _kind_records(
    outcome: KindOutcome, model: ModelProvenance, source: InputProvenance
) -> dict[str, QuantityRecord]:
    from src.shared.python.loop_nullspace.evidence import (  # noqa: PLC0415
        AllocationDependence,
    )

    names = outcome.aligned.names
    tags = (RESTRICTION_LABEL, f"restriction_kind={outcome.kind.value}")
    if outcome.raw is not None and len(outcome.raw_index) == len(outcome.frames):
        base = outcome.raw.records(model, source)
    else:
        base = {}
    result: dict[str, QuantityRecord] = {}
    for j, name in enumerate(names):
        if name in base:
            rec = base[name]
            result[name] = replace(rec, conventions=(*rec.conventions, *tags))
            continue
        result[name] = QuantityRecord(
            values=np.full((len(outcome.frames), 2), np.nan),
            units=outcome.aligned.units[j],
            allocation_dependence=AllocationDependence.UNAVAILABLE,
            model_provenance=model,
            input_provenance=source,
            conventions=tags,
            reason=(
                f"{name}: {outcome.kind.value} has frames without an envelope "
                "(empty set or unbalanced unactuated rows); see the report outcomes"
            ),
        )
    return result
