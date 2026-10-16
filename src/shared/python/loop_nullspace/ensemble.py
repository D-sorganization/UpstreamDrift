"""Kinematic ensemble and envelope union/intersection (NSX-61, #12586).

The kinematic null space of a closed-loop swing is often larger than the
dynamic one, so a single reconstructed trajectory understates the uncertainty.
:func:`kinematic_ensemble` perturbs an admitted nominal trajectory, projects
every sample onto the closure (``admit(repair="project_to_closure")``, which
delegates to ``same_input.closure.project_to_closure``) and keeps only samples
that pass admission; the others are counted by reason, never dropped silently.

The perturbation law is a declared hypothesis (``H-KINEMATIC-ENSEMBLE-PRIOR``):
a time-constant joint offset ``delta ~ N(0, sigma_q^2 I)`` restricted to the
tangent null space of each frame's closure Jacobian (the same SVD basis the
club-only proposals use), then projected.  It is a modelling choice, not an
estimate of measurement error, and the ensemble is not a confidence set.

:func:`envelope_union_intersection` combines per-sample :class:`EnvelopeSeries`
into labelled interval hulls: the union is the smallest interval containing
every member, the intersection the interval contained in every member (empty
when members are disjoint).  Static per-frame envelopes stay over-approximations.
No engine is imported.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace import registry
from src.shared.python.loop_nullspace.admission import (
    AdmissionGates,
    AdmittedTrajectory,
    Rebuild,
    VerdictKind,
    admit,
)
from src.shared.python.loop_nullspace.envelopes import BoundStatus, EnvelopeSeries
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
    family_range_record,
)
from src.shared.python.motion_matching.club_only.nullspace_proposals import (
    analyze_grip_jacobian_nullspace,
)
from src.shared.python.motion_matching.same_input.closure import ClosurePlant

logger = logging.getLogger(__name__)

__all__ = [
    "ENSEMBLE_HYPOTHESIS_ID",
    "EnsembleEnvelope",
    "EnsembleResult",
    "EnsembleSource",
    "SamplingHypothesis",
    "ensemble_envelope",
    "envelope_union_intersection",
    "kinematic_ensemble",
]

ENSEMBLE_HYPOTHESIS_ID = "H-KINEMATIC-ENSEMBLE-PRIOR"
HULL_LABEL = "ENSEMBLE-INTERVAL-HULL"
_EMPTY_RTOL = 1e-9
#: Reason keys, in the priority used to pick the single counted reason.  The
#: unrepairable gates come first: admission skips repair when one fails, so a
#: closure residual is then a symptom, not the cause.
_REASONS = (
    ("derivative mismatch", "derivative"),
    ("time base", "time_base"),
    ("stance contact", "contact"),
    ("schedule names unknown contact", "contact"),
    ("position closure residual", "position"),
    ("velocity closure residual", "velocity"),
    ("acceleration closure residual", "acceleration"),
)


@dataclass(frozen=True)
class SamplingHypothesis:
    """The declared perturbation law (``H-KINEMATIC-ENSEMBLE-PRIOR``).

    ``sigma_q`` is the standard deviation of the per-coordinate offset in the
    coordinate's own units (rad or m).
    """

    sigma_q: float
    hypothesis_id: str = ENSEMBLE_HYPOTHESIS_ID

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma_q) or self.sigma_q <= 0.0:
            raise ValueError(f"sigma_q must be positive and finite, got {self.sigma_q}")
        registry.hypothesis(self.hypothesis_id)  # must be declared (ValueError)


@dataclass(frozen=True)
class EnsembleSource:
    """What the ensemble perturbs and how it is re-admitted.

    ``gates`` default to the nominal trajectory's own gates; ``rebuild`` makes
    a snapshot at a given ``(q, v, a)`` (as for :func:`admit`).
    """

    nominal: AdmittedTrajectory
    plant: ClosurePlant
    rebuild: Rebuild
    sampling: SamplingHypothesis
    gates: AdmissionGates | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.nominal, AdmittedTrajectory):
            raise TypeError("nominal must be an AdmittedTrajectory")
        if not isinstance(self.sampling, SamplingHypothesis):
            raise TypeError("sampling must be a SamplingHypothesis")

    @property
    def effective_gates(self) -> AdmissionGates:
        """The gates used for re-admission."""
        if self.gates is not None:
            return self.gates
        nominal_report = self.nominal.report
        return nominal_report.gates


@dataclass(frozen=True)
class EnsembleResult:
    """Accepted samples plus the full rejection accounting.

    ``rejected_by_reason`` counts each rejected sample once, under its first
    failing reason (``position``, ``velocity``, ``acceleration``,
    ``derivative``, ``time_base``, ``contact``) or ``projection_failed`` when
    the closure projection did not converge.
    """

    accepted: tuple[AdmittedTrajectory, ...]
    sample_indices: tuple[int, ...]
    n_requested: int
    seed: int
    sigma_q: float
    hypothesis_id: str
    rejected_by_reason: dict[str, int] = field(default_factory=dict)
    n_repaired: int = 0

    @property
    def n_accepted(self) -> int:
        """Number of admitted samples."""
        return len(self.accepted)

    @property
    def n_rejected(self) -> int:
        """Number of rejected samples."""
        return sum(self.rejected_by_reason.values())

    @property
    def rejection_rate(self) -> float:
        """``n_rejected / n_requested``."""
        return self.n_rejected / self.n_requested


# ---------------------------------------------------------------- sampling --
def _tangent_offset(jac: Array, draw: Array) -> Array:
    basis = analyze_grip_jacobian_nullspace(jac).basis
    if basis.shape[1] < 1:
        raise ValueError("closure Jacobian has no null space to sample")
    return basis @ (basis.T @ draw)


def _reason_key(reasons: Sequence[str]) -> str:
    for needle, key in _REASONS:
        if any(needle in r for r in reasons):
            return key
    return "other"


def _perturbed(source: EnsembleSource, draw: Array) -> list:
    return [
        source.rebuild(
            snap.q + _tangent_offset(snap.closure_rows(), draw), snap.v, snap.a
        )
        for snap in source.nominal.snapshots
    ]


def kinematic_ensemble(source: EnsembleSource, n: int, seed: int) -> EnsembleResult:
    """Draw ``n`` perturbed trajectories, project and admit them.

    Preconditions (``ValueError``/``TypeError``): ``n >= 1``, integer ``seed``,
    a closure Jacobian with a non-empty null space.  Postconditions: the draws
    depend only on ``(seed, sigma_q, nominal)`` (sample ``k`` uses row ``k`` of
    one seeded normal matrix, so rejections never shift later samples);
    ``n_accepted + n_rejected == n``; every accepted trajectory is a real
    ``AdmittedTrajectory`` (REPAIRED samples count in ``n_repaired``).
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int")
    report = source.nominal.report
    nq = source.nominal.snapshots[0].q.size
    draws = source.sampling.sigma_q * np.random.default_rng(seed).normal(size=(n, nq))
    accepted, indices, repaired = [], [], 0
    rejected: Counter[str] = Counter()
    for k in range(n):
        try:
            rep = admit(
                _perturbed(source, draws[k]),
                gates=source.effective_gates,
                times=report.times,
                plant=source.plant,
                repair="project_to_closure",
                rebuild=source.rebuild,
            )
        except ArithmeticError:
            rejected["projection_failed"] += 1
            continue
        if rep.verdict.kind is VerdictKind.REJECTED:
            rejected[_reason_key(rep.verdict.reasons)] += 1
            continue
        repaired += rep.verdict.kind is VerdictKind.REPAIRED
        accepted.append(AdmittedTrajectory.from_report(rep))
        indices.append(k)
    logger.info("kinematic ensemble: %d/%d admitted", len(accepted), n)
    return EnsembleResult(
        accepted=tuple(accepted),
        sample_indices=tuple(indices),
        n_requested=n,
        seed=seed,
        sigma_q=source.sampling.sigma_q,
        hypothesis_id=source.sampling.hypothesis_id,
        rejected_by_reason=dict(rejected),
        n_repaired=repaired,
    )


# ------------------------------------------------------- union/intersection --
Status = tuple[tuple[BoundStatus, ...], ...]


@dataclass(frozen=True, eq=False)
class EnsembleEnvelope:
    """Labelled interval hulls of member envelopes, indexed ``[frame, quantity]``.

    Bounds are NaN unless the matching status is ``BOUNDED``.  An empty
    intersection has ``INFEASIBLE`` status and ``intersection_empty`` set.
    """

    names: tuple[str, ...]
    units: tuple[str, ...]
    member_labels: tuple[str, ...]
    hypotheses: tuple[str, ...]
    union_lower: Array
    union_upper: Array
    union_status_lower: Status
    union_status_upper: Status
    intersection_lower: Array
    intersection_upper: Array
    intersection_status: Status
    intersection_empty: np.ndarray
    label: str = HULL_LABEL
    non_claims: tuple[str, ...] = (
        "interval hull of member envelopes: not a probability or confidence set",
        "members are draws from a declared hypothesis, not measured uncertainty",
        "static per-frame sets: over-approximate trajectory-feasible ranges",
    )

    @property
    def n_members(self) -> int:
        """Number of member envelopes."""
        return len(self.member_labels)

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.union_lower.shape[0])

    def records(
        self, model: ModelProvenance, source: InputProvenance
    ) -> dict[str, QuantityRecord]:
        """``<name>:union`` and ``<name>:intersection`` records, ``(frames, 2)``."""
        out = {}
        for j, name in enumerate(self.names):
            pairs = {
                "union": (
                    self.union_lower[:, j],
                    self.union_upper[:, j],
                    [
                        s is BoundStatus.BOUNDED
                        for t in range(self.n_frames)
                        for s in (
                            self.union_status_lower[t][j],
                            self.union_status_upper[t][j],
                        )
                    ],
                ),
                "intersection": (
                    self.intersection_lower[:, j],
                    self.intersection_upper[:, j],
                    [
                        self.intersection_status[t][j] is BoundStatus.BOUNDED
                        for t in range(self.n_frames)
                    ],
                ),
            }
            for kind, (lo, hi, ok) in pairs.items():
                out[f"{name}:{kind}"] = self._record(
                    f"{name}:{kind}", lo, hi, all(ok), j, model, source
                )
        return out

    def _record(
        self,
        key: str,
        lower: Array,
        upper: Array,
        bounded: bool,
        j: int,
        model: ModelProvenance,
        source: InputProvenance,
    ) -> QuantityRecord:
        vals = np.stack([lower, upper], axis=1)
        return family_range_record(
            vals,
            unavailable_reason=(
                None if bounded else f"{key}: unbounded or empty in some frames"
            ),
            hypotheses=self.hypotheses,
            units=self.units[j],
            model=model,
            source=source,
            label=self.label,
        )


def _interval(series: EnvelopeSeries, t: int, j: int) -> tuple[float, float] | None:
    """Member interval at ``(t, j)``; ``None`` = empty; unbounded side = +-inf."""
    sl, su = series.status_lower[t][j], series.status_upper[t][j]
    if BoundStatus.INFEASIBLE in (sl, su):
        return None
    lo = -math.inf if sl is BoundStatus.UNBOUNDED else float(series.lower[t, j])
    hi = math.inf if su is BoundStatus.UNBOUNDED else float(series.upper[t, j])
    return lo, hi


def _cell(
    intervals: list[tuple[float, float] | None],
) -> tuple[tuple[float, float, BoundStatus, BoundStatus], tuple[float, float, bool]]:
    """Union ``(lo, hi, status_lo, status_hi)`` and intersection of one cell."""
    live = [iv for iv in intervals if iv is not None]
    if live:
        u_lo, u_hi = min(i[0] for i in live), max(i[1] for i in live)
    else:
        u_lo, u_hi = math.nan, math.nan
    if not live:
        union = (math.nan, math.nan, BoundStatus.INFEASIBLE, BoundStatus.INFEASIBLE)
    else:
        union = (
            u_lo if math.isfinite(u_lo) else math.nan,
            u_hi if math.isfinite(u_hi) else math.nan,
            BoundStatus.BOUNDED if math.isfinite(u_lo) else BoundStatus.UNBOUNDED,
            BoundStatus.BOUNDED if math.isfinite(u_hi) else BoundStatus.UNBOUNDED,
        )
    if len(live) < len(intervals):
        return union, (math.nan, math.nan, True)
    i_lo, i_hi = max(i[0] for i in live), min(i[1] for i in live)
    tol = _EMPTY_RTOL * max(
        1.0, *(abs(x) for x in (i_lo, i_hi) if math.isfinite(x)), 1.0
    )
    if i_lo > i_hi + tol:
        return union, (math.nan, math.nan, True)
    return union, (i_lo, i_hi, False)


def _check_members(
    members: Sequence[EnvelopeSeries], labels: Sequence[str] | None
) -> tuple[str, ...]:
    if len(members) < 1:
        raise ValueError("at least one member envelope is required")
    for m in members:
        if not isinstance(m, EnvelopeSeries):
            raise TypeError(f"members must be EnvelopeSeries, got {type(m)!r}")
    first = members[0]
    for m in members:
        if m.names != first.names or m.lower.shape != first.lower.shape:
            raise ValueError(
                "members must share quantity names and the same (frames, quantities) shape"
            )
    out = (
        tuple(labels)
        if labels is not None
        else tuple(f"member-{i}" for i in range(len(members)))
    )
    if len(out) != len(members) or len(set(out)) != len(out):
        raise ValueError("labels must be unique, one per member")
    return out


def envelope_union_intersection(
    members: Sequence[EnvelopeSeries],
    labels: Sequence[str] | None = None,
    *,
    hypothesis_ids: Sequence[str] = (ENSEMBLE_HYPOTHESIS_ID,),
) -> EnsembleEnvelope:
    """Interval-hull union and intersection of member envelopes.

    Preconditions: at least one :class:`EnvelopeSeries`, the same names and
    shape, unique labels (``ValueError``/``TypeError``).  Postconditions:
    wherever a bound is ``BOUNDED`` the union contains every non-empty member
    interval and the intersection lies inside every member; an INFEASIBLE
    member is skipped by the union and empties the intersection; a disjoint
    intersection is reported empty, never raised.  The union bound is
    unbounded if any live member is.
    """
    names_labels = _check_members(members, labels)
    first = members[0]
    f, q = first.lower.shape
    u_lo, u_hi = np.full((f, q), np.nan), np.full((f, q), np.nan)
    i_lo, i_hi = np.full((f, q), np.nan), np.full((f, q), np.nan)
    empty = np.zeros((f, q), dtype=bool)
    st_lo: list[list[BoundStatus]] = []
    st_hi: list[list[BoundStatus]] = []
    st_i: list[list[BoundStatus]] = []
    for t in range(f):
        row: tuple[list[BoundStatus], list[BoundStatus], list[BoundStatus]] = (
            [],
            [],
            [],
        )
        for j in range(q):
            union, inter = _cell([_interval(m, t, j) for m in members])
            u_lo[t, j], u_hi[t, j] = union[0], union[1]
            row[0].append(union[2])
            row[1].append(union[3])
            i_lo[t, j], i_hi[t, j], empty[t, j] = inter
            if empty[t, j]:
                row[2].append(BoundStatus.INFEASIBLE)
            else:
                bounded = math.isfinite(i_lo[t, j]) and math.isfinite(i_hi[t, j])
                row[2].append(BoundStatus.BOUNDED if bounded else BoundStatus.UNBOUNDED)
        st_lo.append(row[0])
        st_hi.append(row[1])
        st_i.append(row[2])
    hyps: dict[str, None] = dict.fromkeys(hypothesis_ids)
    for m in members:
        for ids in m.frame_hypothesis_ids:
            hyps.update(dict.fromkeys(ids))
    return EnsembleEnvelope(
        names=first.names,
        units=first.units,
        member_labels=names_labels,
        hypotheses=tuple(hyps),
        union_lower=u_lo,
        union_upper=u_hi,
        union_status_lower=tuple(tuple(r) for r in st_lo),
        union_status_upper=tuple(tuple(r) for r in st_hi),
        intersection_lower=i_lo,
        intersection_upper=i_hi,
        intersection_status=tuple(tuple(r) for r in st_i),
        intersection_empty=empty,
    )


def ensemble_envelope(
    result: EnsembleResult,
    series_of: Callable[[AdmittedTrajectory], EnvelopeSeries],
) -> EnsembleEnvelope:
    """Envelope hull over the accepted samples of ``result``.

    ``series_of`` builds the per-trajectory :class:`EnvelopeSeries` (feasible
    sets and ``static_envelopes`` live with the caller).  Members are labelled
    ``sample-<index>`` after their draw index.
    """
    if not result.accepted:
        raise ValueError("ensemble has no accepted samples")
    members = [series_of(t) for t in result.accepted]
    return envelope_union_intersection(
        members,
        [f"sample-{i}" for i in result.sample_indices],
        hypothesis_ids=(result.hypothesis_id,),
    )
