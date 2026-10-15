"""Single entry point of the loop-nullspace analysis (NSX-40, #12572).

``analyze(trajectory, engine, preset, options)`` composes the existing NSX
modules, in the order the contract fixes, and returns the plain ``dict`` that
``schemas/loop-nullspace-report-v1.json`` describes.  There is **no new
mathematics** here: decomposition, admission, invariants, feasible sets, static
envelopes and the allocation are the functions of their own modules.

Pipeline (each stage can report progress and be cancelled between stages)::

    decomposition -> admission -> invariants -> feasible_sets -> envelopes
                  -> allocation (optional)

Honesty rules (R-series of the epic):

* A frame whose balance cannot be closed is **refused with its residual**, never
  repaired and never given a relaxed tolerance; the report lists it.
* Admission never repairs (``repair="none"``).  A rejected trajectory yields a
  report with the verdict and reasons and no downstream numbers.
* An infeasible frame appears in the envelope section with its minimal
  conflicting hypotheses; the allocation excludes it by name and says so.
* The allocation is a hypothesis-conditioned representative, never an estimate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from src.shared.python.loop_nullspace import registry
from src.shared.python.loop_nullspace.admission import (
    AdmissionGates,
    AdmittedTrajectory,
    VerdictKind,
    admit,
)
from src.shared.python.loop_nullspace.allocation import (
    allocate_trajectory,
    torque_scale_from_capacity,
)
from src.shared.python.loop_nullspace.allocation_spec import ObjectiveSpec
from src.shared.python.loop_nullspace.capacity import CapacityVector
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    InputProvenance,
    ModelProvenance,
)
from src.shared.python.loop_nullspace.decomposition import (
    AllocationMetric,
    BalanceInconsistentError,
    Decomposition,
    decompose,
)
from src.shared.python.loop_nullspace.envelopes import (
    BoundStatus,
    EnvelopeQuantity,
    joint_torque_quantities,
    static_envelopes,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.ground_contact import (
    GroundObservation,
    foot_friction_cones,
)
from src.shared.python.loop_nullspace.ground_envelopes import (
    closure_wrench_quantities,
    ground_force_quantities,
)
from src.shared.python.loop_nullspace.invariants import compute_invariants
from src.shared.python.loop_nullspace.report_serialization import (
    admission_dict,
    allocation_dict,
    envelope_dict,
    hypotheses_dict,
    invariants_dict,
    json_float,
)
from src.shared.python.motion_matching.same_input.closure import ClosurePlant

logger = logging.getLogger(__name__)

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "STAGES",
    "AnalysisCancelledError",
    "AnalysisOptions",
    "EngineContext",
    "RefusedFrame",
    "TrajectoryInput",
    "analyze",
]

REPORT_SCHEMA_VERSION = "1.0.0"
STAGES = (
    "decomposition",
    "admission",
    "invariants",
    "feasible_sets",
    "envelopes",
    "allocation",
)
_PROGRESS = {
    "decomposition": 10.0,
    "admission": 20.0,
    "invariants": 45.0,
    "feasible_sets": 60.0,
    "envelopes": 90.0,
    "allocation": 98.0,
}
NON_CLAIMS = (
    "Static per-frame feasible sets over-approximate trajectory-feasible ranges.",
    "FAMILY_DEPENDENT quantities are conditional on the declared hypotheses; "
    "torque capacities are priors, not measurements.",
    "An allocation is a hypothesis-conditioned representative of the admissible "
    "family, not an estimate of the true hand loads or joint torques.",
    "Refused, rejected and infeasible frames are reported as such and are never "
    "repaired or given a relaxed tolerance.",
    "Restriction sensitivity is not computed by this contract version.",
)

StageHook = Callable[[str, float, "Mapping[str, Any] | None"], None]
CancelCheck = Callable[[], bool]
ObservationProvider = Callable[[int, ClosedLoopSnapshot], "GroundObservation | None"]
ClubNewtonEuler = Callable[[ClosedLoopSnapshot], Mapping[str, Any]]


class AnalysisCancelledError(Exception):
    """Raised between stages once the caller's cancel check returns ``True``."""


@dataclass(frozen=True)
class RefusedFrame:
    """A frame that could not be turned into a snapshot or balanced."""

    index: int
    stage: str
    reason: str
    residual_scaled: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON form."""
        return {
            "index": int(self.index),
            "stage": self.stage,
            "reason": self.reason,
            "residual_scaled": None
            if self.residual_scaled is None
            else json_float(self.residual_scaled),
        }


@dataclass(frozen=True)
class TrajectoryInput:
    """Snapshots of the frames that could be built, plus those that could not.

    ``frame_indices`` and ``times_s`` run with ``snapshots``; ``refused`` holds
    frames refused while building (their indices are disjoint from
    ``frame_indices``).

    Raises:
        ValueError: On length mismatches or repeated indices.
    """

    snapshots: tuple[ClosedLoopSnapshot, ...]
    times_s: tuple[float, ...]
    frame_indices: tuple[int, ...]
    provenance: InputProvenance
    refused: tuple[RefusedFrame, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.snapshots)
        if len(self.times_s) != n or len(self.frame_indices) != n:
            raise ValueError(
                "snapshots, times_s and frame_indices must have one entry per frame"
            )
        names = [*self.frame_indices, *(r.index for r in self.refused)]
        if len(set(names)) != len(names):
            raise ValueError("frame indices must be unique across built and refused")
        for s in self.snapshots:
            if not isinstance(s, ClosedLoopSnapshot):
                raise TypeError("snapshots must be ClosedLoopSnapshot instances")


@dataclass(frozen=True)
class EngineContext:
    """The engine/model side of a run (capacity, plant and optional data feeds).

    ``ground_groups`` maps a report label to a contact-name suffix (a foot
    side); ``observation_for(index, snapshot)`` supplies ground-reaction data
    when the preset declares ``H-GRF-OBSERVATION``.
    """

    name: str
    model_provenance: ModelProvenance
    capacity: CapacityVector | None = None
    plant: ClosurePlant | None = None
    club_newton_euler: ClubNewtonEuler | None = None
    observation_for: ObservationProvider | None = None
    ground_groups: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("EngineContext requires a non-empty name")


@dataclass(frozen=True)
class AnalysisOptions:
    """What to compute; every field enters the result-cache key.

    Raises:
        ValueError: On an unknown metric or a non-finite / negative weight.
    """

    metric: Literal["identity", "inverse_mass"] = "identity"
    include_envelopes: bool = True
    include_allocation: bool = False
    effort_weight: float = 1.0
    rate_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.metric not in ("identity", "inverse_mass"):
            raise ValueError(f"unknown metric {self.metric!r}")
        for name in ("effort_weight", "rate_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        if self.include_allocation and not self.include_envelopes:
            raise ValueError("an allocation needs the envelope stage's feasible sets")

    def as_dict(self) -> dict[str, Any]:
        """JSON form (also the cache-key payload)."""
        return {
            "metric": self.metric,
            "include_envelopes": self.include_envelopes,
            "include_allocation": self.include_allocation,
            "effort_weight": float(self.effort_weight),
            "rate_weight": float(self.rate_weight),
        }


class _Run:
    """Progress and cancellation plumbing of one run."""

    def __init__(self, on_stage: StageHook | None, cancel: CancelCheck | None) -> None:
        self._on_stage = on_stage
        self._cancel = cancel

    def check(self) -> None:
        if self._cancel is not None and self._cancel():
            raise AnalysisCancelledError("the analysis was cancelled")

    def done(self, stage: str, partial: Mapping[str, Any] | None = None) -> None:
        if self._on_stage is not None:
            self._on_stage(stage, _PROGRESS[stage], partial)
        self.check()


def _metric(options: AnalysisOptions, snap: ClosedLoopSnapshot) -> AllocationMetric:
    if options.metric == "inverse_mass":
        return AllocationMetric.inverse_mass(snap)
    return AllocationMetric.identity(snap.topology.n_actuated)


def _decompose_all(
    trajectory: TrajectoryInput, options: AnalysisOptions
) -> tuple[list[int], list[Decomposition], list[RefusedFrame]]:
    """Decompose every built frame; a frame that cannot balance is refused."""
    kept: list[int] = []
    decs: list[Decomposition] = []
    refused = list(trajectory.refused)
    for pos, snap in enumerate(trajectory.snapshots):
        index = trajectory.frame_indices[pos]
        try:
            decs.append(decompose(snap, weights=_metric(options, snap)))
        except BalanceInconsistentError as err:
            refused.append(
                RefusedFrame(index, "decomposition", str(err), err.residual_scaled)
            )
            continue
        kept.append(pos)
    return kept, decs, refused


def _frame_set(
    dec: Decomposition,
    snap: ClosedLoopSnapshot,
    index: int,
    preset: registry.HypothesisSet,
    engine: EngineContext,
) -> FeasibleSet:
    declared = preset.hypothesis_ids()
    observation = (
        engine.observation_for(index, snap)
        if "H-GRF-OBSERVATION" in declared and engine.observation_for is not None
        else None
    )
    return FeasibleSet.from_hypotheses(
        dec,
        preset,
        capacity=engine.capacity if "H-TORQUE-CAPACITY" in declared else None,
        cones=foot_friction_cones(snap) if snap.foot_contacts else (),
        observation=observation,
    )


def _quantities(
    sets: Sequence[FeasibleSet],
    snaps: Sequence[ClosedLoopSnapshot],
    engine: EngineContext,
) -> Callable[[FeasibleSet], list[EnvelopeQuantity]]:
    """Joint torques, plus the closure wrench and ground loads when they exist."""
    torques = joint_torque_quantities(None)
    wrench = (
        closure_wrench_quantities()
        if snaps[0].topology.closure_kind.rows == 6
        else None
    )
    names = {
        id(fs): tuple(c.name for c in s.foot_contacts)
        for fs, s in zip(sets, snaps, strict=True)
    }
    ground = None
    if any(s.foot_contacts for s in snaps):
        groups = {
            label: (lambda n, sfx=suffix: n.endswith(sfx))
            for label, suffix in engine.ground_groups.items()
        }
        ground = ground_force_quantities(lambda fs: names[id(fs)], groups)

    def build(fs: FeasibleSet) -> list[EnvelopeQuantity]:
        out = list(torques(fs))
        for factory in (wrench, ground):
            if factory is not None:
                out += factory(fs)
        return out

    return build


def _allocation(
    sets: Sequence[FeasibleSet],
    times: Sequence[float],
    frame_indices: Sequence[int],
    feasible: Sequence[bool],
    engine: EngineContext,
    options: AnalysisOptions,
) -> dict[str, Any]:
    if engine.capacity is None:
        raise ValueError("the allocation needs the engine's torque capacity")
    keep = [k for k, ok in enumerate(feasible) if ok]
    excluded = [
        {"index": int(frame_indices[k]), "reason": "static feasible set is empty"}
        for k, ok in enumerate(feasible)
        if not ok
    ]
    if len(keep) < 1:
        return {
            "status": "not_run",
            "reason": "no frame has a non-empty feasible set",
            "frame_indices": [],
            "excluded_frames": excluded,
        }
    objective = ObjectiveSpec(
        times_s=np.asarray([times[k] for k in keep]),
        torque_scale=torque_scale_from_capacity(engine.capacity),
        effort_weight=options.effort_weight,
        rate_weight=options.rate_weight,
    )
    result = allocate_trajectory([sets[k] for k in keep], objective)
    return allocation_dict(result, [frame_indices[k] for k in keep], excluded)


def _envelope_section(
    sets: Sequence[FeasibleSet],
    snaps: Sequence[ClosedLoopSnapshot],
    frame_indices: Sequence[int],
    engine: EngineContext,
    source: InputProvenance,
) -> tuple[dict[str, Any], list[bool]]:
    series = static_envelopes(sets, _quantities(sets, snaps, engine))
    feasible = [st is not BoundStatus.INFEASIBLE for st in series.chebyshev_status]
    conflicts = {k: fs.conflict_set() for k, fs in enumerate(sets) if not feasible[k]}
    records = series.records(engine.model_provenance, source)
    return envelope_dict(series, records, frame_indices, conflicts), feasible


def _validate(trajectory: object, engine: object, options: object) -> None:
    if not isinstance(trajectory, TrajectoryInput):
        raise TypeError("trajectory must be a TrajectoryInput")
    if not isinstance(engine, EngineContext):
        raise TypeError("engine must be an EngineContext")
    if not isinstance(options, AnalysisOptions):
        raise TypeError("options must be an AnalysisOptions")


def analyze(
    trajectory: TrajectoryInput,
    engine: EngineContext,
    preset: str,
    options: AnalysisOptions | None = None,
    *,
    on_stage: StageHook | None = None,
    should_cancel: CancelCheck | None = None,
) -> dict[str, Any]:
    """Run the loop-nullspace analysis and return the report mapping.

    ``on_stage(stage, percent, partial)`` is called after each stage; the
    ``invariants`` stage passes ``{"invariants": ...}`` so a client can show
    the invariant quantities while the envelopes are still being solved.
    ``should_cancel()`` is checked between stages; ``AnalysisCancelledError``
    is raised once it returns ``True``.

    Preconditions (``TypeError``/``ValueError``): typed inputs; ``preset`` is a
    registry preset; a preset declaring ``H-TORQUE-CAPACITY`` needs
    ``engine.capacity``; one declaring ``H-GRF-OBSERVATION`` needs
    ``engine.observation_for``.

    Postconditions: the report validates against
    ``schemas/loop-nullspace-report-v1.json``; every quantity carries all six
    evidence axes; no frame is silently dropped (refused, failing and
    infeasible frames are listed); nothing is repaired.
    """
    options = options if options is not None else AnalysisOptions()
    _validate(trajectory, engine, options)
    reg = registry.load_registry()
    try:
        hyp_set = reg.preset(preset)
    except KeyError as exc:
        raise ValueError(f"unknown preset {preset!r}") from exc
    declared = hyp_set.hypothesis_ids()
    if "H-TORQUE-CAPACITY" in declared and engine.capacity is None:
        raise ValueError("the preset declares H-TORQUE-CAPACITY but no capacity is set")
    if "H-GRF-OBSERVATION" in declared and engine.observation_for is None:
        raise ValueError(
            "the preset declares H-GRF-OBSERVATION but the engine supplies no observation"
        )
    run = _Run(on_stage, should_cancel)
    run.check()

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "registry_version": reg.registry_version,
        "preset": preset,
        "engine": {
            "name": engine.name,
            "model_provenance": {
                "name": engine.model_provenance.name,
                "version": engine.model_provenance.version,
                "content_hash": engine.model_provenance.content_hash,
            },
        },
        "options": options.as_dict(),
        "hypotheses": hypotheses_dict(reg, hyp_set),
        "non_claims": list(NON_CLAIMS),
        "admission": None,
        "invariants": None,
        "envelopes": None,
        "allocation": None,
        "restriction_sensitivity": None,
    }
    kept, decs, refused = _decompose_all(trajectory, options)
    snaps = [trajectory.snapshots[p] for p in kept]
    indices = [trajectory.frame_indices[p] for p in kept]
    times = [trajectory.times_s[p] for p in kept]
    report["trajectory"] = {
        "input_provenance": {
            "kind": trajectory.provenance.kind,
            "source": trajectory.provenance.source,
            "note": trajectory.provenance.note,
        },
        "n_input_frames": len(trajectory.snapshots) + len(trajectory.refused),
        "frame_indices": sorted(
            [*trajectory.frame_indices, *(r.index for r in trajectory.refused)]
        ),
        "analyzed_frame_indices": indices,
        "refused_frames": [r.as_dict() for r in sorted(refused, key=lambda r: r.index)],
    }
    run.done("decomposition")
    if len(snaps) < 3:
        report["status"] = "insufficient_frames"
        report["reason"] = (
            f"{len(snaps)} frame(s) survived decomposition; admission needs at least 3"
        )
        return report

    gates = AdmissionGates(require_position=engine.plant is not None)
    adm_report = admit(
        snaps, gates=gates, times=times, plant=engine.plant, repair="none"
    )
    report["admission"] = admission_dict(adm_report, indices)
    run.done("admission")
    if adm_report.verdict.kind is VerdictKind.REJECTED:
        report["status"] = "admission_rejected"
        report["reason"] = "; ".join(adm_report.verdict.reasons)
        return report

    admitted = AdmittedTrajectory.from_report(adm_report)
    club = (
        None
        if engine.club_newton_euler is None
        else [engine.club_newton_euler(s) for s in snaps]
    )
    inv = invariants_dict(compute_invariants(admitted, decs, club_ne=club), indices)
    report["invariants"] = inv
    run.done("invariants", {"invariants": inv})
    if not options.include_envelopes:
        return report

    sets = [
        _frame_set(d, s, i, hyp_set, engine)
        for d, s, i in zip(decs, snaps, indices, strict=True)
    ]
    run.done("feasible_sets")
    report["envelopes"], feasible = _envelope_section(
        sets, snaps, indices, engine, trajectory.provenance
    )
    run.done("envelopes")
    if options.include_allocation:
        report["allocation"] = _allocation(
            sets, times, indices, feasible, engine, options
        )
        run.done("allocation")
    return report
