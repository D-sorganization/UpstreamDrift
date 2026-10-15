"""Support for the anthro lower-body benchmark and report (NSX-60, NSX-22).

Builds, for frames of the ground-supported anthro record, the two-foot
snapshot (rigid contacts where the compliant sphere gap is within the declared
tolerance), its decomposition, and the feasible set without and with
ground-reaction observations.  Three variants per frame:

* ``free``: no ground-reaction data (the free lower body);
* ``record``: the record's net vertical load and centre of pressure
  (``normal_force_n``, ``cop_m``), which come from the compliant simulation,
  not from a force plate;
* ``plate``: per-foot force sums of the compliant contact law at the
  projected state, a simulation standing in for two force plates.

Neither observation is a measurement.  Declared uncertainties are hypotheses
(``H-GRF-OBSERVATION``).
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace import registry
from src.shared.python.loop_nullspace.capacity import CapacitySubject, capacity_for
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    InputProvenance,
)
from src.shared.python.loop_nullspace.decomposition import (
    BalanceInconsistentError,
    Decomposition,
    decompose,
)
from src.shared.python.loop_nullspace.envelopes import (
    EnvelopeSeries,
    joint_torque_quantities,
    static_envelopes,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.ground_envelopes import (
    closure_wrench_quantities,
    ground_force_quantities,
)
from src.shared.python.loop_nullspace.ground_contact import (
    GroundObservation,
    ObservationAccounting,
    foot_friction_cones,
    observation_accounting,
)
from src.shared.python.loop_nullspace.snapshot import snapshot_from_adapter
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    EngineType,
    create_engine_force_adapter,
)
from src.shared.python.motion_matching.same_input import (
    VectorPlant,
    project_to_closure,
)

EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/full_body_models/evidence/ground_support"
)
SPEC = EVIDENCE / "full_body_spec_hipcal_scaled.json"
RECORD = EVIDENCE / "dynamics_record.npz"
ROOT_PREFIXES = ("Translation", "HipInput")

#: NSX-22 benchmark frames: 200 evenly strided frames of the 606-frame record.
BENCHMARK_FRAMES = tuple(range(0, 600, 3))
#: A sphere is a rigid contact when its gap to the ground is at most this
#: (the compliant law penetrates exactly when the gap is negative).
GAP_TOL_M = 0.0
#: Declared 1-sigma uncertainties of the stand-in plates (hypotheses).
SIGMA_FORCE_N = 5.0
SIGMA_COP_M = 0.005
#: Subject used for the (ASSUMED) capacity prior.
SUBJECT = CapacitySubject(80.0, 1.8)
SIDES = {"right_foot_fz": "_r", "left_foot_fz": "_l"}


@dataclass(frozen=True)
class Frame:
    """One frame: snapshot, decomposition and the named projected state."""

    index: int
    snapshot: ClosedLoopSnapshot
    decomposition: Decomposition
    coordinates: dict[str, float]
    rates: dict[str, float]
    contact_names: tuple[str, ...]


class AnthroBench:
    """Adapter, plant, record and capacity shared by every frame."""

    def __init__(self) -> None:
        self.adapter = create_engine_force_adapter(
            EngineType.PINOCCHIO, SPEC, allow_synthetic=False
        )
        self.plant = VectorPlant("pinocchio", SPEC.read_bytes())
        with np.load(RECORD) as rec:
            self.record = {k: rec[k] for k in rec.files}
        self.order = list(self.adapter.coordinate_order)
        actuated = [n for n in self.order if not n.startswith(ROOT_PREFIXES)]
        self.capacity = capacity_for(self.order, SUBJECT, actuated=actuated)
        self.actuated = actuated

    def frame(self, index: int) -> Frame:
        """Snapshot and decomposition of record frame ``index``."""
        rec = self.record
        proj = project_to_closure(self.plant, rec["q"][index], rec["v"][index])
        accel = self.plant.acceleration(proj.q, proj.v, rec["tau"][index])
        named = [self.plant._named(x) for x in (proj.q, proj.v, accel)]
        q, v, a = (np.array([d[n] for n in self.order]) for d in named)
        snap = snapshot_from_adapter(
            self.adapter,
            q,
            v,
            a,
            input_provenance=InputProvenance("anthro ground_support", f"frame {index}"),
            foot_contact_tol_m=GAP_TOL_M,
        )
        dec = decompose(snap, weights=np.eye(snap.topology.n_actuated))
        return Frame(
            index,
            snap,
            dec,
            named[0],
            named[1],
            tuple(c.name for c in snap.foot_contacts),
        )

    def record_observation(self, frame: Frame) -> GroundObservation | None:
        """Net vertical load and centre of pressure from the record.

        ``None`` for a flight frame and for a frame whose recorded normal load is
        zero (the centre of pressure is undefined without load).
        """
        i = frame.index
        if not frame.contact_names or float(self.record["normal_force_n"][i]) <= 0.0:
            return None
        return GroundObservation.from_normal_load_and_cop(
            frame.snapshot,
            normal_load_n=float(self.record["normal_force_n"][i]),
            cop_m=self.record["cop_m"][i],
            sigma_force_n=SIGMA_FORCE_N,
            sigma_cop_m=SIGMA_COP_M,
        )

    def plate_observation(self, frame: Frame) -> GroundObservation | None:
        """Per-foot force sums of the compliant contact law (simulation plates).

        ``None`` for a flight frame (no contact, nothing to observe).
        """
        if not frame.contact_names:
            return None
        samples = (
            self.adapter._plant.contact_forces(  # compliant law of the model itself
                frame.coordinates, frame.rates
            )
        )
        groups = {
            label: [n for n in frame.contact_names if n.endswith(suffix)]
            for label, suffix in SIDES.items()
        }
        groups = {k: v for k, v in groups.items() if v}
        forces = {
            label: sum(
                samples[n].normal_force_n + samples[n].friction_force_n for n in names
            )
            for label, names in groups.items()
        }
        return GroundObservation.from_contact_groups(
            frame.snapshot, groups, forces, sigma_n=SIGMA_FORCE_N
        )

    def feasible_set(
        self, frame: Frame, observation: GroundObservation | None
    ) -> FeasibleSet:
        """Capacity, friction-cone and (optionally) observation rows."""
        preset = "envelope_only" if observation is None else "envelope_grf_observed"
        return FeasibleSet.from_hypotheses(
            frame.decomposition,
            registry.preset(preset),
            capacity=self.capacity,
            cones=foot_friction_cones(frame.snapshot) if frame.contact_names else (),
            observation=observation,
        )


def accounting_table(
    bench: AnthroBench, frames: Sequence[Frame]
) -> dict[str, list[ObservationAccounting]]:
    """Observation accounting of both observation kinds on every frame."""
    out: dict[str, list[ObservationAccounting]] = {}
    for kind, make in (
        ("record", bench.record_observation),
        ("plate", bench.plate_observation),
    ):
        out[kind] = [
            observation_accounting(f.decomposition, o)
            for f in frames
            if (o := make(f)) is not None
        ]
    return out


def timed_envelopes(
    sets: Sequence[FeasibleSet],
    factory: Callable[[FeasibleSet], Any] | None = None,
) -> tuple[EnvelopeSeries, float]:
    """``static_envelopes`` over ``sets`` and its wall time in seconds."""
    start = time.perf_counter()
    series = static_envelopes(sets, factory or joint_torque_quantities(None))
    return series, time.perf_counter() - start


def report_quantities(
    frames_by_set: dict[int, Frame],
) -> Callable[[FeasibleSet], list[Any]]:
    """Joint torques, the closure wrench and the ground loads of a frame."""
    torques = joint_torque_quantities(None)
    wrench = closure_wrench_quantities()

    def build(fs: FeasibleSet) -> list[Any]:
        names = frames_by_set[id(fs)].contact_names
        ground = ground_force_quantities(
            lambda _fs: names,
            {
                label: (lambda n, s=suffix: n.endswith(s))
                for label, suffix in SIDES.items()
            },
        )
        return [*torques(fs), *wrench(fs), *ground(fs)]

    return build


def build_frames(
    bench: AnthroBench, indices: Sequence[int]
) -> tuple[list[Frame], dict[int, float]]:
    """Frames that decompose, and the scaled balance residual of those refused.

    A frame is refused (never repaired, ``BALANCE_TOL`` unchanged) when its
    unactuated rows cannot be balanced; in the benchmark that is the airborne
    frame, which has no contact and a solver-noise residual in ``a``.
    """
    frames, refused = [], {}
    for i in indices:
        try:
            frames.append(bench.frame(i))
        except BalanceInconsistentError as err:
            refused[i] = float(err.residual_scaled)
    return frames, refused


def _stats(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def variant_report(
    bench: AnthroBench, frames: Sequence[Frame], kind: str
) -> tuple[dict[str, Any], EnvelopeSeries]:
    """Envelope series and its summary for ``kind`` in free, record, plate."""
    make = {
        "free": lambda f: None,
        "record": bench.record_observation,
        "plate": bench.plate_observation,
    }[kind]
    observations = [make(f) for f in frames]
    sets = [bench.feasible_set(f, o) for f, o in zip(frames, observations, strict=True)]
    lookup = {id(fs): f for fs, f in zip(sets, frames, strict=True)}
    series, wall = timed_envelopes(sets, report_quantities(lookup))
    status = [st.value for st in series.chebyshev_status]
    infeasible = [
        i for i, st in enumerate(series.chebyshev_status) if st.name == "INFEASIBLE"
    ]
    conflicts: dict[str, int] = {}
    for i in infeasible:
        for h in sets[i].conflict_set():
            conflicts[h] = conflicts.get(h, 0) + 1
    width = series.width()
    summary = {
        "frames": len(frames),
        "wall_s": wall,
        "status_counts": {s: status.count(s) for s in sorted(set(status))},
        "infeasible_frames": [frames[i].index for i in infeasible],
        "conflict_hypotheses": conflicts,
        "chebyshev_margin": _stats(series.chebyshev_margin.tolist()),
        "median_width": {
            n: float(np.nanmedian(width[:, j]))
            if np.isfinite(width[:, j]).any()
            else None
            for j, n in enumerate(series.names)
        },
        "simplex_iterations": series.simplex_iterations,
    }
    return summary, series


def host_record() -> dict[str, str]:
    """Where the numbers were measured."""
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def accounting_summary(accounts: Sequence[ObservationAccounting]) -> dict[str, Any]:
    """Distribution of the rank removed, singular values and data consistency."""
    ranks = [a.rank_visible for a in accounts]
    smallest_visible = [
        float(a.singular_values_over_sigma[a.rank_visible - 1])
        for a in accounts
        if a.rank_visible
    ]
    return {
        "frames_observed": len(accounts),
        "rank_visible_counts": {str(r): ranks.count(r) for r in sorted(set(ranks))},
        "nullity_free": _stats([a.nullity_free for a in accounts]),
        "nullity_observed": _stats([a.nullity_observed for a in accounts]),
        "weakest_visible_singular_value_over_sigma": _stats(smallest_visible),
        "max_normalised_residual": _stats(
            [a.max_normalised_residual for a in accounts]
        ),
    }


def compare_series(free: EnvelopeSeries, observed: EnvelopeSeries) -> dict[str, Any]:
    """Which conclusions change when the lower body is observed.

    Per quantity: the median width ratio observed/free on frames bounded in
    both, and the number of frames whose sign class differs.  A conclusion that
    changes is a sign class that was INDETERMINATE and became decided (or a
    bound that tightened); the rows of the observed set contain the free set's
    rows, so a bound can only narrow.
    """
    out: dict[str, Any] = {}
    free_w, obs_w = free.width(), observed.width()
    for j, name in enumerate(free.names):
        both = np.isfinite(free_w[:, j]) & np.isfinite(obs_w[:, j])
        ratio = obs_w[both, j] / np.maximum(free_w[both, j], 1e-12)
        changed = [
            (_sign(free.sign_class[i], j), _sign(observed.sign_class[i], j))
            for i in range(free.n_frames)
            if _sign(free.sign_class[i], j) != _sign(observed.sign_class[i], j)
        ]
        out[name] = {
            "frames_compared": int(both.sum()),
            "median_width_ratio": float(np.median(ratio)) if ratio.size else None,
            "min_width_ratio": float(ratio.min()) if ratio.size else None,
            "sign_class_changes": {
                f"{a}->{b}": changed.count((a, b)) for a, b in sorted(set(changed))
            },
            "widened_frames": int(np.sum(ratio > 1.0 + 1e-6)),
        }
    return out


def bounded_frames(
    frames: Sequence[Frame], free_summary: dict[str, Any]
) -> list[Frame]:
    """Frames whose free feasible set is non-empty.

    An observation only adds rows to a set, so a frame that is infeasible
    without it stays infeasible with it.  The observed variants therefore run on
    the non-empty frames and inherit the others' ``INFEASIBLE`` status; the
    solver reports an empty set of rows of mixed scale as ``NUMERICAL`` on
    some frames (frame 342 with the plate rows), which is an inconclusive
    status and is not recoded as infeasible by the library.
    """
    empty = set(free_summary["infeasible_frames"])
    return [f for f in frames if f.index not in empty]


def _sign(row: Any, j: int) -> str:
    """Sign-class name of quantity ``j`` ("NONE" when the frame has none)."""
    return "NONE" if not row or row[j] is None else str(row[j].name)
