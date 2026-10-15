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
from src.shared.python.loop_nullspace.decomposition import Decomposition, decompose
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

    def record_observation(self, frame: Frame) -> GroundObservation:
        """Net vertical load and centre of pressure from the record."""
        i = frame.index
        return GroundObservation.from_normal_load_and_cop(
            frame.snapshot,
            normal_load_n=float(self.record["normal_force_n"][i]),
            cop_m=self.record["cop_m"][i],
            sigma_force_n=SIGMA_FORCE_N,
            sigma_cop_m=SIGMA_COP_M,
        )

    def plate_observation(self, frame: Frame) -> GroundObservation:
        """Per-foot force sums of the compliant contact law (simulation plates)."""
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
        preset = "envelope_grf_observed" if observation else "envelope_only"
        return FeasibleSet.from_hypotheses(
            frame.decomposition,
            registry.preset(preset),
            capacity=self.capacity,
            cones=foot_friction_cones(frame.snapshot),
            observation=observation,
        )


def accounting_table(
    bench: AnthroBench, frames: Sequence[Frame]
) -> dict[str, list[ObservationAccounting]]:
    """Observation accounting of both observation kinds on every frame."""
    return {
        "record": [
            observation_accounting(f.decomposition, bench.record_observation(f))
            for f in frames
        ],
        "plate": [
            observation_accounting(f.decomposition, bench.plate_observation(f))
            for f in frames
        ],
    }


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
            {label: (lambda n, s=suffix: n.endswith(s)) for label, suffix in SIDES.items()},
        )
        return [*torques(fs), *wrench(fs), *ground(fs)]

    return build
