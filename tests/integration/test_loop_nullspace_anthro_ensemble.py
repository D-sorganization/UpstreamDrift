"""NSX-61 (#12586): kinematic ensemble on the anthro fixture.

Sampling and admission are kinematic and run on the 41-coordinate Pinocchio
anthro model.  The envelope half uses the NSX-60 (#12585) foot-contact rows:
each ensemble member is rebuilt with ``foot_contact_tol_m`` (rigid contacts where
the compliant sphere gap is at most zero, the benchmark's declared tolerance),
decomposed, given the ``H-TORQUE-CAPACITY`` prior (ASSUMED, unchanged) and Coulomb
cones, and enveloped with ``static_envelopes``.  The hypothesis-declared
perturbation moves the root as well as the joints, so a sample can lift a foot
or leave fewer than three contacts: its root rows then cannot be balanced
(``a`` is held at the nominal value) and the member is counted as refused, never
repaired.  The report is a modest strided sample (three windows of three
consecutive record frames) because the host is shared.

Set ``LOOP_NULLSPACE_ENSEMBLE_JSON`` to a file path to write the measured record.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from src.shared.python.loop_nullspace.admission import (  # noqa: E402
    AdmissionGates,
    AdmittedTrajectory,
    admit,
)
from src.shared.python.loop_nullspace.contracts import InputProvenance  # noqa: E402
from src.shared.python.loop_nullspace.decomposition import (  # noqa: E402
    BalanceInconsistentError,
    decompose,
)
from src.shared.python.loop_nullspace.ensemble import (  # noqa: E402
    ENSEMBLE_HYPOTHESIS_ID,
    EnsembleSource,
    SamplingHypothesis,
    envelope_union_intersection,
    kinematic_ensemble,
)
from src.shared.python.loop_nullspace.envelopes import (  # noqa: E402
    BoundStatus,
    joint_torque_quantities,
    static_envelopes,
)
from src.shared.python.loop_nullspace.ground_envelopes import (  # noqa: E402
    closure_wrench_quantities,
)
from src.shared.python.loop_nullspace.snapshot import snapshot_from_adapter  # noqa: E402
from src.shared.python.motion_matching.multi_engine_torque_allocator import (  # noqa: E402
    EngineType,
    create_engine_force_adapter,
)
from src.shared.python.motion_matching.same_input import (  # noqa: E402
    VectorPlant,
    project_to_closure,
)

from tests.integration.loop_nullspace_lower_body_support import (  # noqa: E402
    GAP_TOL_M,
    AnthroBench,
    Frame,
    host_record,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_pinocchio]

EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "docs/development/full_body_models/evidence/ground_support"
)
SPEC = EVIDENCE / "full_body_spec_hipcal_scaled.json"
FRAMES = (0, 1, 2)
N_SAMPLES = 8
SEED = 12586
SIGMA_Q = 0.02
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def adapter():
    return create_engine_force_adapter(
        EngineType.PINOCCHIO, SPEC, allow_synthetic=False
    )


class _AdapterOrderPlant:
    """``VectorPlant`` (spec order) seen in the adapter's coordinate order."""

    def __init__(self, plant: VectorPlant, order: tuple[str, ...]) -> None:
        self.plant = plant
        self.order = order
        self.perm = [list(plant.coordinate_order).index(n) for n in order]

    def _to_plant(self, x):
        return self.plant._ordered(dict(zip(self.order, x, strict=True)))

    def closure_pose_residual(self, q):
        return self.plant.closure_pose_residual(self._to_plant(q))

    def closure_rate_matrix(self, q):
        return self.plant.closure_rate_matrix(self._to_plant(q))[:, self.perm]


@pytest.fixture(scope="module")
def vector_plant():
    return VectorPlant("pinocchio", SPEC.read_bytes())


@pytest.fixture(scope="module")
def plant(adapter, vector_plant):
    return _AdapterOrderPlant(vector_plant, tuple(adapter.coordinate_order))


def _rebuild(adapter):
    def rebuild(q, v, a):
        return snapshot_from_adapter(
            adapter, q, v, a, input_provenance=InputProvenance("test", "anthro")
        )

    return rebuild


@pytest.fixture(scope="module")
def nominal(adapter, plant, vector_plant):
    with np.load(EVIDENCE / "dynamics_record.npz") as rec:
        times = rec["time_s"][list(FRAMES)]
        states = [(rec["q"][k], rec["v"][k], rec["tau"][k]) for k in FRAMES]
    rebuild = _rebuild(adapter)
    snaps = []
    for q0, v0, tau in states:
        proj = project_to_closure(vector_plant, q0, v0)
        order = list(adapter.coordinate_order)
        q, v, a = (
            np.array([d[n] for n in order])
            for d in (
                vector_plant._named(proj.q),
                vector_plant._named(proj.v),
                vector_plant._named(vector_plant.acceleration(proj.q, proj.v, tau)),
            )
        )
        snaps.append(rebuild(q, v, a))
    report = admit(
        snaps,
        gates=AdmissionGates(),
        times=times,
        plant=plant,
        repair="project_to_closure",
        rebuild=rebuild,
    )
    return AdmittedTrajectory.from_report(report)


@pytest.fixture(scope="module")
def result(adapter, plant, nominal):
    source = EnsembleSource(
        nominal=nominal,
        plant=plant,
        rebuild=_rebuild(adapter),
        sampling=SamplingHypothesis(sigma_q=SIGMA_Q),
    )
    return kinematic_ensemble(source, N_SAMPLES, SEED)


def test_anthro_ensemble_admitted_and_rejected_counts(result) -> None:
    logger.info(
        "anthro ensemble n=%d seed=%d sigma_q=%g: accepted=%d (repaired=%d) "
        "rejected=%d by %s",
        result.n_requested,
        result.seed,
        result.sigma_q,
        result.n_accepted,
        result.n_repaired,
        result.n_rejected,
        result.rejected_by_reason,
    )
    assert result.n_accepted + result.n_rejected == N_SAMPLES
    assert result.hypothesis_id == ENSEMBLE_HYPOTHESIS_ID
    assert result.n_accepted >= 1


def test_anthro_samples_differ_from_nominal_and_are_deterministic(
    result, nominal, adapter, plant
) -> None:
    q0 = nominal.snapshots[0].q
    q1 = result.accepted[0].snapshots[0].q
    assert 0.0 < np.max(np.abs(q1 - q0)) < 1.0
    assert np.max(np.abs(plant.closure_pose_residual(q1))) < 1e-6


# ----------------------------------------------------- envelope half (NSX-60) --
#: Three windows of three consecutive record frames, spread over the record.
WINDOW_STARTS = (0, 150, 300)
WINDOW_LENGTH = 3
#: Envelope sample: 1 mm / 1 mrad keeps about half the samples in ground contact
#: (the 0.02 of the kinematic checks above lifts or sinks the feet, so almost
#: every sample is refused; both are reported).
ENVELOPE_SIGMA_Q = 0.001
ENVELOPE_SAMPLES = 12


@pytest.fixture(scope="module")
def bench() -> AnthroBench:
    return AnthroBench()


def _contact_rebuild(bench):
    def rebuild(q, v, a):
        return snapshot_from_adapter(
            bench.adapter,
            q,
            v,
            a,
            input_provenance=InputProvenance("test", "anthro ensemble"),
            foot_contact_tol_m=GAP_TOL_M,
        )

    return rebuild


def _member_sets(bench, snapshots):
    """Feasible sets of a trajectory, or ``None`` when a frame cannot be balanced."""
    sets = []
    for snap in snapshots:
        try:
            dec = decompose(snap, weights=np.eye(snap.topology.n_actuated))
        except BalanceInconsistentError:
            return None
        names = tuple(c.name for c in snap.foot_contacts)
        frame = Frame(-1, snap, dec, {}, {}, names)
        sets.append(bench.feasible_set(frame, None))
    return sets


def _quantities():
    torques, wrench = joint_torque_quantities(None), closure_wrench_quantities()
    return lambda fs: [*torques(fs), *wrench(fs)]


def _class_names(series, t, j):
    cls = series.sign_class[t]
    return "NONE" if not cls or cls[j] is None else cls[j].name


def _sign_report(nominal, members, union):
    """Cells where the feasible members' sign classes differ, and the union's class.

    An infeasible member frame has no sign class; it is counted separately
    (``member_frames_infeasible_under_capacity``), not as a class change.
    """
    n_frames, n_q = nominal.lower.shape
    member_changes = np.zeros((n_frames, n_q), dtype=bool)
    lost = np.zeros((n_frames, n_q), dtype=bool)
    for t in range(n_frames):
        for j in range(n_q):
            classes = {_class_names(m, t, j) for m in members} - {"NONE"}
            member_changes[t, j] = len(classes) > 1
            decided = _class_names(nominal, t, j) in ("POSITIVE", "NEGATIVE")
            lo, hi = union.union_lower[t, j], union.union_upper[t, j]
            union_decided = bool(lo > 0.0 or hi < 0.0)
            lost[t, j] = decided and not union_decided
    return {
        "cells": int(member_changes.size),
        "cells_with_member_class_disagreement": int(member_changes.sum()),
        "frames_with_member_class_disagreement": int(member_changes.any(axis=1).sum()),
        "decided_nominal_cells_indeterminate_in_union": int(lost.sum()),
        "frames_losing_a_decided_sign_in_union": int(lost.any(axis=1).sum()),
    }


def _width_ratio(num_lo, num_hi, nominal):
    width = nominal.upper - nominal.lower
    ok = np.isfinite(num_hi - num_lo) & np.isfinite(width) & (width > 1e-9)
    ratio = (num_hi - num_lo)[ok] / width[ok]
    return {
        "cells": int(ok.sum()),
        "median": float(np.median(ratio)) if ratio.size else None,
        "max": float(ratio.max()) if ratio.size else None,
        "min": float(ratio.min()) if ratio.size else None,
    }


def _window_report(bench, start, plant, rebuild, sigma, n_samples):
    indices = tuple(range(start, start + WINDOW_LENGTH))
    frames = [bench.frame(i) for i in indices]
    times = bench.record["time_s"][list(indices)]
    nominal_report = admit(
        [f.snapshot for f in frames],
        gates=AdmissionGates(),
        times=times,
        plant=plant,
        repair="project_to_closure",
        rebuild=rebuild,
    )
    nominal = AdmittedTrajectory.from_report(nominal_report)
    result = kinematic_ensemble(
        EnsembleSource(
            nominal=nominal,
            plant=plant,
            rebuild=rebuild,
            sampling=SamplingHypothesis(sigma_q=sigma),
        ),
        n_samples,
        SEED,
    )
    factory = _quantities()
    nominal_series = static_envelopes(
        [bench.feasible_set(f, None) for f in frames], factory
    )
    series, labels, unbalanced = [nominal_series], ["nominal"], 0
    for traj, k in zip(result.accepted, result.sample_indices, strict=True):
        sets = _member_sets(bench, traj.snapshots)
        if sets is None:
            unbalanced += 1
            continue
        series.append(static_envelopes(sets, factory))
        labels.append(f"sample-{k}")
    union = envelope_union_intersection(series, labels)
    infeasible_member_frames = sum(
        st is BoundStatus.INFEASIBLE for m in series[1:] for st in m.chebyshev_status
    )
    empty = union.intersection_empty
    report = {
        "record_frames": list(indices),
        "nominal_verdict": nominal_report.verdict.kind.name,
        "nominal_infeasible_frames": int(
            sum(st is BoundStatus.INFEASIBLE for st in nominal_series.chebyshev_status)
        ),
        "requested": result.n_requested,
        "admitted": result.n_accepted,
        "rejected_by_reason": result.rejected_by_reason,
        "unbalanced_refused": unbalanced,
        "members_enveloped": len(series) - 1,
        "member_frames_infeasible_under_capacity": int(infeasible_member_frames),
        "frames_with_an_infeasible_member": sum(
            any(m.chebyshev_status[t] is BoundStatus.INFEASIBLE for m in series[1:])
            for t in range(WINDOW_LENGTH)
        ),
        "intersection_empty_cells": int(empty.sum()),
        "union_width_over_nominal": _width_ratio(
            union.union_lower, union.union_upper, nominal_series
        ),
        "intersection_width_over_nominal": _width_ratio(
            union.intersection_lower, union.intersection_upper, nominal_series
        ),
        "sign": _sign_report(nominal_series, series, union),
    }
    return report, union, series


@pytest.mark.slow
def test_anthro_ensemble_envelope_union_and_intersection_report(bench, plant) -> None:
    pytest.importorskip("highspy")
    rebuild = _contact_rebuild(bench)
    plant_in_order = plant
    windows, coarse = [], {}
    for start in WINDOW_STARTS:
        report, union, series = _window_report(
            bench, start, plant_in_order, rebuild, ENVELOPE_SIGMA_Q, ENVELOPE_SAMPLES
        )
        windows.append(report)
        # The union holds every member interval; the intersection lies in each.
        for m in series[1:]:
            both = np.isfinite(m.lower) & np.isfinite(union.union_lower)
            assert np.all(union.union_lower[both] <= m.lower[both] + 1e-9)
            both = np.isfinite(m.upper) & np.isfinite(union.union_upper)
            assert np.all(union.union_upper[both] >= m.upper[both] - 1e-9)
        assert report["admitted"] + sum(report["rejected_by_reason"].values()) == (
            ENVELOPE_SAMPLES
        )
        assert (
            report["members_enveloped"] + report["unbalanced_refused"]
            == (report["admitted"])
        )
    assert sum(w["members_enveloped"] for w in windows) >= len(WINDOW_STARTS)
    # The union is never narrower than the nominal envelope it contains.
    for w in windows:
        med = w["union_width_over_nominal"]["median"]
        assert med is None or med >= 1.0 - 1e-6

    coarse["kinematic_default_sigma"] = _coarse_refusal(bench, rebuild, plant)
    target = os.environ.get("LOOP_NULLSPACE_ENSEMBLE_JSON")
    logger.info("anthro ensemble envelope report: %s", windows)
    if target:
        record = {
            "host": host_record(),
            "seed": SEED,
            "sigma_q": ENVELOPE_SIGMA_Q,
            "windows": windows,
            **coarse,
        }
        Path(target).write_text(json.dumps(record, indent=2, default=str))


def _coarse_refusal(bench, rebuild, plant):
    """Share of ensemble members that can be balanced at the larger sigma."""
    out = {}
    start = WINDOW_STARTS[0]
    indices = tuple(range(start, start + WINDOW_LENGTH))
    frames = [bench.frame(i) for i in indices]
    nominal = AdmittedTrajectory.from_report(
        admit(
            [f.snapshot for f in frames],
            gates=AdmissionGates(),
            times=bench.record["time_s"][list(indices)],
            plant=plant,
            repair="project_to_closure",
            rebuild=rebuild,
        )
    )
    result = kinematic_ensemble(
        EnsembleSource(
            nominal=nominal,
            plant=plant,
            rebuild=rebuild,
            sampling=SamplingHypothesis(sigma_q=SIGMA_Q),
        ),
        ENVELOPE_SAMPLES,
        SEED,
    )
    balanced = sum(
        _member_sets(bench, t.snapshots) is not None for t in result.accepted
    )
    out.update(
        sigma_q=SIGMA_Q,
        requested=ENVELOPE_SAMPLES,
        admitted=result.n_accepted,
        balanced=balanced,
    )
    return out
