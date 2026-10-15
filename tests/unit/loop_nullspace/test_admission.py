"""NSX-14: kinematic and contact admission gate (#12559)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.shared.python.loop_nullspace.admission import (
    AdmissionGates,
    AdmittedTrajectory,
    VerdictKind,
    admit,
)
from src.shared.python.loop_nullspace.fixtures import (
    generic_planar_config,
    load_golden,
    planar_golfer_snapshot,
)
from src.shared.python.loop_nullspace.planar_model import (
    PlanarGolferParams,
    planar_dynamics,
)
from src.shared.python.loop_nullspace.spatial_fixture import build_case

pytestmark = pytest.mark.unit

TIMES = np.array([0.0, 0.01, 0.02, 0.03])


class _PlanarPlant:
    """``ClosurePlant`` over the independent numpy planar model."""

    def __init__(self) -> None:
        self.params = PlanarGolferParams.from_golden(load_golden("planar_two_arm_full"))

    def closure_pose_residual(self, q):
        return planar_dynamics(q, np.zeros(5), self.params)["phi"]

    def closure_rate_matrix(self, q):
        return planar_dynamics(q, np.zeros(5), self.params)["Jc"]

    def rebuild(self, q, v, a):
        dyn = planar_dynamics(q, v, self.params)
        base = planar_golfer_snapshot().snapshot
        return dataclasses.replace(
            base,
            q=q,
            v=v,
            a=a,
            M=dyn["M"],
            h=dyn["h"],
            J_c=dyn["Jc"],
            Jdot_v=dyn["Jdot_v"],
        )


@pytest.fixture(scope="module")
def plant() -> _PlanarPlant:
    return _PlanarPlant()


@pytest.fixture(scope="module")
def oracle():
    return planar_golfer_snapshot(generic_planar_config()).snapshot


def _traj(snap, n=4):
    return [snap] * n


def test_admitted_on_oracle(oracle, plant):
    report = admit(_traj(oracle), gates=AdmissionGates(), times=TIMES, plant=plant)
    assert report.verdict.kind is VerdictKind.ADMITTED
    assert report.verdict.failing_frames == ()
    assert len(report.frames) == 4
    assert all(
        f.phi < 1e-6 and f.phi_dot < 1e-6 and f.phi_ddot < 1e-6 for f in report.frames
    )
    assert report.derivative_method == "analytic"
    adm = AdmittedTrajectory.from_report(report)
    assert len(adm.snapshots) == 4


def test_position_violation_names_frame(oracle, plant):
    bad = dataclasses.replace(oracle, q=oracle.q + np.array([1e-3, 0, 0, 0, 0]))
    traj = [oracle, bad, oracle, oracle]
    report = admit(traj, gates=AdmissionGates(), times=TIMES, plant=plant)
    assert report.verdict.kind is VerdictKind.REJECTED
    assert report.verdict.failing_frames == (1,)
    assert any("frame 1" in r and "position" in r for r in report.verdict.reasons)
    with pytest.raises(ValueError, match="REJECTED"):
        AdmittedTrajectory.from_report(report)


def test_velocity_violation(oracle, plant):
    v_bad = oracle.v + np.linalg.pinv(oracle.J_c) @ np.array([1e-3, 0.0])
    traj = [oracle, oracle, dataclasses.replace(oracle, v=v_bad), oracle]
    report = admit(traj, gates=AdmissionGates(), times=TIMES, plant=plant)
    assert report.verdict.kind is VerdictKind.REJECTED
    assert report.verdict.failing_frames == (2,)
    assert any("velocity" in r for r in report.verdict.reasons)
    assert report.frames[2].phi_dot > 1e-6


def test_acceleration_violation(oracle, plant):
    a_bad = oracle.a + np.linalg.pinv(oracle.J_c) @ np.array([0.0, 1e-3])
    traj = [dataclasses.replace(oracle, a=a_bad), oracle, oracle, oracle]
    report = admit(traj, gates=AdmissionGates(), times=TIMES, plant=plant)
    assert report.verdict.failing_frames == (0,)
    assert any("acceleration" in r for r in report.verdict.reasons)


def test_scaling_uses_characteristic_scales(oracle, plant):
    v_bad = oracle.v + np.linalg.pinv(oracle.J_c) @ np.array([1e-3, 0.0])
    traj = [dataclasses.replace(oracle, v=v_bad)] * 4
    loose = AdmissionGates(time_scale_s=1e-3)  # velocity scale T/L shrinks 1e3
    report = admit(traj, gates=loose, times=TIMES, plant=plant)
    base = admit(traj, gates=AdmissionGates(), times=TIMES, plant=plant)
    assert report.frames[0].phi_dot == pytest.approx(
        base.frames[0].phi_dot * 1e-3, rel=1e-9
    )


def test_missing_plant_fails_closed(oracle):
    report = admit(_traj(oracle), gates=AdmissionGates(), times=TIMES)
    assert report.verdict.kind is VerdictKind.REJECTED
    assert any("position" in r and "plant" in r for r in report.verdict.reasons)
    assert np.isnan(report.frames[0].phi)
    rec = report.residual_record("phi")
    assert rec.reason


def test_position_can_be_explicitly_skipped(oracle):
    gates = AdmissionGates(require_position=False)
    report = admit(_traj(oracle), gates=gates, times=TIMES)
    assert report.verdict.kind is VerdictKind.ADMITTED
    assert report.position_evaluated is False


def test_preconditions(oracle, plant):
    with pytest.raises(ValueError, match="at least 3"):
        admit([oracle, oracle], gates=AdmissionGates(), times=[0.0, 1.0], plant=plant)
    with pytest.raises(ValueError, match="strictly increasing"):
        admit(
            _traj(oracle, 3), gates=AdmissionGates(), times=[0.0, 0.1, 0.1], plant=plant
        )
    with pytest.raises(ValueError, match="one time per frame"):
        admit(_traj(oracle, 3), gates=AdmissionGates(), times=[0.0, 0.1], plant=plant)
    with pytest.raises(ValueError, match="repair"):
        admit(
            _traj(oracle, 3),
            gates=AdmissionGates(),
            times=TIMES[:3],
            plant=plant,
            repair="bogus",
        )
    with pytest.raises(ValueError, match="plant and rebuild"):
        admit(
            _traj(oracle, 3),
            gates=AdmissionGates(),
            times=TIMES[:3],
            plant=plant,
            repair="project_to_closure",
        )
    with pytest.raises(ValueError):
        AdmissionGates(phi_tol=0.0)


def test_time_jitter_gate(oracle, plant):
    gates = AdmissionGates(max_dt_jitter=0.01)
    ragged = [0.0, 0.01, 0.025, 0.03]
    report = admit(_traj(oracle), gates=gates, times=ragged, plant=plant)
    assert report.verdict.kind is VerdictKind.REJECTED
    assert any("time" in r for r in report.verdict.reasons)
    assert report.time_sync.max_dt_jitter > 0.01
    ok = admit(_traj(oracle), gates=gates, times=TIMES, plant=plant)
    assert ok.time_sync.ok


def test_central_difference_derivative_consistency(oracle, plant):
    q0, v0, a0 = oracle.q, oracle.v, oracle.a
    frames = [
        dataclasses.replace(oracle, q=q0 + v0 * t + 0.5 * a0 * t * t, v=v0 + a0 * t)
        for t in TIMES
    ]
    gates = AdmissionGates(
        derivative_method="central_difference", require_position=False
    )
    report = admit(frames, gates=gates, times=TIMES)
    assert report.derivative_method == "central_difference"
    assert report.noise_floor < 1e-9
    frames[2] = dataclasses.replace(frames[2], v=frames[2].v + 0.5)
    bad = admit(frames, gates=gates, times=TIMES)
    assert bad.noise_floor > 1e-3
    assert any("derivative" in r for r in bad.verdict.reasons)


def test_contact_schedule_stance_velocity():
    case = build_case(floating_root=True, feet=True)
    snap = case.snapshot
    # remove stance foot velocity: v in the null space of [J_c; J_g]
    stacked = np.vstack([snap.J_c, snap.J_g])
    v_ok = snap.v - np.linalg.pinv(stacked) @ (stacked @ snap.v)
    good = dataclasses.replace(snap, v=v_ok)
    sched = [frozenset({"foot_0", "foot_1"})] * 3
    gates = AdmissionGates(require_position=False, contact_schedule=tuple(sched))
    report = admit([good, good, good], gates=gates, times=[0.0, 0.1, 0.2])
    assert report.contact.evaluated
    assert report.verdict.kind is VerdictKind.ADMITTED
    moving = [good, snap, good]
    bad = admit(moving, gates=gates, times=[0.0, 0.1, 0.2])
    assert bad.verdict.kind is VerdictKind.REJECTED
    assert bad.contact.failing_frames == (1,)
    assert any("frame 1" in r and "stance" in r for r in bad.verdict.reasons)


def test_contact_schedule_unknown_name_and_length():
    case = build_case(floating_root=True, feet=True)
    snap = case.snapshot
    gates = AdmissionGates(
        require_position=False, contact_schedule=(frozenset({"nope"}),) * 3
    )
    report = admit([snap] * 3, gates=gates, times=[0.0, 0.1, 0.2])
    assert report.verdict.kind is VerdictKind.REJECTED
    assert any("unknown contact" in r for r in report.verdict.reasons)
    short = AdmissionGates(require_position=False, contact_schedule=(frozenset(),))
    with pytest.raises(ValueError, match="contact_schedule"):
        admit([snap] * 3, gates=short, times=[0.0, 0.1, 0.2])


def test_repair_is_explicit_recorded_and_reversible(oracle, plant):
    bad = dataclasses.replace(
        oracle,
        q=oracle.q + np.array([2e-3, -1e-3, 0, 0, 0]),
        v=oracle.v + np.linalg.pinv(oracle.J_c) @ np.array([1e-3, 0.0]),
    )
    traj = [oracle, bad, oracle, oracle]
    rejected = admit(traj, gates=AdmissionGates(), times=TIMES, plant=plant)
    assert rejected.verdict.kind is VerdictKind.REJECTED
    fixed = admit(
        traj,
        gates=AdmissionGates(),
        times=TIMES,
        plant=plant,
        repair="project_to_closure",
        rebuild=plant.rebuild,
    )
    assert fixed.verdict.kind is VerdictKind.REPAIRED
    record = fixed.verdict.repair
    assert [c.frame for c in record.corrections] == [1]
    c = record.corrections[0]
    assert c.dq_norm > 1e-4 and c.dv_norm > 1e-4
    np.testing.assert_array_equal(c.q_before, bad.q)
    np.testing.assert_array_equal(c.v_before, bad.v)
    np.testing.assert_allclose(c.q_after - c.q_before, fixed.snapshots[1].q - bad.q)
    assert fixed.snapshots[0] is oracle
    assert all(
        f.phi < 1e-6 and f.phi_dot < 1e-6 and f.phi_ddot < 1e-6 for f in fixed.frames
    )
    assert AdmittedTrajectory.from_report(fixed).report.verdict.repair is record


def test_repair_not_applied_to_unrepairable_failures(oracle, plant):
    a_bad = oracle.a + np.linalg.pinv(oracle.J_c) @ np.array([0.0, 1e-3])
    traj = [dataclasses.replace(oracle, a=a_bad), oracle, oracle, oracle]
    report = admit(
        traj,
        gates=AdmissionGates(),
        times=TIMES,
        plant=plant,
        repair="project_to_closure",
        rebuild=plant.rebuild,
    )
    assert report.verdict.kind is VerdictKind.REJECTED
    assert report.verdict.failing_frames == (0,)


def test_admitted_trajectory_cannot_be_forged(oracle, plant):
    report = admit(
        [dataclasses.replace(oracle, q=oracle.q + 1e-2)] * 3,
        gates=AdmissionGates(),
        times=TIMES[:3],
        plant=plant,
    )
    with pytest.raises(ValueError):
        AdmittedTrajectory.from_report(report)
    with pytest.raises(TypeError):
        AdmittedTrajectory(report=report, snapshots=report.snapshots)  # noqa


def test_residual_record_carries_evidence_axes(oracle, plant):
    report = admit(_traj(oracle), gates=AdmissionGates(), times=TIMES, plant=plant)
    rec = report.residual_record("phi_dot")
    assert rec.values.shape == (4,)
    assert rec.model_provenance == oracle.model_provenance
    assert rec.input_provenance == oracle.input_provenance
    with pytest.raises(ValueError):
        report.residual_record("nope")
