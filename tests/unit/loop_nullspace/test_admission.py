"""Unit tests for ClosedLoopSnapshot kinematic and contact admission gate (#12559)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.loop_nullspace.admission import (
    AdmissionGates,
    AdmissionVerdict,
    AdmittedTrajectory,
    admit,
)
from src.shared.python.loop_nullspace.contracts import (
    InputProvenance,
)
from src.shared.python.loop_nullspace.fixtures import (
    planar_golfer_snapshot,
    spatial_two_arm_club_snapshot,
    straight_arm_snapshot,
)

pytestmark = pytest.mark.unit


def test_admit_accepts_analytic_oracle_snapshots() -> None:
    """admit() returns ADMITTED and an AdmittedTrajectory for all analytic oracle fixtures."""
    planar_snap, _ = planar_golfer_snapshot()
    straight_snap, _ = straight_arm_snapshot()
    spatial_fixed, _ = spatial_two_arm_club_snapshot(
        floating_base=False, include_feet=False
    )
    spatial_floating, _ = spatial_two_arm_club_snapshot(
        floating_base=True, include_feet=True
    )

    for snap in (planar_snap, straight_snap, spatial_fixed, spatial_floating):
        report = admit(snap)
        assert report.verdict == AdmissionVerdict.ADMITTED
        assert report.is_admitted is True
        assert report.failing_frames == ()
        assert report.admitted_trajectory is not None
        assert len(report.admitted_trajectory) == 1
        assert report.admitted_trajectory[0] is snap
        diag = report.frame_diagnostics[0]
        assert diag.passed is True
        assert diag.phi_scaled <= AdmissionGates().phi_max
        assert diag.phi_dot_scaled <= AdmissionGates().phi_dot_max
        assert diag.phi_ddot_scaled <= AdmissionGates().phi_ddot_max


def test_admitted_trajectory_cannot_be_constructed_directly() -> None:
    """AdmittedTrajectory enforces token guard so callers cannot bypass admit()."""
    snap, _ = planar_golfer_snapshot()
    with pytest.raises(ValueError, match="admit"):
        AdmittedTrajectory(
            snapshots=(snap,),
            verdict=AdmissionVerdict.ADMITTED,
        )


def test_admit_rejects_position_velocity_and_acceleration_closure_violations() -> None:
    """Perturbing position, velocity, or acceleration closure residual triggers REJECTED with frame index and reason."""
    base, _ = spatial_two_arm_club_snapshot(floating_base=False, include_feet=False)

    # 1. Position closure violation via q perturbation
    q_pert = base.q.copy()
    q_pert[0] += 0.02
    bad_pos = replace(base, q=q_pert)
    rep_pos = admit([base, bad_pos])
    assert rep_pos.verdict == AdmissionVerdict.REJECTED
    assert rep_pos.is_admitted is False
    assert rep_pos.admitted_trajectory is None
    assert rep_pos.failing_frames == (1,)
    assert any("scaled Phi " in r for r in rep_pos.frame_diagnostics[1].reasons)

    # 2. Velocity closure violation via v perturbation
    v_pert = base.v.copy()
    v_pert[0] += 0.25
    bad_vel = replace(base, v=v_pert)
    rep_vel = admit([base, bad_vel])
    assert rep_vel.verdict == AdmissionVerdict.REJECTED
    assert rep_vel.failing_frames == (1,)
    assert any("scaled Phi_dot" in r for r in rep_vel.frame_diagnostics[1].reasons)

    # 3. Acceleration closure violation via a perturbation
    a_pert = base.a.copy()
    a_pert[1] += 5.0
    bad_acc = replace(base, a=a_pert)
    rep_acc = admit([base, bad_acc])
    assert rep_acc.verdict == AdmissionVerdict.REJECTED
    assert rep_acc.failing_frames == (1,)
    assert any("scaled Phi_ddot" in r for r in rep_acc.frame_diagnostics[1].reasons)


def test_admit_rejects_stance_foot_contact_slip() -> None:
    """Active stance foot with non-zero contact velocity triggers REJECTED."""
    base, _ = spatial_two_arm_club_snapshot(floating_base=True, include_feet=True)
    assert len(base.foot_contacts) == 2

    # Perturb floating-base pelvis translation x velocity -> stance foot slips while hands stay still
    v_slip = base.v.copy()
    v_slip[0] += 0.5
    bad_foot = replace(base, v=v_slip)
    report = admit(bad_foot)
    assert report.verdict == AdmissionVerdict.REJECTED
    assert report.contact_mode_ok is False
    assert report.failing_frames == (0,)
    assert any("stance foot velocity" in r for r in report.frame_diagnostics[0].reasons)


def test_admit_rejects_bad_derivative_provenance_and_time_sync() -> None:
    """Excessive finite-difference noise floor, unsynchronized time status, or non-increasing timestamps are rejected."""
    base, _ = planar_golfer_snapshot()

    # 1. Excessive finite-difference noise floor:
    # sigma_qddot = sqrt(6) * 0.01 / (0.005)^2 = 979.8 m/s^2 -> scaled = 979.8 / 400.0 = 2.45 > 1.0
    noisy_gates = AdmissionGates(
        position_noise_std_m=0.01, max_accel_noise_floor_scaled=1.0
    )
    rep_noise = admit(base, gates=noisy_gates, dt_s=0.005)
    assert rep_noise.verdict == AdmissionVerdict.REJECTED
    assert rep_noise.derivative_noise_floor > 400.0
    assert any(
        "derivative provenance/noise floor rejected" in r
        for r in rep_noise.frame_diagnostics[0].reasons
    )

    # 2. Unsynchronized time status
    unsync_prov = InputProvenance(
        kinematic_source=base.input_provenance.kinematic_source,
        derivative_method=base.input_provenance.derivative_method,
        time_sync_status="UNSYNCHRONIZED",
    )
    rep_sync = admit(replace(base, input_provenance=unsync_prov))
    assert rep_sync.verdict == AdmissionVerdict.REJECTED
    assert rep_sync.time_sync_ok is False
    assert any("time_sync_status" in r for r in rep_sync.frame_diagnostics[0].reasons)

    # 3. Non-increasing timestamps across a multi-frame trajectory
    rep_time = admit([base, base], time_s=[0.010, 0.005])
    assert rep_time.verdict == AdmissionVerdict.REJECTED
    assert rep_time.time_sync_ok is False
    assert rep_time.failing_frames == (0, 1)
    assert any("time_s is not strictly increasing" in r for r in rep_time.reasons)


def test_admit_repair_projects_to_closure_and_records_reversible_diagnostics() -> None:
    """admit(..., repair='project_to_closure') repairs small closure drift into REPAIRED with full RepairRecord."""
    base, _ = spatial_two_arm_club_snapshot(floating_base=False, include_feet=False)

    dq = np.zeros_like(base.q)
    dq[0] = 1e-3
    dv = np.zeros_like(base.v)
    dv[1] = 2e-2
    da = np.zeros_like(base.a)
    da[2] = 0.5
    drifted = replace(base, q=base.q + dq, v=base.v + dv, a=base.a + da)

    # Without repair, drifted snapshot is rejected
    rejected = admit(drifted, repair="none")
    assert rejected.verdict == AdmissionVerdict.REJECTED
    assert rejected.admitted_trajectory is None

    # With repair="project_to_closure", drifted snapshot is repaired and admitted
    repaired_report = admit(drifted, repair="project_to_closure")
    assert repaired_report.verdict == AdmissionVerdict.REPAIRED
    assert repaired_report.is_admitted is True
    assert repaired_report.failing_frames == ()
    assert repaired_report.admitted_trajectory is not None
    assert repaired_report.repair_record is not None

    rec = repaired_report.repair_record
    assert rec.method == "project_to_closure"
    assert rec.repaired_frames == (0,)
    assert rec.original_snapshots[0] is drifted
    assert rec.max_delta_q_norm > 0.0
    assert rec.max_delta_v_norm > 0.0
    assert rec.max_delta_a_norm > 0.0
    assert rec.phi_scaled_before[0] > AdmissionGates().phi_max
    assert rec.phi_scaled_after[0] <= 1e-12
    assert rec.phi_dot_scaled_after[0] <= 1e-12
    assert rec.phi_ddot_scaled_after[0] <= 1e-12
