"""Unit tests for sequential and isolated joint kinematics (#15666)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.motion_matching.isolated_joint_kinematics import (
    IsolatedJointResult,
    JointRomSpec,
    SequentialSweepReport,
    build_synthetic_arm_club_fk,
    isolate_joint_trajectory,
    run_sequential_joint_sweep,
    validate_isolated_trajectory,
)

pytestmark = pytest.mark.unit
ORDER = ("shoulder_yaw", "shoulder_pitch", "elbow_flex", "wrist_cock", "wrist_roll")


def test_isolate_joint_trajectory_locks_non_target_dofs_strictly() -> None:
    """Non-target DOFs must remain bitwise identical to reference posture."""
    names, ref_q = ("shoulder_yaw", "elbow_flex", "wrist_cock"), (0.15, -0.45, 0.22)
    rom = JointRomSpec("elbow_flex", -1.2, 0.2, 1)
    res = isolate_joint_trajectory(names, ref_q, "elbow_flex", rom, n_steps=9)
    assert isinstance(res, IsolatedJointResult) and res.max_locked_drift_rad == 0.0
    assert res.sweep_angles_rad[0] == pytest.approx(-1.2)
    assert res.sweep_angles_rad[-1] == pytest.approx(0.2)
    traj = np.asarray(res.trajectory_q_rad, dtype=float)
    assert traj.shape == (9, 3)
    assert np.array_equal(traj[:, 0], np.full(9, 0.15))
    assert np.array_equal(traj[:, 2], np.full(9, 0.22))


def test_isolate_joint_trajectory_fails_closed_on_unknown_and_rom() -> None:
    """Unknown joint names and out-of-ROM sweep angles must raise ValueError."""
    names = ("shoulder_yaw", "elbow_flex")
    bad_rom = JointRomSpec("ankle_flex", -0.5, 0.5)
    with pytest.raises(ValueError, match="Unknown target joint"):
        isolate_joint_trajectory(names, (0.0, 0.0), "ankle_flex", bad_rom)
    rom = JointRomSpec("elbow_flex", -1.0, 0.2)
    for bad in ((-1.05, -0.5, 0.0), (-0.5, 0.0, 0.25)):
        with pytest.raises(ValueError, match="ROM bounds"):
            isolate_joint_trajectory(
                names, (0.0, 0.0), "elbow_flex", rom, sweep_angles_rad=bad
            )


def test_joint_rom_spec_rejects_invalid_bounds() -> None:
    """ROM specification must require finite bounds with q_min_rad < q_max_rad."""
    for lo, hi in ((0.5, 0.5), (0.5, -0.5), (-math.inf, 0.5)):
        with pytest.raises(ValueError):
            JointRomSpec("elbow_flex", lo, hi)


def test_validate_isolated_trajectory_rejects_contamination() -> None:
    """Any mutation of non-target DOFs must fail closed."""
    ref_q = np.array([0.1, -0.3, 0.2], dtype=float)
    bad = np.tile(ref_q, (5, 1))
    bad[:, 1] = np.linspace(-0.5, 0.1, 5)
    bad[2, 0] += 1e-9
    with pytest.raises(ValueError, match="Locked-DOF contamination"):
        validate_isolated_trajectory(bad, ref_q, target_index=1)
    ref_t = tuple(float(v) for v in ref_q)
    ang_t = tuple(float(v) for v in bad[:, 1])
    traj_t = tuple(tuple(float(x) for x in r) for r in bad)
    rom = JointRomSpec("elbow_flex", -0.6, 0.2)
    with pytest.raises(ValueError, match="Locked-DOF contamination"):
        IsolatedJointResult(
            "elbow_flex", 1, ("a", "elbow_flex", "c"), ref_t, ang_t, traj_t, 0.0, rom
        )


def test_synthetic_arm_club_fk_geometry_at_zero_posture() -> None:
    """Synthetic FK fixture returns exact grip and clubhead poses at zero."""
    fk = build_synthetic_arm_club_fk()
    club_pos, club_rot, grip_pos = fk.evaluate(np.zeros(5, dtype=float))
    assert fk.joint_names == ORDER
    assert np.allclose(grip_pos, np.array([0.56, 0.0, 0.0]))
    assert np.allclose(club_pos, np.array([1.66, 0.05, 0.0]))
    assert np.allclose(club_rot, np.eye(3))


def test_run_sequential_joint_sweep_sensitivities_and_grip_drift() -> None:
    """Sequential sweep preserves order, zero locked drift, and sensitivities."""
    report = run_sequential_joint_sweep(build_synthetic_arm_club_fk(), n_steps=11)
    assert isinstance(report, SequentialSweepReport)
    assert report.max_locked_drift_rad == 0.0 and report.sweep_order == ORDER
    by_name = {e.joint_name: e for e in report.joint_sweeps}
    expected = {
        "shoulder_yaw": math.hypot(1.66, 0.05),
        "shoulder_pitch": 1.66,
        "elbow_flex": math.hypot(1.36, 0.05),
        "wrist_cock": 1.10,
        "wrist_roll": 0.05,
    }
    for name, sens in expected.items():
        e = by_name[name]
        assert e.isolated_result.max_locked_drift_rad == 0.0
        assert e.clubhead_displacement_sensitivity_m_per_rad == pytest.approx(
            sens, rel=1e-4
        )
        assert e.clubhead_orientation_sensitivity_rad_per_rad == pytest.approx(
            1.0, rel=1e-4
        )
        assert e.max_clubhead_displacement_m > 0.0
    for prox in ORDER[:3]:
        assert by_name[prox].max_grip_drift_m > 0.1
    assert by_name["wrist_cock"].max_grip_drift_m == pytest.approx(0.0, abs=1e-12)
    assert by_name["wrist_roll"].max_grip_drift_m == pytest.approx(0.0, abs=1e-12)


def test_run_sequential_joint_sweep_rejects_bad_order() -> None:
    """Supplying a distal-before-proximal order must fail closed."""
    fk = build_synthetic_arm_club_fk()
    with pytest.raises(ValueError, match="proximal-to-distal"):
        run_sequential_joint_sweep(fk, sweep_order=tuple(reversed(ORDER)))
