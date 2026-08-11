"""Contracts for the one-arm, three-link attribution adapter."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.one_arm_attribution import (
    one_arm_joint_transfer_trajectory,
)
from src.shared.python.biomechanics.drift_control_transfer import (
    compute_path_frame,
    compute_power_and_work,
)
from src.shared.python.pendulum_simulator.physics_triple import (
    TriplePendulumParams,
    equations_of_motion,
    net_joint_forces,
)

pytestmark = pytest.mark.scientific


def _fixture() -> tuple[
    TriplePendulumParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    params = TriplePendulumParams(
        m1=2.1,
        m2=1.4,
        m3=0.45,
        L1=0.32,
        L2=0.30,
        L3=1.05,
        b1=0.08,
        b2=0.05,
        b3=0.02,
    )
    time = np.array([0.0, 0.04, 0.08])
    q = np.array([[-1.0, -0.8, -0.5], [-0.8, -0.5, -0.2], [-0.5, -0.2, 0.1]])
    v = np.array([[2.0, 1.5, 0.5], [3.0, 2.0, 1.0], [4.0, 2.5, 1.5]])
    controls = np.array([[35.0, 12.0, -6.0], [38.0, 10.0, 0.0], [40.0, 8.0, 7.0]])
    return params, time, q, v, controls


def test_one_arm_adapter_reports_all_joints_and_closes_attribution() -> None:
    params, time, q, v, controls = _fixture()
    trajectory = one_arm_joint_transfer_trajectory(time, q, v, controls, params)

    assert trajectory.joint_names == ("shoulder", "elbow", "wrist")
    assert trajectory.model_tier == "one_arm_three_link_point_mass"
    np.testing.assert_allclose(
        trajectory.force_total,
        trajectory.force_drift + trajectory.force_control,
    )
    np.testing.assert_allclose(
        trajectory.couple_total,
        trajectory.couple_drift + trajectory.couple_control,
    )
    power = compute_power_and_work(trajectory)
    np.testing.assert_allclose(
        power.total_work_total,
        power.total_work_drift + power.total_work_control,
    )


def test_one_arm_adapter_reproduces_existing_joint_force_solver() -> None:
    params, time, q, v, controls = _fixture()
    trajectory = one_arm_joint_transfer_trajectory(time, q, v, controls, params)
    sample = 1
    state = np.concatenate((q[sample], v[sample]))
    control = tuple(float(value) for value in controls[sample])
    qddot = equations_of_motion(state, time[sample], params, lambda _: control)[3:]
    expected = net_joint_forces(state, qddot, params)

    np.testing.assert_allclose(trajectory.force_total[sample, 0], expected["shoulder"])
    np.testing.assert_allclose(trajectory.force_total[sample, 1], expected["wrist1"])
    np.testing.assert_allclose(trajectory.force_total[sample, 2], expected["wrist2"])


def test_one_arm_moving_joints_have_path_frames_while_fixed_shoulder_does_not() -> None:
    params, time, q, v, controls = _fixture()
    trajectory = one_arm_joint_transfer_trajectory(time, q, v, controls, params)
    frame = compute_path_frame(trajectory.velocity, speed_epsilon=1e-9)

    assert not np.any(frame.valid[:, 0])
    assert np.all(frame.valid[:, 1:])
