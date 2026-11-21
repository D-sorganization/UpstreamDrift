"""MuJoCo Robertson–Winter interface power (#15390)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_mujoco_available = importlib.util.find_spec("mujoco") is not None

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _mujoco_available, reason="mujoco not installed"),
]

TWO_LINK_XML = """
<mujoco>
  <option gravity="0 0 -9.80665"/>
  <worldbody>
    <body name="l1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="1 0 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.05" mass="2"/>
      <body name="l2" pos="0 0 -0.5">
        <joint name="j2" type="hinge" axis="1 0 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.05" mass="1.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

BRANCHED_XML = """
<mujoco>
  <option gravity="0 0 -9.80665"/>
  <worldbody>
    <body name="trunk" pos="0 0 1.2">
      <joint name="j_trunk" type="hinge" axis="0 1 0"/>
      <geom type="box" size="0.15 0.08 0.25" mass="8"/>
      <body name="left" pos="0.2 0 0">
        <joint name="j_left" type="hinge" axis="1 0 0"/>
        <geom type="capsule" fromto="0 0 0 0.35 0 0" size="0.04" mass="1.2"/>
      </body>
      <body name="right" pos="-0.2 0 0">
        <joint name="j_right" type="hinge" axis="1 0 0"/>
        <geom type="capsule" fromto="0 0 0 -0.35 0 0" size="0.04" mass="1.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _state_after_forward(model, qpos, qvel):
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    return data.qacc.copy()


@pytest.fixture
def two_link_model():
    import mujoco

    model = mujoco.MjModel.from_xml_string(TWO_LINK_XML)
    return model


@pytest.fixture
def branched_model():
    import mujoco

    return mujoco.MjModel.from_xml_string(BRANCHED_XML)


def test_joint_interface_differs_from_lumped_gravity_dot_v(two_link_model) -> None:
    """cfrc_int joint power must not reduce to ``-m g · v_com``."""
    from mujoco_humanoid_golf.interface_power import MujocoInterfacePowerAnalyzer

    model = two_link_model
    qpos = np.array([0.4, -0.6])
    qvel = np.array([2.1, -1.3])
    qacc = _state_after_forward(model, qpos, qvel)

    frame = MujocoInterfacePowerAnalyzer(model).frame_at(qpos, qvel, qacc)
    j2 = next(j for j in frame.joints if j.joint == "j2")
    assert abs(j2.force_power_on_child_w) > 0.1
    assert j2.power_on_child_w != pytest.approx(-5.430266606044359, rel=0.05)
    assert j2.power_on_child_w == pytest.approx(
        j2.force_power_on_child_w + j2.moment_power_on_child_w, abs=1e-12
    )


def test_parent_child_interface_sums_to_joint_coordinate_power(two_link_model) -> None:
    from mujoco_humanoid_golf.interface_power import MujocoInterfacePowerAnalyzer

    model = two_link_model
    qpos = np.array([0.4, -0.6])
    qvel = np.array([2.1, -1.3])
    qacc = _state_after_forward(model, qpos, qvel)
    frame = MujocoInterfacePowerAnalyzer(model).frame_at(qpos, qvel, qacc)
    for joint in frame.joints:
        interface_sum = joint.power_on_child_w + joint.power_on_parent_w
        assert interface_sum == pytest.approx(
            joint.joint_coordinate_power_w, rel=1e-6, abs=1e-5
        )


def test_finite_difference_energy_balance_two_link(two_link_model) -> None:
    from mujoco_humanoid_golf.interface_power import MujocoInterfacePowerAnalyzer

    model = two_link_model
    qpos = np.array([0.4, -0.6])
    qvel = np.array([2.1, -1.3])
    qacc = _state_after_forward(model, qpos, qvel)
    frame = MujocoInterfacePowerAnalyzer(model).frame_at(qpos, qvel, qacc)
    residuals = [s.balance_residual_w for s in frame.segments]
    assert max(abs(r) for r in residuals) < 0.05


def test_branched_trunk_sums_two_child_interfaces(branched_model) -> None:
    from mujoco_humanoid_golf.interface_power import MujocoInterfacePowerAnalyzer

    model = branched_model
    qpos = np.array([0.25, 0.35, -0.4])
    qvel = np.array([1.0, -0.8, 1.1])
    qacc = _state_after_forward(model, qpos, qvel)
    frame = MujocoInterfacePowerAnalyzer(model).frame_at(qpos, qvel, qacc)
    assert len(frame.joints) == 3
    trunk = next(s for s in frame.segments if s.body == "trunk")
    parent_iface = sum(
        j.power_on_parent_w for j in frame.joints if j.parent_body == "trunk"
    )
    assert trunk.total_interface_power_w == pytest.approx(parent_iface, abs=1e-10)
    assert max(abs(s.balance_residual_w) for s in frame.segments) < 0.08


def test_power_flow_reuses_interface_analyzer(two_link_model) -> None:
    from mujoco_humanoid_golf.power_flow import PowerFlowAnalyzer

    model = two_link_model
    qpos = np.array([0.2, 0.1])
    qvel = np.array([0.5, -0.3])
    qacc = np.zeros(model.nv)
    analyzer = PowerFlowAnalyzer(model)
    before = analyzer.compute_power_flow(qpos, qvel, qacc, qacc)
    rw1 = analyzer.compute_interface_power_frame(qpos, qvel, qacc)
    rw2 = analyzer.compute_interface_power_frame(qpos, qvel, qacc)
    assert before.joint_powers.shape == (model.nv,)
    assert rw1.joints[0].joint == rw2.joints[0].joint
    assert analyzer._interface_power is not None
