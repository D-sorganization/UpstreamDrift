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


@pytest.fixture
def two_link_model():
    import mujoco

    model = mujoco.MjModel.from_xml_string(TWO_LINK_XML)
    return model, mujoco.MjData(model)


def test_mujoco_segment_balance_residual_on_rigid_chain(two_link_model) -> None:
    from mujoco_humanoid_golf.interface_power import MujocoInterfacePowerAnalyzer

    import mujoco

    model, data = two_link_model
    qpos = np.array([0.4, -0.6])
    qvel = np.array([2.1, -1.3])
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    qacc = data.qacc.copy()

    analyzer = MujocoInterfacePowerAnalyzer(model)
    frame = analyzer.frame_at(qpos, qvel, qacc, time_s=0.0)
    residuals = [s.balance_residual_w for s in frame.segments if s.mass_kg > 0.0]
    assert residuals, "expected at least one massive segment"
    assert max(abs(r) for r in residuals) < 1e-7


def test_power_flow_exposes_robertson_winter_without_breaking_joint_powers(
    two_link_model,
) -> None:
    from mujoco_humanoid_golf.power_flow import PowerFlowAnalyzer

    model, _ = two_link_model
    qpos = np.array([0.2, 0.1])
    qvel = np.array([0.5, -0.3])
    qacc = np.zeros(model.nv)
    tau = np.zeros(model.nv)

    analyzer = PowerFlowAnalyzer(model)
    before = analyzer.compute_power_flow(qpos, qvel, qacc, tau)
    rw = analyzer.compute_interface_power_frame(qpos, qvel, qacc)
    assert before.joint_powers.shape == (model.nv,)
    assert len(rw.segments) >= 2
    assert any(s.total_interface_power_w != 0.0 for s in rw.segments)