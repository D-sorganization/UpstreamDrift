"""Unit tests for MujocoForceTorqueSource (FTO-9, #11294)."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import patch
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.shared.python.body_part_viz.mujoco_axial_loads import MujocoAxialLoadSource
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    WrenchKind,
)
from src.shared.python.force_overlay.conversions import (
    SegmentAxis,
    axial_loads_from_reactions,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

# Synthetic MJCF fixture 1: Hanging pendulum with single hinge about y
SYNTHETIC_HANGING_PENDULUM = """
<mujoco model="synthetic_hanging_pendulum">
  <option gravity="0 0 -9.81" timestep="0.001"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="hinge1" type="hinge" axis="0 1 0" pos="0 0 0"/>
      <geom name="geom1" type="capsule" fromto="0 0 0 0 0 -1" size="0.05" mass="2.0"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="hinge1" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""

# Synthetic MJCF fixture 2: Two-joint chain with non-axis-aligned joints
SYNTHETIC_TWO_JOINT_OBLIQUE = """
<mujoco model="synthetic_two_joint_oblique">
  <option gravity="0 0 -9.81" timestep="0.001"/>
  <worldbody>
    <body name="body1" pos="0 0 0">
      <joint name="jnt1" type="hinge" axis="0.6 0.8 0" pos="0 0 0"/>
      <geom name="geom1" type="cylinder" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1.5"/>
      <body name="body2" pos="0 0 -0.5">
        <joint name="jnt2" type="hinge" axis="0 0.6 0.8" pos="0 0 0"/>
        <geom name="geom2" type="cylinder" fromto="0 0 0 0 0 -0.5" size="0.03" mass="1.0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="act1" joint="jnt1" gear="1"/>
    <motor name="act2" joint="jnt2" gear="1" forcelimited="true" forcerange="-10 10"/>
  </actuator>
</mujoco>
"""

# Synthetic MJCF fixture 3: Box resting on ground plane
SYNTHETIC_BOX_ON_FLOOR = """
<mujoco model="synthetic_box_on_floor">
  <option gravity="0 0 -9.81" timestep="0.001"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom name="box_geom" type="box" size="0.5 0.5 0.5" mass="10.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_hanging_pendulum_reaction_equilibrium() -> None:
    """Hanging pendulum at rest with gravity emits +m*g reaction force at anchor."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
        MujocoForceTorqueSource,
    )

    model = mujoco.MjModel.from_xml_string(SYNTHETIC_HANGING_PENDULUM)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    source = MujocoForceTorqueSource(model)
    frame = source.sample(data)

    assert isinstance(frame, ForceTorqueFrame)
    assert frame.engine == "mujoco"
    assert frame.world_frame == "world_Zup"

    # Find reaction wrench on link1
    reactions = [
        w
        for w in frame.wrenches
        if w.kind == WrenchKind.JOINT_REACTION and w.body == "link1"
    ]
    assert len(reactions) == 1
    rx = reactions[0]

    # Mass is 2.0 kg, gravity is -9.81 m/s^2 -> weight is -19.62 N.
    # Parent-on-child reaction supporting the link must be (0, 0, +19.62) N.
    expected_force = 2.0 * 9.81
    assert rx.force_n is not None
    assert math.isclose(rx.force_n[0], 0.0, abs_tol=1e-5)
    assert math.isclose(rx.force_n[1], 0.0, abs_tol=1e-5)
    assert math.isclose(rx.force_n[2], expected_force, rel_tol=1e-5)

    # Anchor is pos="0 0 0"
    assert math.isclose(rx.point_m[0], 0.0, abs_tol=1e-5)
    assert math.isclose(rx.point_m[1], 0.0, abs_tol=1e-5)
    assert math.isclose(rx.point_m[2], 0.0, abs_tol=1e-5)


def test_sign_agreement_with_mujoco_axial_load_source() -> None:
    """Reaction force sign matches MujocoAxialLoadSource tension/compression."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
        MujocoForceTorqueSource,
    )

    model = mujoco.MjModel.from_xml_string(SYNTHETIC_HANGING_PENDULUM)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    source = MujocoForceTorqueSource(model)
    frame = source.sample(data)

    # Native axial load source
    axial_source = MujocoAxialLoadSource(model)
    native_axial = axial_source.sample(data)
    assert native_axial is not None
    assert "link1" in native_axial.values_n
    native_load = native_axial.values_n["link1"]

    # FTO-2 axial_loads_from_reactions
    axis = SegmentAxis(
        segment="link1",
        joint_label="reaction:hinge1",
        proximal_m=(0.0, 0.0, 0.0),
        distal_m=(0.0, 0.0, -1.0),
    )
    converted_axial = axial_loads_from_reactions(
        frame, (axis,), source="test_conversion"
    )
    assert converted_axial is not None
    assert "link1" in converted_axial.values_n
    converted_load = converted_axial.values_n["link1"]

    # Hanging under gravity is in tension -> positive load
    assert native_load is not None and native_load > 0
    assert converted_load is not None and converted_load > 0
    assert math.isclose(native_load, converted_load, rel_tol=1e-5)


def test_joint_reaction_label_uses_joint_name_like_other_engines() -> None:
    """Reactions are labelled by joint (Drake, Pinocchio, OpenSim convention).

    A jointed body is labelled by its first joint (the one whose anchor is
    reported); the wrench still acts on the body. Cross-engine consumers (the
    FTO-21 parity suite, ``axial_loads_from_reactions``) look reactions up by
    joint name.
    """
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
        MujocoForceTorqueSource,
    )

    model = mujoco.MjModel.from_xml_string(SYNTHETIC_TWO_JOINT_OBLIQUE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frame = MujocoForceTorqueSource(model).sample(data)

    labels = {
        w.body: w.label for w in frame.wrenches if w.kind == WrenchKind.JOINT_REACTION
    }
    assert labels == {"body1": "reaction:jnt1", "body2": "reaction:jnt2"}


def test_joint_reaction_label_falls_back_to_body_name_without_joint() -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
        MujocoForceTorqueSource,
    )

    xml = (
        '<mujoco><worldbody><body name="fixed_link" pos="0 0 1">'
        '<geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frame = MujocoForceTorqueSource(model).sample(data)
    labels = [w.label for w in frame.wrenches if w.kind == WrenchKind.JOINT_REACTION]
    assert labels in ([], ["reaction:fixed_link"])


def test_actuated_hinge_and_clamped_qfrc() -> None:
    """Motor torque uses qfrc_actuator (including clamp limits), not unconstrained ctrl."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
        MujocoForceTorqueSource,
    )

    model = mujoco.MjModel.from_xml_string(SYNTHETIC_TWO_JOINT_OBLIQUE)
    data = mujoco.MjData(model)

    # Actuator 1: ctrl = 5.0 (within range)
    # Actuator 2: ctrl = 50.0 (force limited to [-10, 10])
    data.ctrl[0] = 5.0
    data.ctrl[1] = 50.0
    mujoco.mj_forward(model, data)

    source = MujocoForceTorqueSource(model)
    frame = source.sample(data)

    actuators = {
        w.label: w for w in frame.wrenches if w.kind == WrenchKind.JOINT_ACTUATOR
    }

    assert "actuator:jnt1" in actuators
    assert "actuator:jnt2" in actuators

    w1 = actuators["actuator:jnt1"]
    w2 = actuators["actuator:jnt2"]

    # Actuator 1: axis is [0.6, 0.8, 0], torque = 5.0 * [0.6, 0.8, 0] = [3.0, 4.0, 0.0]
    assert w1.torque_nm is not None
    np.testing.assert_allclose(w1.torque_nm, [3.0, 4.0, 0.0], atol=1e-5)
    assert w1.force_n is None

    # Actuator 2: ctrl is 50.0, but clamped by forcerange to 10.0
    # Axis is [0, 0.6, 0.8] -> torque = 10.0 * [0, 0.6, 0.8] = [0.0, 6.0, 8.0]
    assert w2.torque_nm is not None
    np.testing.assert_allclose(w2.torque_nm, [0.0, 6.0, 8.0], atol=1e-5)


def test_two_joint_chain_non_axis_aligned_slicing() -> None:
    """Joint axis must be read from data.xaxis[j], avoiding 3*j slicing error."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
        MujocoForceTorqueSource,
    )

    model = mujoco.MjModel.from_xml_string(SYNTHETIC_TWO_JOINT_OBLIQUE)
    data = mujoco.MjData(model)
    data.ctrl[0] = 1.0
    data.ctrl[1] = 1.0
    mujoco.mj_forward(model, data)

    source = MujocoForceTorqueSource(model)
    frame = source.sample(data)

    actuators = {
        w.label: w for w in frame.wrenches if w.kind == WrenchKind.JOINT_ACTUATOR
    }
    w2 = actuators["actuator:jnt2"]
    assert w2.torque_nm is not None
    # If the old bug `xaxis[3*j:3*j+3]` was present, j=1 would index out of bounds or read wrong memory
    np.testing.assert_allclose(w2.torque_nm, [0.0, 0.6, 0.8], atol=1e-5)


def test_box_resting_on_floor_contact_equilibrium() -> None:
    """Resting box contact forces sum to (0, 0, m*g) applied on floor plane."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
        MujocoForceTorqueSource,
    )

    model = mujoco.MjModel.from_xml_string(SYNTHETIC_BOX_ON_FLOOR)
    data = mujoco.MjData(model)

    # Settle box on floor
    data.qpos[2] = 0.5  # half-extent is 0.5, so bottom touches z=0
    for _ in range(500):
        mujoco.mj_step(model, data)

    source = MujocoForceTorqueSource(model)
    frame = source.sample(data)

    contacts = [
        w for w in frame.wrenches if w.kind == WrenchKind.CONTACT and w.body == "box"
    ]
    assert len(contacts) > 0

    total_contact_force = np.zeros(3)
    for c in contacts:
        assert c.force_n is not None
        total_contact_force += np.array(c.force_n)
        # Contact points should lie at z ≈ 0
        assert math.isclose(c.point_m[2], 0.0, abs_tol=0.05)

    # Box mass is 10.0 kg -> weight = 98.1 N
    expected_total_grf = 10.0 * 9.81
    assert math.isclose(total_contact_force[0], 0.0, abs_tol=0.5)
    assert math.isclose(total_contact_force[1], 0.0, abs_tol=0.5)
    assert math.isclose(total_contact_force[2], expected_total_grf, rel_tol=0.05)


def test_caller_data_immutability() -> None:
    """Caller MjData arrays are bit-identical before and after sample()."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
        MujocoForceTorqueSource,
    )

    model = mujoco.MjModel.from_xml_string(SYNTHETIC_HANGING_PENDULUM)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.42
    data.qvel[0] = -1.23
    mujoco.mj_forward(model, data)

    qpos_before = data.qpos.copy()
    qvel_before = data.qvel.copy()
    cfrc_int_before = data.cfrc_int.copy()

    source = MujocoForceTorqueSource(model)
    _ = source.sample(data)

    np.testing.assert_array_equal(data.qpos, qpos_before)
    np.testing.assert_array_equal(data.qvel, qvel_before)
    np.testing.assert_array_equal(data.cfrc_int, cfrc_int_before)


def test_engine_delegation_and_source_caching() -> None:
    """Engine caches MujocoForceTorqueSource and reuses across calls."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )

    engine = MuJoCoPhysicsEngine()
    engine.load_from_string(SYNTHETIC_HANGING_PENDULUM)

    frame1 = engine.get_force_torque_frame()
    assert frame1 is not None

    with patch(
        "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source.MujocoForceTorqueSource.__init__",
        side_effect=AssertionError("Should not re-construct source"),
    ):
        for _ in range(100):
            frame = engine.get_force_torque_frame()
            assert frame is not None
            loads = engine.get_segment_axial_loads()
            assert loads is not None
            cforces = engine.get_contact_forces()
            assert cforces is not None

    caps = engine.get_capabilities()
    assert caps.force_visualization.name.lower() == "full"


def test_frame_roundtrip_to_dict() -> None:
    """Generated ForceTorqueFrame validates against schema dictionary representation."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
        MujocoForceTorqueSource,
    )

    model = mujoco.MjModel.from_xml_string(SYNTHETIC_HANGING_PENDULUM)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    source = MujocoForceTorqueSource(model)
    frame = source.sample(data)

    frame_dict = frame.to_dict()
    assert frame_dict["schema_version"] == "force-torque-frame-v1"
    assert frame_dict["engine"] == "mujoco"
    assert frame_dict["world_frame"] == "world_Zup"
    assert len(frame_dict["wrenches"]) > 0

    reconstructed = ForceTorqueFrame.from_dict(frame_dict)
    assert reconstructed.time_s == frame.time_s
    assert len(reconstructed.wrenches) == len(frame.wrenches)


# --- GCV-8 (#11714): GRIP wrenches from the grip weld efc_force ---------------

_GRIP_HELD_CLUB = """
<mujoco><option gravity="0 0 -9.81"/>
<worldbody>
 <body name="hand_r" mocap="true" pos="0.1 0 1"><site name="hr" size="0.01"/></body>
 <body name="hand_l" mocap="true" pos="-0.1 0 1"><site name="hl" size="0.01"/></body>
 <body name="club" pos="0 0 1"><freejoint/>
  <geom type="box" size="0.2 0.02 0.02" mass="0.5"/>
  <site name="cr" pos="0.1 0 0" size="0.01"/><site name="cl" pos="-0.1 0 0" size="0.01"/>
 </body>
</worldbody>
<equality>
 <weld name="grip_weld_r" site1="hr" site2="cr"/>
 <weld name="grip_weld_l" site1="hl" site2="cl"/>
</equality>
</mujoco>"""


def test_sample_emits_grip_wrenches_that_balance_the_club_weight() -> None:
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
        MujocoForceTorqueSource,
    )

    model = mujoco.MjModel.from_xml_string(_GRIP_HELD_CLUB)
    data = mujoco.MjData(model)
    for _ in range(3000):
        mujoco.mj_step(model, data)
    frame = MujocoForceTorqueSource(model).sample(data)
    grip = {w.label: w for w in frame.wrenches if w.kind is WrenchKind.GRIP}
    assert {"grip:hand_left", "grip:hand_right", "grip:net_midpoint"} <= set(grip)
    net = np.array(grip["grip:net_midpoint"].force_n)
    np.testing.assert_allclose(net, [0.0, 0.0, 0.5 * 9.81], atol=1e-3)


def test_sample_without_grip_welds_emits_no_grip_wrenches() -> None:
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
        MujocoForceTorqueSource,
    )

    model = mujoco.MjModel.from_xml_string(SYNTHETIC_HANGING_PENDULUM)
    data = mujoco.MjData(model)
    frame = MujocoForceTorqueSource(model).sample(data)
    assert not [w for w in frame.wrenches if w.kind is WrenchKind.GRIP]
