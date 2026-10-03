"""Unit tests for MuJoCo force and torque provider (ADR-0052, #11294)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import jsonschema
import mujoco
import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (
    MujocoForceTorqueSource,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
    MuJoCoPhysicsEngine,
)
from src.shared.python.body_part_viz.mujoco_axial_loads import MujocoAxialLoadSource
from src.shared.python.engine_core.engine_availability import skip_if_unavailable
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    WrenchKind,
)
from src.shared.python.force_overlay.conversions import (
    SegmentAxis,
    axial_loads_from_reactions,
)
from tests.integration.cross_engine.force_overlay_fixtures import (
    BOX,
    HINGE,
    LINK,
    STANDARD,
    PendulumFixture,
)
from tests.integration.cross_engine.test_force_overlay_parity import (
    mujoco_pendulum_mjcf,
    mujoco_resting_mjcf,
)

pytestmark = [skip_if_unavailable("mujoco"), pytest.mark.unit]


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


@pytest.fixture
def schema(repo_root: Path) -> dict:
    schema_path = repo_root / "schemas" / "force-torque-frame-v1.json"
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def test_hanging_pendulum_reaction_is_weight_up_and_tension() -> None:
    """1. Hanging pendulum at rest: JOINT_REACTION force is (0, 0, +m*g) within 1e-6*m*g."""
    fx = STANDARD
    m = mujoco.MjModel.from_xml_string(mujoco_pendulum_mjcf(fx))
    d = mujoco.MjData(m)
    d.qpos[0] = 0.0
    d.qvel[0] = 0.0

    source = MujocoForceTorqueSource(m)
    frame = source.sample(d)

    reactions = [
        w
        for w in frame.by_kind(WrenchKind.JOINT_REACTION)
        if w.body == LINK and w.label.split(":", 1)[1].startswith(HINGE)
    ]
    assert len(reactions) == 1
    rx = reactions[0]
    assert rx.force_n is not None
    expected_force = (0.0, 0.0, fx.weight)
    np.testing.assert_allclose(
        rx.force_n,
        expected_force,
        rtol=1e-6,
        atol=1e-6 * fx.weight,
    )
    np.testing.assert_allclose(rx.point_m, fx.pivot, rtol=1e-6, atol=1e-6)


def test_sign_agreement_axial_loads() -> None:
    """2. Sign agreement: axial_loads_from_reactions matches MujocoAxialLoadSource."""
    fx = STANDARD
    # Test on pendulum with a capsule geom so MujocoAxialLoadSource discovers the axis
    mjcf = f"""
    <mujoco model="pendulum_capsule">
      <option gravity="0 0 {-fx.gravity}"/>
      <worldbody>
        <body name="{LINK}" pos="0 0 {fx.pivot_height}">
          <joint name="{HINGE}" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-fx.length}" size="0.05" mass="{fx.mass}"/>
        </body>
      </worldbody>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(mjcf)
    d = mujoco.MjData(m)
    d.qpos[0] = 0.0
    d.qvel[0] = 0.0

    source = MujocoForceTorqueSource(m)
    frame = source.sample(d)

    assert frame.axial_loads is not None
    native_axial = frame.axial_loads.values_n.get(LINK)
    assert native_axial is not None
    assert native_axial == pytest.approx(fx.weight, rel=1e-5)

    # Compute via axial_loads_from_reactions helper
    axes = [
        SegmentAxis(
            segment=LINK,
            joint_label=f"joint_reaction:{HINGE}",
            proximal_m=fx.pivot,
            distal_m=(0.0, 0.0, fx.pivot_height - fx.length),
        )
    ]
    reconstructed = axial_loads_from_reactions(frame, axes, source="test")
    rec_val = reconstructed.values_n.get(LINK)
    assert rec_val is not None
    assert rec_val == pytest.approx(native_axial, rel=1e-6, abs=1e-6)


def test_actuated_hinge_torque() -> None:
    """3. Actuated hinge with motor gear 1 and ctrl=5: JOINT_ACTUATOR is 5*axis. Clamped uses qfrc."""
    # Model with clamped actuator: ctrlrange="-3 3"
    mjcf = """
    <mujoco model="hinge_test">
      <worldbody>
        <body name="arm" pos="0 0 1">
          <joint name="j1" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="m1" joint="j1" gear="1" ctrllimited="true" ctrlrange="-3 3"/>
      </actuator>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(mjcf)
    d = mujoco.MjData(m)

    source = MujocoForceTorqueSource(m)

    # Within limits: ctrl=2.5 -> torque=2.5*axis
    d.ctrl[0] = 2.5
    frame = source.sample(d)
    actuators = frame.by_kind(WrenchKind.JOINT_ACTUATOR)
    assert len(actuators) == 1
    assert actuators[0].torque_nm is not None
    np.testing.assert_allclose(actuators[0].torque_nm, (0.0, 2.5, 0.0), atol=1e-6)

    # Clamped: ctrl=5.0 -> torque clamped to 3.0 (from qfrc_actuator)
    d.ctrl[0] = 5.0
    frame_clamped = source.sample(d)
    actuators_clamped = frame_clamped.by_kind(WrenchKind.JOINT_ACTUATOR)
    assert len(actuators_clamped) == 1
    assert actuators_clamped[0].torque_nm is not None
    np.testing.assert_allclose(
        actuators_clamped[0].torque_nm, (0.0, 3.0, 0.0), atol=1e-6
    )


def test_two_joint_chain_xaxis_row_slicing() -> None:
    """4. Two-joint chain with non-axis-aligned joints: verify xaxis[j] row slicing."""
    mjcf = """
    <mujoco model="chain_test">
      <worldbody>
        <body name="link1" pos="0 0 1">
          <joint name="j0" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.1" mass="1"/>
          <body name="link2" pos="0 0 -0.5">
            <joint name="j1" type="hinge" axis="1 0 0"/>
            <geom type="sphere" size="0.1" mass="1"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="m0" joint="j0" gear="1"/>
        <motor name="m1" joint="j1" gear="1"/>
      </actuator>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(mjcf)
    d = mujoco.MjData(m)
    d.ctrl[0] = 2.0
    d.ctrl[1] = 3.0

    source = MujocoForceTorqueSource(m)
    frame = source.sample(d)

    actuators = {w.label: w for w in frame.by_kind(WrenchKind.JOINT_ACTUATOR)}
    assert "actuator:j0" in actuators
    assert "actuator:j1" in actuators

    # j0 is about y: torque is (0, 2, 0)
    assert actuators["actuator:j0"].torque_nm is not None
    np.testing.assert_allclose(
        actuators["actuator:j0"].torque_nm, (0.0, 2.0, 0.0), atol=1e-6
    )

    # j1 is about x: torque is (3, 0, 0).
    # If 3*j row slice was used on shape (2,3), indexing row 3 would crash or fail.
    assert actuators["actuator:j1"].torque_nm is not None
    np.testing.assert_allclose(
        actuators["actuator:j1"].torque_nm, (3.0, 0.0, 0.0), atol=1e-6
    )


def test_box_resting_contact_forces() -> None:
    """5. Box resting on floor: sum of CONTACT forces on box approx (0, 0, m*g), points near z=0."""
    fx = STANDARD
    m = mujoco.MjModel.from_xml_string(mujoco_resting_mjcf(fx))
    d = mujoco.MjData(m)
    d.qpos[:3] = [0.0, 0.0, fx.box_rest_height]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    for _ in range(round(0.2 / 0.002)):
        mujoco.mj_step(m, d)

    source = MujocoForceTorqueSource(m)
    frame = source.sample(d)

    contacts = [w for w in frame.by_kind(WrenchKind.CONTACT) if w.body == BOX]
    assert contacts, "No contacts found for box"
    forces = [w.force_n for w in contacts]
    assert all(f is not None for f in forces)
    total_force = np.sum(forces, axis=0)
    np.testing.assert_allclose(
        total_force,
        (0.0, 0.0, fx.weight),
        rtol=0.02,
        atol=0.05,
    )
    assert all(abs(w.point_m[2]) <= 0.01 for w in contacts)


def test_caller_data_unmutated() -> None:
    """6. Caller's data (qpos, qvel, cfrc_int) bit-identical before and after sample."""
    fx = STANDARD
    m = mujoco.MjModel.from_xml_string(mujoco_pendulum_mjcf(fx))
    d = mujoco.MjData(m)
    d.qpos[0] = 0.42
    d.qvel[0] = -1.23

    # Run forward to populate initial state
    mujoco.mj_forward(m, d)

    qpos_before = d.qpos.copy()
    qvel_before = d.qvel.copy()
    cfrc_int_before = d.cfrc_int.copy()

    source = MujocoForceTorqueSource(m)
    _ = source.sample(d)

    np.testing.assert_array_equal(d.qpos, qpos_before)
    np.testing.assert_array_equal(d.qvel, qvel_before)
    np.testing.assert_array_equal(d.cfrc_int, cfrc_int_before)


def test_axial_source_constructed_once() -> None:
    """7. get_segment_axial_loads source constructed once across 100 calls."""
    engine = MuJoCoPhysicsEngine()
    fx = STANDARD
    mjcf = f"""
    <mujoco model="pendulum_capsule">
      <option gravity="0 0 {-fx.gravity}"/>
      <worldbody>
        <body name="{LINK}" pos="0 0 {fx.pivot_height}">
          <joint name="{HINGE}" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-fx.length}" size="0.05" mass="{fx.mass}"/>
        </body>
      </worldbody>
    </mujoco>
    """
    engine.load_from_string(mjcf, "xml")

    with patch(
        "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source.MujocoAxialLoadSource",
        side_effect=MujocoAxialLoadSource,
    ) as spy_source:
        for _ in range(100):
            _ = engine.get_segment_axial_loads()

        # MujocoAxialLoadSource was constructed only once across all 100 calls
        assert spy_source.call_count == 1


def test_schema_validation_roundtrip(schema: dict) -> None:
    """8. Schema validation: to_dict validates against force-torque-frame-v1.json and roundtrips."""
    fx = STANDARD
    m = mujoco.MjModel.from_xml_string(mujoco_pendulum_mjcf(fx))
    d = mujoco.MjData(m)
    d.qpos[0] = fx.inverted_angle
    d.ctrl[0] = fx.hold_torque

    source = MujocoForceTorqueSource(m)
    frame = source.sample(d, include_gravity=True)

    data = frame.to_dict()
    # Validate against JSON schema
    jsonschema.validate(instance=data, schema=schema)

    # Round-trip through from_dict
    roundtrip = ForceTorqueFrame.from_dict(data)
    assert roundtrip.time_s == frame.time_s
    assert roundtrip.engine == frame.engine
    assert roundtrip.world_frame == frame.world_frame
    assert len(roundtrip.wrenches) == len(frame.wrenches)
