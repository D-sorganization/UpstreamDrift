"""Tests for the Pinocchio force/torque provider (FTO-13, #11298)."""

from __future__ import annotations

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
if type(pin).__module__ == "unittest.mock" or not all(
    hasattr(pin, n) for n in ("rnea", "buildModelFromXML", "SE3", "Model")
):  # tests/unit/conftest.py mocks pinocchio when it is not installed
    pytest.skip(
        "real pinocchio runtime required (found mock/stub)", allow_module_level=True
    )

from src.engines.physics_engines.pinocchio.python.pinocchio_force_torque import (  # noqa: E402
    PinocchioForceTorqueSource,
)
from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (  # noqa: E402
    PinocchioPhysicsEngine,
)
from src.shared.python.force_overlay import (  # noqa: E402
    ForceTorqueProvider,
    WrenchKind,
)
from src.shared.python.motion_matching.contact_law import ContactSample  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

MASS = 2.0
LENGTH = 0.5
G = 9.81


def _pendulum_urdf(rpy: str = "0 0 0") -> str:
    return f"""<robot name="synthetic_pendulum">
  <link name="synthetic_base"/>
  <link name="synthetic_rod">
    <inertial>
      <origin xyz="0 0 -{LENGTH}"/>
      <mass value="{MASS}"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="synthetic_hinge" type="revolute">
    <parent link="synthetic_base"/><child link="synthetic_rod"/>
    <origin xyz="0 0 1" rpy="{rpy}"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="100" velocity="10"/>
  </joint>
</robot>"""


def _source(rpy: str = "0 0 0") -> PinocchioForceTorqueSource:
    return PinocchioForceTorqueSource(pin.buildModelFromXML(_pendulum_urdf(rpy)))


def _static(src: PinocchioForceTorqueSource, tau: float = 0.0):
    z = np.zeros(1)
    return src.sample(z, z, z, np.array([tau]))


def test_hanging_reaction_is_world_weight_at_joint_origin() -> None:
    frame = _static(_source())
    (w,) = frame.by_kind(WrenchKind.JOINT_REACTION)
    assert w.body == "synthetic_rod"
    assert w.label == "reaction:synthetic_hinge"
    np.testing.assert_allclose(w.force_n, (0, 0, MASS * G), atol=1e-9)
    np.testing.assert_allclose(w.point_m, (0, 0, 1), atol=1e-12)
    assert frame.axial_loads is not None
    assert frame.axial_loads.values_n["synthetic_rod"] == pytest.approx(MASS * G)


def test_reaction_is_world_frame_after_base_rotation() -> None:
    """Regression: local-frame f must not be reported as world."""
    src = _source("1.5707963267948966 0 0")
    frame = _static(src)
    (w,) = frame.by_kind(WrenchKind.JOINT_REACTION)
    np.testing.assert_allclose(w.force_n, (0, 0, MASS * G), atol=1e-9)
    data = src.model.createData()
    z = np.zeros(1)
    pin.rnea(src.model, data, z, z, z)
    local = np.asarray(data.f[1].linear)
    assert not np.allclose(local, w.force_n, atol=1e-6)


def test_actuator_is_tau_times_world_axis_at_anchor() -> None:
    frame = _static(_source("1.5707963267948966 0 0"), tau=3.0)
    (w,) = frame.by_kind(WrenchKind.JOINT_ACTUATOR)
    # local axis y rotated +90 deg about x is world +z
    np.testing.assert_allclose(w.torque_nm, (0, 0, 3.0), atol=1e-9)
    assert w.force_n is None
    np.testing.assert_allclose(w.point_m, (0, 0, 1), atol=1e-12)
    assert w.body == "synthetic_rod"


def test_sample_leaves_engine_data_unchanged() -> None:
    model = pin.buildModelFromXML(_pendulum_urdf())
    engine_data = model.createData()
    q = np.array([0.3])
    pin.rnea(model, engine_data, q, np.zeros(1), np.zeros(1))
    oMi = [m.copy() for m in engine_data.oMi]
    f = [x.copy() for x in engine_data.f]
    src = PinocchioForceTorqueSource(model)
    src.sample(np.array([0.7]), np.array([1.0]), np.array([2.0]), np.zeros(1))
    for a, b in zip(oMi, engine_data.oMi, strict=True):
        np.testing.assert_array_equal(a.homogeneous, b.homogeneous)
    for a, b in zip(f, engine_data.f, strict=True):
        np.testing.assert_array_equal(a.vector, b.vector)


def _contact_sample() -> ContactSample:
    return ContactSample(
        0.001,
        0.0,
        np.array([0.1, 0.2, 0.0]),
        np.array([0.0, 0.0, 50.0]),
        np.array([1.0, 2.0, 0.0]),
    )


def test_contact_sample_passes_through_with_its_body() -> None:
    z = np.zeros(1)
    frame = _source().sample(
        z, z, z, z, contact_samples={"synthetic_rod": _contact_sample()}
    )
    (c,) = frame.by_kind(WrenchKind.CONTACT)
    assert c.body == "synthetic_rod"
    np.testing.assert_allclose(c.point_m, (0.1, 0.2, 0.0))
    np.testing.assert_allclose(c.force_n, (1.0, 2.0, 50.0))
    assert c.torque_nm is None


def test_contact_with_unknown_body_is_omitted_not_labelled_world() -> None:
    z = np.zeros(1)
    frame = _source().sample(
        z, z, z, z, contact_samples={"no_such_body": _contact_sample()}
    )
    assert frame.by_kind(WrenchKind.CONTACT) == ()


def test_contact_samples_must_be_a_mapping() -> None:
    z = np.zeros(1)
    with pytest.raises(TypeError, match="mapping"):
        _source().sample(z, z, z, z, contact_samples=(_contact_sample(),))


def test_uses_actual_acceleration_not_zero_torque() -> None:
    """Moving pendulum: reaction changes with a (guards the ZTCF bug)."""
    src = _source()
    z = np.zeros(1)
    still = src.sample(z, z, z, z).by_kind(WrenchKind.JOINT_REACTION)[0]
    acc = src.sample(z, z, np.array([4.0]), z).by_kind(WrenchKind.JOINT_REACTION)[0]
    assert not np.allclose(still.force_n, acc.force_n)


def test_size_validation() -> None:
    src = _source()
    z = np.zeros(1)
    with pytest.raises(ValueError, match="q"):
        src.sample(np.zeros(3), z, z, z)
    with pytest.raises(ValueError, match="tau_applied"):
        src.sample(z, z, z, np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        src.sample(np.array([np.nan]), z, z, z)


def test_free_flyer_joint_is_omitted() -> None:
    model = pin.buildModelFromXML(_pendulum_urdf(), pin.JointModelFreeFlyer())
    src = PinocchioForceTorqueSource(model)
    q = pin.neutral(model)
    z = np.zeros(model.nv)
    frame = src.sample(q, z, z, z)
    labels = [w.label for w in frame.by_kind(WrenchKind.JOINT_ACTUATOR)]
    assert "actuator:root_joint" not in labels
    assert all(w.body != "root_joint" for w in frame.by_kind(WrenchKind.JOINT_ACTUATOR))


def test_engine_adapter_exposes_provider() -> None:
    engine = PinocchioPhysicsEngine()
    engine.load_from_string(_pendulum_urdf(), "urdf")
    assert isinstance(engine, ForceTorqueProvider)
    engine.set_control(np.array([1.5]))
    frame = engine.get_force_torque_frame()
    assert frame is not None and frame.engine == "pinocchio"
    (act,) = frame.by_kind(WrenchKind.JOINT_ACTUATOR)
    np.testing.assert_allclose(act.torque_nm, (0, 1.5, 0), atol=1e-9)
    loads = engine.get_segment_axial_loads()
    assert loads is not None and loads.values_n["synthetic_rod"] > 0
    tau = engine.get_applied_torques()
    np.testing.assert_allclose(tau, [1.5])
    assert not tau.flags.writeable
    caps = engine.get_capabilities()
    assert caps.force_visualization.name == "FULL"


def test_engine_uninitialized_returns_none() -> None:
    engine = PinocchioPhysicsEngine()
    assert engine.get_force_torque_frame() is None
    assert engine.get_segment_axial_loads() is None


def test_engine_contact_forces_zero_without_model_and_sum_with() -> None:
    engine = PinocchioPhysicsEngine()
    engine.load_from_string(_pendulum_urdf(), "urdf")
    np.testing.assert_allclose(engine.compute_contact_forces(), [0, 0, 0])
    sample = ContactSample(
        0.0, 0.0, np.zeros(3), np.array([0.0, 0.0, 50.0]), np.array([1.0, 0.0, 0.0])
    )
    engine.set_contact_samples({"synthetic_rod": sample})
    np.testing.assert_allclose(engine.compute_contact_forces(), [1.0, 0.0, 50.0])
    frame = engine.get_force_torque_frame()
    assert frame is not None and len(frame.by_kind(WrenchKind.CONTACT)) == 1


def test_engine_recomputes_acceleration_instead_of_using_stale_a() -> None:
    engine = PinocchioPhysicsEngine()
    engine.load_from_string(_pendulum_urdf(), "urdf")
    engine.set_state(np.array([0.8]), np.array([0.0]))  # engine.a is still zero
    assert float(engine.a[0]) == 0.0
    frame = engine.get_force_torque_frame()
    assert frame is not None
    model = engine.model
    data = model.createData()
    a_true = pin.aba(model, data, np.array([0.8]), np.zeros(1), np.zeros(1))
    assert abs(float(a_true[0])) > 1.0
    pin.rnea(model, data, np.array([0.8]), np.zeros(1), a_true)
    pin.forwardKinematics(model, data, np.array([0.8]))
    expected = data.oMi[1].rotation @ np.asarray(data.f[1].linear)
    (w,) = frame.by_kind(WrenchKind.JOINT_REACTION)
    np.testing.assert_allclose(w.force_n, expected, atol=1e-9)


def test_source_acceleration_matches_aba() -> None:
    src = _source()
    q, v, tau = np.array([0.4]), np.array([0.3]), np.array([1.0])
    expected = pin.aba(src.model, src.model.createData(), q, v, tau)
    np.testing.assert_allclose(src.acceleration(q, v, tau), expected)


_TREE_URDF = """<robot name="synthetic_tree">
  <link name="synthetic_root"/>
  <link name="trunk"><inertial><origin xyz="0 0 0.1"/><mass value="1"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
  </inertial></link>
  <link name="left"><inertial><origin xyz="0 0 0.2"/><mass value="1"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
  </inertial></link>
  <link name="right"><inertial><origin xyz="0 0 0.2"/><mass value="1"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
  </inertial></link>
  <joint name="j_trunk" type="revolute">
    <parent link="synthetic_root"/><child link="trunk"/>
    <origin xyz="0 0 1"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="100" velocity="10"/>
  </joint>
  <joint name="j_left" type="revolute">
    <parent link="trunk"/><child link="left"/>
    <origin xyz="0.2 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="100" velocity="10"/>
  </joint>
  <joint name="j_right" type="revolute">
    <parent link="trunk"/><child link="right"/>
    <origin xyz="-0.2 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="100" velocity="10"/>
  </joint>
</robot>"""

_COINCIDENT_URDF = """<robot name="synthetic_chain">
  <link name="synthetic_root"/>
  <link name="a"><inertial><mass value="1"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
  </inertial></link>
  <link name="b"><inertial><mass value="1"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
  </inertial></link>
  <link name="c"><inertial><origin xyz="0 0 -0.2"/><mass value="1"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
  </inertial></link>
  <joint name="j_a" type="revolute">
    <parent link="synthetic_root"/><child link="a"/>
    <origin xyz="0 0 1"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="100" velocity="10"/>
  </joint>
  <joint name="j_b" type="revolute">
    <parent link="a"/><child link="b"/>
    <origin xyz="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-3" upper="3" effort="100" velocity="10"/>
  </joint>
  <joint name="j_c" type="revolute">
    <parent link="b"/><child link="c"/>
    <origin xyz="0 0 -0.4"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="100" velocity="10"/>
  </joint>
</robot>"""


def _axial(urdf: str):
    model = pin.buildModelFromXML(urdf)
    z = np.zeros(model.nv)
    frame = PinocchioForceTorqueSource(model).sample(pin.neutral(model), z, z, z)
    assert frame.axial_loads is not None
    return dict(frame.axial_loads.values_n)


def test_branching_body_has_no_axial_load_but_leaves_do() -> None:
    loads = _axial(_TREE_URDF)
    assert loads["trunk"] is None  # two children: axis is ambiguous
    assert loads["left"] is not None and loads["right"] is not None


def test_coincident_intermediate_joint_gives_no_axial_load() -> None:
    loads = _axial(_COINCIDENT_URDF)
    assert loads["a"] is None  # child joint coincides with the parent joint
    assert loads["b"] is not None
