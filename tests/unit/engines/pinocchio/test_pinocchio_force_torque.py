"""Tests for the Pinocchio force/torque provider (FTO-13, #11298).

Uses a real Pinocchio model built from an inline ``synthetic_`` URDF. Skipped
when Pinocchio is missing or when only the stub PyPI wheel is installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
if not hasattr(pin, "buildModelFromXML"):  # stub PyPI wheel
    pytest.skip(
        "real Pinocchio required (stub wheel installed)", allow_module_level=True
    )

from src.engines.physics_engines.pinocchio.python.pinocchio_force_torque import (  # noqa: E402
    PinocchioForceTorqueSource,
)
from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (  # noqa: E402
    PinocchioPhysicsEngine,
)
from src.shared.python.engine_core.capabilities import CapabilityLevel  # noqa: E402
from src.shared.python.force_overlay import (  # noqa: E402
    ForceTorqueFrame,
    ForceTorqueProvider,
    WrenchKind,
)
from src.shared.python.motion_matching.contact_law import ContactSample  # noqa: E402

pytestmark = pytest.mark.unit

MASS = 2.0
G = 9.81
LENGTH = 0.5


def _urdf(base_rpy: str = "0 0 0", axis: str = "0 1 0") -> str:
    return f"""<robot name="synthetic_pendulum">
  <link name="world"/>
  <link name="base"/>
  <link name="rod">
    <inertial>
      <origin xyz="0 0 -{LENGTH}"/>
      <mass value="{MASS}"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="mount" type="fixed">
    <parent link="world"/><child link="base"/><origin rpy="{base_rpy}"/>
  </joint>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="rod"/><axis xyz="{axis}"/>
    <limit lower="-3" upper="3" effort="10" velocity="10"/>
  </joint>
</robot>"""


def _model(**kwargs: str):
    return pin.buildModelFromXML(_urdf(**kwargs))


def _zeros(model):
    return np.zeros(model.nq), np.zeros(model.nv), np.zeros(model.nv)


def test_hanging_pendulum_reaction_is_world_weight_at_joint_origin() -> None:
    model = _model()
    q, v, a = _zeros(model)
    source = PinocchioForceTorqueSource(model)
    frame = source.sample(q, v, a, np.zeros(model.nv))
    (reaction,) = frame.by_kind(WrenchKind.JOINT_REACTION)
    assert reaction.force_n == pytest.approx((0.0, 0.0, MASS * G), abs=1e-9)
    assert reaction.point_m == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert frame.axial_loads is not None
    (axial,) = frame.axial_loads.values_n.values()
    assert axial == pytest.approx(MASS * G, abs=1e-9)  # tension positive


def test_reaction_stays_in_world_frame_when_base_rotated() -> None:
    model = _model(base_rpy=f"{math.pi / 2} 0 0")
    q, v, a = _zeros(model)
    data = model.createData()
    pin.rnea(model, data, q, v, a)
    local_force = np.array(data.f[1].linear)
    assert abs(local_force[2] - MASS * G) > 1.0  # old local-as-world bug differs
    frame = PinocchioForceTorqueSource(model).sample(q, v, a, np.zeros(model.nv))
    (reaction,) = frame.by_kind(WrenchKind.JOINT_REACTION)
    assert reaction.force_n == pytest.approx((0.0, 0.0, MASS * G), abs=1e-9)


def test_uses_actual_acceleration_not_zero_torque() -> None:
    model = _model()
    q, v, _ = _zeros(model)
    a = np.array([2.0])
    source = PinocchioForceTorqueSource(model)
    data = model.createData()
    pin.rnea(model, data, q, v, a)
    expected = np.asarray(data.f[1].angular)
    frame = source.sample(q, v, a, np.zeros(model.nv))
    (reaction,) = frame.by_kind(WrenchKind.JOINT_REACTION)
    assert reaction.torque_nm == pytest.approx(tuple(expected), abs=1e-9)
    assert abs(expected[1]) > 1e-3


def test_actuator_torque_is_tau_times_world_axis_at_anchor() -> None:
    model = _model(base_rpy=f"{math.pi / 2} 0 0")
    q, v, a = _zeros(model)
    tau = np.array([3.0])
    frame = PinocchioForceTorqueSource(model).sample(q, v, a, tau)
    (actuator,) = frame.by_kind(WrenchKind.JOINT_ACTUATOR)
    # joint axis local +y; base rotated +90 deg about x maps y -> z.
    assert actuator.torque_nm == pytest.approx((0.0, 0.0, 3.0), abs=1e-6)
    assert actuator.force_n is None
    assert actuator.point_m == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


def test_unaligned_axis_actuator_uses_normalised_world_axis() -> None:
    model = _model(axis="1 1 0")
    q, v, a = _zeros(model)
    frame = PinocchioForceTorqueSource(model).sample(q, v, a, np.array([2.0]))
    (actuator,) = frame.by_kind(WrenchKind.JOINT_ACTUATOR)
    s = 2.0 / math.sqrt(2.0)
    assert actuator.torque_nm == pytest.approx((s, s, 0.0), abs=1e-9)


def test_spherical_joint_moment_is_rotated_tau() -> None:
    model = pin.Model()
    inertia = pin.Inertia(1.0, np.array([0.0, 0.0, -0.3]), np.eye(3) * 0.01)
    model.addJoint(0, pin.JointModelSpherical(), pin.SE3.Identity(), "ball")
    model.appendBodyToJoint(1, inertia, pin.SE3.Identity())
    q = np.asarray(pin.neutral(model))
    z = np.zeros(model.nv)
    tau = np.array([1.0, 2.0, 3.0])
    frame = PinocchioForceTorqueSource(model).sample(q, z, z, tau)
    (actuator,) = frame.by_kind(WrenchKind.JOINT_ACTUATOR)
    assert actuator.torque_nm == pytest.approx((1.0, 2.0, 3.0), abs=1e-9)


def test_sample_does_not_mutate_engine_data() -> None:
    model = _model()
    engine_data = model.createData()
    q, v, a = _zeros(model)
    pin.forwardKinematics(model, engine_data, q, v, a)
    before_oMi = engine_data.oMi[1].homogeneous.copy()
    before_f = np.array(engine_data.f[1].vector)
    source = PinocchioForceTorqueSource(model)
    source.sample(q + 0.3, v + 0.1, a + 0.2, np.zeros(model.nv))
    assert np.array_equal(engine_data.oMi[1].homogeneous, before_oMi)
    assert np.array_equal(np.array(engine_data.f[1].vector), before_f)


def test_contact_sample_passes_through_as_contact_wrench() -> None:
    model = _model()
    q, v, a = _zeros(model)
    point = np.array([0.1, 0.2, 0.0])
    sample = ContactSample(
        0.01, 0.0, point, np.array([0.0, 0.0, 50.0]), np.array([1.0, 2.0, 0.0])
    )
    frame = PinocchioForceTorqueSource(model).sample(
        q, v, a, np.zeros(model.nv), contact_samples={"left_foot": sample}
    )
    (contact,) = frame.by_kind(WrenchKind.CONTACT)
    assert contact.label == "contact:left_foot"
    assert contact.point_m == pytest.approx(tuple(point))
    assert contact.force_n == pytest.approx((1.0, 2.0, 50.0))
    assert contact.torque_nm is None


@pytest.mark.parametrize("bad", ["q", "v", "a", "tau"])
def test_size_mismatch_raises_value_error(bad: str) -> None:
    model = _model()
    q, v, a = _zeros(model)
    args = {"q": q, "v": v, "a": a, "tau": np.zeros(model.nv)}
    args[bad] = np.zeros(5)
    with pytest.raises(ValueError, match=bad):
        PinocchioForceTorqueSource(model).sample(
            args["q"], args["v"], args["a"], args["tau"]
        )


def test_non_finite_input_raises() -> None:
    model = _model()
    q, v, a = _zeros(model)
    q[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        PinocchioForceTorqueSource(model).sample(q, v, a, np.zeros(model.nv))


def test_free_flyer_has_no_actuator_wrench() -> None:
    model = pin.buildModelFromXML(_urdf(), pin.JointModelFreeFlyer())
    q = np.asarray(pin.neutral(model))
    v, a = np.zeros(model.nv), np.zeros(model.nv)
    frame = PinocchioForceTorqueSource(model).sample(q, v, a, np.zeros(model.nv))
    actuators = frame.by_kind(WrenchKind.JOINT_ACTUATOR)
    assert [w.label for w in actuators] == ["joint_actuator:hinge"]


def _loaded_engine() -> PinocchioPhysicsEngine:
    engine = PinocchioPhysicsEngine()
    engine.load_from_string(_urdf(), "urdf")
    return engine


def test_engine_is_provider_with_full_capability() -> None:
    engine = _loaded_engine()
    assert isinstance(engine, ForceTorqueProvider)
    caps = engine.get_capabilities()
    assert caps.force_visualization == CapabilityLevel.FULL


def test_engine_frame_uses_current_state_and_tau() -> None:
    engine = _loaded_engine()
    engine.set_control(np.array([1.5]))
    frame = engine.get_force_torque_frame()
    assert isinstance(frame, ForceTorqueFrame)
    assert frame.engine == "pinocchio"
    (actuator,) = frame.by_kind(WrenchKind.JOINT_ACTUATOR)
    assert actuator.torque_nm == pytest.approx((0.0, 1.5, 0.0), abs=1e-9)
    (reaction,) = frame.by_kind(WrenchKind.JOINT_REACTION)
    assert reaction.force_n == pytest.approx((0.0, 0.0, MASS * G), abs=1e-9)
    loads = engine.get_segment_axial_loads()
    assert loads is not None
    assert next(iter(loads.values_n.values())) == pytest.approx(MASS * G)


def test_engine_applied_torques_read_only_copy() -> None:
    engine = _loaded_engine()
    engine.set_control(np.array([2.0]))
    tau = engine.get_applied_torques()
    with pytest.raises(ValueError):
        tau[0] = 99.0
    assert engine.tau[0] == 2.0


def test_engine_unloaded_returns_none() -> None:
    engine = PinocchioPhysicsEngine()
    assert engine.get_force_torque_frame() is None
    assert engine.get_segment_axial_loads() is None


def test_contact_forces_zero_without_contact_model_and_summed_with_one() -> None:
    engine = _loaded_engine()
    assert np.allclose(engine.compute_contact_forces(), [0.0, 0.0, 0.0])
    sample = ContactSample(
        0.01,
        0.0,
        np.zeros(3),
        np.array([0.0, 0.0, 40.0]),
        np.array([3.0, 0.0, 0.0]),
    )
    engine.set_contact_provider(lambda: {"foot": sample})
    assert np.allclose(engine.compute_contact_forces(), [3.0, 0.0, 40.0])
    frame = engine.get_force_torque_frame()
    assert frame is not None
    assert len(frame.by_kind(WrenchKind.CONTACT)) == 1
    engine.set_contact_provider(None)
    assert np.allclose(engine.compute_contact_forces(), [0.0, 0.0, 0.0])
