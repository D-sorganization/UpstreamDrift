"""Tests for the OpenSim force/torque provider (FTO-15, #11300).

All models are programmatic ``synthetic_`` fixtures. The tests need the real
``opensim`` package and skip cleanly when it is unavailable. They never start
the Simbody visualizer.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

osim = pytest.importorskip("opensim", reason="OpenSim not installed")
if not hasattr(osim, "Model"):
    pytest.skip("real opensim is unavailable", allow_module_level=True)

from src.engines.physics_engines.opensim.python.opensim_force_torque import (  # noqa: E402
    R_ZUP_FROM_OPENSIM_GROUND,
    OpenSimForceTorqueSource,
)
from src.engines.physics_engines.opensim.python.opensim_physics_engine import (  # noqa: E402
    OpenSimPhysicsEngine,
)
from src.shared.python.engine_core.capabilities import CapabilityLevel  # noqa: E402
from src.shared.python.force_overlay import (  # noqa: E402
    ForceTorqueProvider,
    WrenchKind,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

MASS = 2.0
G = 9.80665
PIN_HEIGHT_M = 1.0
COM_DROP_M = 0.5
BALL_RADIUS_M = 0.1
BALL_MASS = 1.0
COM_OFFSET_X_M = 0.05

_HUNT_CROSSLEY_XML = """<HuntCrossleyForce name="hc">
<appliesForce>true</appliesForce>
<HuntCrossleyForce::ContactParametersSet name="contact_parameters"><objects>
<HuntCrossleyForce::ContactParameters>
<geometry>floor ball_cs</geometry>
<stiffness>1e6</stiffness><dissipation>1.0</dissipation>
<static_friction>0.8</static_friction><dynamic_friction>0.6</dynamic_friction>
<viscous_friction>0.01</viscous_friction>
</HuntCrossleyForce::ContactParameters></objects><groups/>
</HuntCrossleyForce::ContactParametersSet>
<transition_velocity>0.01</transition_velocity>
</HuntCrossleyForce>"""


def _pendulum(with_actuator: bool = False):
    """Hanging pendulum: PinJoint about +z at (0, PIN_HEIGHT_M, 0) from ground."""
    model = osim.Model()
    model.setName("synthetic_pendulum")
    body = osim.Body(
        "link",
        MASS,
        osim.Vec3(0, -COM_DROP_M, 0),
        osim.Inertia(0.01, 0.01, 0.01),
    )
    model.addBody(body)
    joint = osim.PinJoint(
        "pin",
        model.getGround(),
        osim.Vec3(0, PIN_HEIGHT_M, 0),
        osim.Vec3(0, 0, 0),
        body,
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0),
    )
    model.addJoint(joint)
    if with_actuator:
        actuator = osim.CoordinateActuator("pin_coord")
        actuator.setName("pin_motor")
        model.addForce(actuator)
        actuator.setCoordinate(joint.updCoordinate())
        actuator.setOptimalForce(1.0)
    state = model.initSystem()
    return model, state


def _slider_with_actuator():
    model = osim.Model()
    model.setName("synthetic_slider")
    body = osim.Body("block", 1.0, osim.Vec3(0, 0, 0), osim.Inertia(0.01, 0.01, 0.01))
    model.addBody(body)
    joint = osim.SliderJoint(
        "slide",
        model.getGround(),
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0),
        body,
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0),
    )
    model.addJoint(joint)
    actuator = osim.CoordinateActuator("slide_coord")
    actuator.setName("slide_motor")
    model.addForce(actuator)
    actuator.setCoordinate(joint.updCoordinate())
    actuator.setOptimalForce(1.0)
    return model, model.initSystem()


def _ball_model_xml(
    directory: Path, com_x: float = 0.0, with_contact: bool = True
) -> Path:
    """Write a free ball + half-space model with a HuntCrossleyForce.

    The contact sphere is centred on the body origin; ``com_x`` offsets the mass
    centre in x so the contact point is horizontally displaced from the origin.
    """
    model = osim.Model()
    model.setName("synthetic_ball")
    body = osim.Body(
        "ball",
        BALL_MASS,
        osim.Vec3(com_x, 0, 0),
        osim.Inertia(0.01, 0.01, 0.01),
    )
    model.addBody(body)
    joint = osim.FreeJoint(
        "free",
        model.getGround(),
        osim.Vec3(0),
        osim.Vec3(0),
        body,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    model.addJoint(joint)
    floor = osim.ContactHalfSpace(
        osim.Vec3(0), osim.Vec3(0, 0, -np.pi / 2), model.getGround(), "floor"
    )
    model.addContactGeometry(floor)
    sphere = osim.ContactSphere(BALL_RADIUS_M, osim.Vec3(com_x, 0, 0), body, "ball_cs")
    model.addContactGeometry(sphere)
    model.finalizeConnections()
    path = directory / "synthetic_ball.osim"
    model.printToXML(str(path))
    if with_contact:
        text = path.read_text()
        text = re.sub(
            r'(<ForceSet name="forceset">\s*<objects)\s*/>',
            lambda m: m.group(1) + ">" + _HUNT_CROSSLEY_XML + "</objects>",
            text,
        )
        path.write_text(text)
    return path


def _settled_ball(directory: Path, com_x: float = 0.0, settle_s: float = 1.0):
    model = osim.Model(str(_ball_model_xml(directory, com_x)))
    state = model.initSystem()
    joint = model.getJointSet().get(0)
    # free_coord_4 is the vertical (Y-up) translation: start just touching.
    joint.get_coordinates(4).setValue(state, BALL_RADIUS_M)
    manager = osim.Manager(model)
    state.setTime(0.0)
    manager.initialize(state)
    # The manager owns the integrated state; copy it so it outlives the manager.
    return model, osim.State(manager.integrate(settle_s))


_CUSTOM_JOINT_OSIM = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
  <Model name="synthetic_custom">
    <BodySet><objects>
      <Body name="rod"><mass>2</mass><mass_center>0 -0.5 0</mass_center>
        <inertia>0.01 0.01 0.01 0 0 0</inertia></Body>
    </objects></BodySet>
    <JointSet><objects>
      <CustomJoint name="cj">
        <socket_parent_frame>parent_offset</socket_parent_frame>
        <socket_child_frame>child_offset</socket_child_frame>
        <frames>
          <PhysicalOffsetFrame name="parent_offset"><socket_parent>/ground</socket_parent>
            <translation>0 1 0</translation><orientation>0 0 0</orientation>
          </PhysicalOffsetFrame>
          <PhysicalOffsetFrame name="child_offset"><socket_parent>/bodyset/rod</socket_parent>
            <translation>0 0 0</translation><orientation>0 0 0</orientation>
          </PhysicalOffsetFrame>
        </frames>
        <coordinates><Coordinate name="q0"><motion_type>rotational</motion_type></Coordinate></coordinates>
        <SpatialTransform>
          <TransformAxis name="rotation1"><coordinates>q0</coordinates><axis>0 0 1</axis>
            <function><LinearFunction><coefficients>1 0</coefficients></LinearFunction></function></TransformAxis>
          <TransformAxis name="rotation2"><axis>0 1 0</axis><function><Constant value="0"/></function></TransformAxis>
          <TransformAxis name="rotation3"><axis>1 0 0</axis><function><Constant value="0"/></function></TransformAxis>
          <TransformAxis name="translation1"><axis>1 0 0</axis><function><Constant value="0"/></function></TransformAxis>
          <TransformAxis name="translation2"><axis>0 1 0</axis><function><Constant value="0"/></function></TransformAxis>
          <TransformAxis name="translation3"><axis>0 0 1</axis><function><Constant value="0"/></function></TransformAxis>
        </SpatialTransform>
      </CustomJoint>
    </objects></JointSet>
    <ForceSet><objects>
      <CoordinateActuator name="motor"><coordinate>q0</coordinate><optimal_force>1</optimal_force></CoordinateActuator>
    </objects></ForceSet>
  </Model>
</OpenSimDocument>
"""


def _contact_total(frame) -> np.ndarray:
    return np.sum([w.force_n for w in frame.by_kind(WrenchKind.CONTACT)], axis=0)


# --- frame constant ---------------------------------------------------------


def test_zup_constant_maps_opensim_up_to_world_up_and_is_a_rotation() -> None:
    np.testing.assert_allclose(R_ZUP_FROM_OPENSIM_GROUND @ [0, 1, 0], [0, 0, 1])
    assert np.linalg.det(R_ZUP_FROM_OPENSIM_GROUND) == pytest.approx(1.0)
    np.testing.assert_allclose(
        R_ZUP_FROM_OPENSIM_GROUND @ R_ZUP_FROM_OPENSIM_GROUND.T, np.eye(3), atol=1e-15
    )
    assert not R_ZUP_FROM_OPENSIM_GROUND.flags.writeable


# --- joint reactions and axial loads ---------------------------------------


def test_hanging_pendulum_reaction_is_weight_up_at_the_pin() -> None:
    model, state = _pendulum()
    frame = OpenSimForceTorqueSource(model).sample(state)
    (w,) = frame.by_kind(WrenchKind.JOINT_REACTION)
    assert w.label == "reaction:pin"
    assert w.body == "link"
    np.testing.assert_allclose(w.force_n, (0, 0, MASS * G), atol=1e-6)
    np.testing.assert_allclose(w.torque_nm, (0, 0, 0), atol=1e-6)
    # Y-up pin height becomes Z-up height.
    np.testing.assert_allclose(w.point_m, (0, 0, PIN_HEIGHT_M), atol=1e-9)
    assert frame.engine == "opensim"
    assert frame.world_frame == "world_Zup"


def test_hanging_pendulum_axial_load_is_positive_tension() -> None:
    model, state = _pendulum()
    frame = OpenSimForceTorqueSource(model).sample(state)
    assert frame.axial_loads is not None
    assert frame.axial_loads.values_n["link"] == pytest.approx(MASS * G, rel=1e-9)


def test_sample_realizes_the_state_itself() -> None:
    """A state only realized to Position still yields a valid reaction."""
    model, state = _pendulum()
    state.setTime(0.0)
    frame = OpenSimForceTorqueSource(model).sample(state)
    (w,) = frame.by_kind(WrenchKind.JOINT_REACTION)
    assert w.force_n is not None and w.force_n[2] == pytest.approx(MASS * G, rel=1e-6)


def test_frame_time_follows_state_time() -> None:
    model, state = _pendulum()
    state.setTime(0.25)
    assert OpenSimForceTorqueSource(model).sample(state).time_s == pytest.approx(0.25)


def test_no_grip_wrenches_are_emitted() -> None:
    model, state = _pendulum()
    frame = OpenSimForceTorqueSource(model).sample(state)
    assert frame.by_kind(WrenchKind.GRIP) == ()
    assert not any(w.label.startswith("grip:") for w in frame.wrenches)


# --- actuators --------------------------------------------------------------


def test_coordinate_actuator_is_a_moment_about_the_pin_axis() -> None:
    model, state = _pendulum(with_actuator=True)
    actuator = osim.CoordinateActuator.safeDownCast(model.getForceSet().get(0))
    model.realizeDynamics(state)
    actuator.overrideActuation(state, True)
    actuator.setOverrideActuation(state, 5.0)
    frame = OpenSimForceTorqueSource(model).sample(state)
    (w,) = frame.by_kind(WrenchKind.JOINT_ACTUATOR)
    assert w.label == "actuator:pin.pin_coord_0"
    assert w.body == "link"
    assert w.force_n is None
    # Pin axis is OpenSim +z, which is world -y after the Y-up to Z-up rotation.
    np.testing.assert_allclose(w.torque_nm, (0, -5.0, 0), atol=1e-9)
    np.testing.assert_allclose(w.point_m, (0, 0, PIN_HEIGHT_M), atol=1e-9)


def test_actuator_axis_follows_the_child_frame() -> None:
    model, state = _pendulum(with_actuator=True)
    actuator = osim.CoordinateActuator.safeDownCast(model.getForceSet().get(0))
    model.getCoordinateSet().get(0).setValue(state, 0.7)
    model.realizeDynamics(state)
    actuator.overrideActuation(state, True)
    actuator.setOverrideActuation(state, -2.0)
    (w,) = (
        OpenSimForceTorqueSource(model).sample(state).by_kind(WrenchKind.JOINT_ACTUATOR)
    )
    np.testing.assert_allclose(w.torque_nm, (0, 2.0, 0), atol=1e-9)


def test_translational_coordinate_actuator_is_omitted_not_zero() -> None:
    model, state = _slider_with_actuator()
    frame = OpenSimForceTorqueSource(model).sample(state)
    assert frame.by_kind(WrenchKind.JOINT_ACTUATOR) == ()


def test_custom_joint_rotational_actuator_uses_its_transform_axis(
    tmp_path: Path,
) -> None:
    path = tmp_path / "synthetic_custom.osim"
    path.write_text(_CUSTOM_JOINT_OSIM)
    model = osim.Model(str(path))
    state = model.initSystem()
    actuator = osim.CoordinateActuator.safeDownCast(model.getForceSet().get(0))
    model.realizeDynamics(state)
    actuator.overrideActuation(state, True)
    actuator.setOverrideActuation(state, 3.0)
    frame = OpenSimForceTorqueSource(model).sample(state)
    (w,) = frame.by_kind(WrenchKind.JOINT_ACTUATOR)
    assert w.label == "actuator:cj.q0"
    np.testing.assert_allclose(w.torque_nm, (0, -3.0, 0), atol=1e-9)


# --- contacts ---------------------------------------------------------------


def test_model_without_contacts_has_no_contact_wrenches() -> None:
    model, state = _pendulum()
    frame = OpenSimForceTorqueSource(model).sample(state)
    assert frame.by_kind(WrenchKind.CONTACT) == ()


def test_resting_sphere_contact_is_weight_at_the_sphere_bottom(
    tmp_path: Path,
) -> None:
    model, state = _settled_ball(tmp_path)
    frame = OpenSimForceTorqueSource(model).sample(state)
    (c,) = frame.by_kind(WrenchKind.CONTACT)
    assert c.label == "contact:hc"
    assert c.body == "ball"
    total = _contact_total(frame)
    np.testing.assert_allclose(total, (0, 0, BALL_MASS * G), rtol=1e-2, atol=1e-2)
    centre = model.getBodySet().get("ball").getPositionInGround(state)
    bottom_zup = R_ZUP_FROM_OPENSIM_GROUND @ [
        centre.get(0),
        centre.get(1) - BALL_RADIUS_M,
        centre.get(2),
    ]
    np.testing.assert_allclose(c.point_m, bottom_zup, atol=2e-3)
    assert abs(c.point_m[2]) < 2e-3  # the bottom point sits on the floor plane


def test_contact_torque_is_about_the_declared_point_not_the_body_origin(
    tmp_path: Path,
) -> None:
    """Offset mass centre puts the contact point away from the body origin.

    The OpenSim record torque is about the body origin; the wrench torque must
    be moved to the contact point, where a quasi-static vertical load has none.
    """
    model, state = _settled_ball(tmp_path, com_x=COM_OFFSET_X_M)
    (c,) = OpenSimForceTorqueSource(model).sample(state).by_kind(WrenchKind.CONTACT)
    assert c.force_n is not None and c.torque_nm is not None
    assert c.force_n[2] > 0.5 * BALL_MASS * G
    assert np.linalg.norm(c.torque_nm) < 5e-3 * c.force_n[2]
    # The raw record torque about the body origin is not small: it is the
    # moment of the contact force about the origin, so the move is observable.
    force = model.getForceSet().get(0)
    values = force.getRecordValues(state)
    raw_origin_torque = np.array([values.get(9 + i) for i in range(3)])
    assert np.linalg.norm(raw_origin_torque) > 1e-2


# --- validation -------------------------------------------------------------


def test_constructor_requires_a_model() -> None:
    with pytest.raises(TypeError, match="Model"):
        OpenSimForceTorqueSource(object())  # type: ignore[arg-type]


def test_sample_requires_a_state() -> None:
    model, _ = _pendulum()
    with pytest.raises(TypeError, match="State"):
        OpenSimForceTorqueSource(model).sample(object())  # type: ignore[arg-type]


# --- engine adapter ---------------------------------------------------------


def _engine_with(path: Path) -> OpenSimPhysicsEngine:
    engine = OpenSimPhysicsEngine()
    engine.load_from_path(str(path))
    return engine


def test_engine_exposes_provider_axial_loads_and_capabilities(tmp_path: Path) -> None:
    model, _ = _pendulum()
    path = tmp_path / "synthetic_pendulum.osim"
    model.printToXML(str(path))
    engine = _engine_with(path)
    assert isinstance(engine, ForceTorqueProvider)
    frame = engine.get_force_torque_frame()
    assert frame is not None and frame.engine == "opensim"
    (w,) = frame.by_kind(WrenchKind.JOINT_REACTION)
    np.testing.assert_allclose(w.force_n, (0, 0, MASS * G), atol=1e-6)
    loads = engine.get_segment_axial_loads()
    assert loads is not None and loads.values_n["link"] > 0
    caps = engine.get_capabilities()
    assert caps.force_visualization == CapabilityLevel.FULL
    assert caps.contact_forces == CapabilityLevel.FULL


def test_engine_without_model_returns_none() -> None:
    engine = OpenSimPhysicsEngine()
    assert engine.get_force_torque_frame() is None
    assert engine.get_segment_axial_loads() is None


def test_engine_contact_forces_are_the_world_summed_contact(tmp_path: Path) -> None:
    engine = _engine_with(_ball_model_xml(tmp_path))
    # 1 mm penetration along the Y-up vertical coordinate (free_coord_4).
    coordinate = engine._model.getCoordinateSet().get("free_coord_4")
    coordinate.setValue(engine._state, BALL_RADIUS_M - 1e-3)
    total = engine.compute_contact_forces()
    frame = engine.get_force_torque_frame()
    assert frame is not None
    np.testing.assert_allclose(total, _contact_total(frame), atol=1e-9)
    assert total[2] > 1.0  # pushes up in world Z


def test_engine_contact_forces_zero_without_contact_model(tmp_path: Path) -> None:
    model, _ = _pendulum()
    path = tmp_path / "synthetic_pendulum.osim"
    model.printToXML(str(path))
    np.testing.assert_allclose(_engine_with(path).compute_contact_forces(), (0, 0, 0))
