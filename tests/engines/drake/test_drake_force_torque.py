"""Drake force/torque provider tests (FTO-11, #11296).

Models are synthetic, built inline; no GUI or MeshCat is started.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("pydrake.multibody.plant")  # skips if pydrake is mocked

from pydrake.math import RigidTransform  # noqa: E402
from pydrake.multibody.parsing import Parser  # noqa: E402
from pydrake.multibody.plant import (  # noqa: E402
    AddMultibodyPlantSceneGraph,
    ContactModel,
)
from pydrake.systems.analysis import Simulator  # noqa: E402
from pydrake.systems.framework import DiagramBuilder  # noqa: E402

from src.engines.physics_engines.drake.python.drake_force_torque import (  # noqa: E402
    DrakeForceTorqueSource,
)
from src.shared.python.force_overlay import WrenchKind  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_drake]

G = 9.81
ROD_MASS = 2.0
TIP_MASS = 0.01
TOTAL = ROD_MASS + TIP_MASS

SYNTHETIC_PENDULUM_URDF = f"""<robot name="synthetic_pendulum">
<link name="base"><inertial><mass value="1"/>
<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="rod"><inertial><origin xyz="0 0 -1"/><mass value="{ROD_MASS}"/>
<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="tip"><inertial><mass value="{TIP_MASS}"/>
<inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="shoulder" type="revolute"><parent link="base"/><child link="rod"/>
<axis xyz="0 1 0"/><limit lower="-4" upper="4" effort="100" velocity="10"/></joint>
<joint name="tip_weld" type="fixed"><parent link="rod"/><child link="tip"/>
<origin xyz="0 0 -1"/></joint>
<transmission name="t"><type>transmission_interface/SimpleTransmission</type>
<joint name="shoulder"><hardwareInterface>EffortJointInterface</hardwareInterface></joint>
<actuator name="m"><mechanicalReduction>1</mechanicalReduction></actuator>
</transmission></robot>"""

_PROX = (
    '<drake:proximity_properties xmlns:drake="http://drake.mit.edu">'
    "<drake:compliant_hydroelastic/>"
    '<drake:hydroelastic_modulus value="1e6"/>'
    '<drake:mesh_resolution_hint value="0.05"/></drake:proximity_properties>'
)
SYNTHETIC_BOX_URDF = """<robot name="synthetic_box">
<link name="box"><inertial><mass value="2"/>
<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
<collision><geometry><box size="0.2 0.2 0.2"/></geometry>{prox}</collision></link>
</robot>"""
SYNTHETIC_BALL_URDF = """<robot name="synthetic_ball">
<link name="ball"><inertial><mass value="2"/>
<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
<collision><geometry><sphere radius="0.1"/></geometry></collision></link>
</robot>"""
_GROUND_PROX = "<drake:proximity_properties><drake:rigid_hydroelastic/></drake:proximity_properties>"
SYNTHETIC_GROUND_SDF = """<?xml version="1.0"?>
<sdf version="1.7" xmlns:drake="http://drake.mit.edu"><model name="synthetic_ground">
<static>true</static><link name="g"><collision name="c">
<pose>0 0 -0.5 0 0 0</pose><geometry><box><size>5 5 1</size></box></geometry>
{prox}</collision></link></model></sdf>"""


def _pendulum(time_step: float = 0.0, actuated_input: float | None = None):
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step)
    Parser(plant).AddModelsFromString(SYNTHETIC_PENDULUM_URDF, "urdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"))
    plant.Finalize()
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    pctx = plant.GetMyContextFromRoot(ctx)
    if actuated_input is not None:
        plant.get_actuation_input_port().FixValue(pctx, [actuated_input])
    return plant, diagram, pctx


def _resting(urdf: str, model: str, hydro: bool):
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, 1e-3)
    parser = Parser(plant)
    parser.AddModelsFromString(urdf.replace("{prox}", _PROX if hydro else ""), "urdf")
    ground = SYNTHETIC_GROUND_SDF.replace("{prox}", _GROUND_PROX if hydro else "")
    parser.AddModelsFromString(ground, "sdf")
    plant.set_contact_model(
        ContactModel.kHydroelasticWithFallback if hydro else ContactModel.kPoint
    )
    plant.Finalize()
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    pctx = plant.GetMyContextFromRoot(ctx)
    plant.SetFreeBodyPose(pctx, plant.GetBodyByName(model), RigidTransform([0, 0, 0.1]))
    Simulator(diagram, ctx).AdvanceTo(1.0)
    return plant, diagram, pctx


def test_hanging_pendulum_reaction_and_tension() -> None:
    plant, diagram, pctx = _pendulum()
    frame = DrakeForceTorqueSource(plant, diagram).sample(pctx)
    by_label = {w.label: w for w in frame.by_kind(WrenchKind.JOINT_REACTION)}
    shoulder = by_label["joint_reaction:shoulder"]
    assert shoulder.body == "rod"
    np.testing.assert_allclose(shoulder.force_n, (0, 0, (TOTAL) * G), atol=1e-9)
    np.testing.assert_allclose(shoulder.point_m, (0, 0, 0), atol=1e-12)
    assert frame.axial_loads is not None
    assert frame.axial_loads.values_n["rod"] == pytest.approx(TOTAL * G)
    assert frame.axial_loads.values_n["rod"] > 0.0


def test_inverted_pendulum_actuator_torque_and_compression() -> None:
    theta = 0.3
    plant, diagram, pctx = _pendulum(actuated_input=TOTAL * G * math.sin(theta))
    plant.GetJointByName("shoulder").set_angle(pctx, math.pi - theta)
    frame = DrakeForceTorqueSource(plant, diagram).sample(pctx)
    (act,) = frame.by_kind(WrenchKind.JOINT_ACTUATOR)
    assert act.label == "joint_actuator:shoulder"
    np.testing.assert_allclose(
        act.torque_nm, (0, TOTAL * G * math.sin(theta), 0), atol=1e-9
    )
    assert act.force_n is None
    assert frame.axial_loads is not None
    assert frame.axial_loads.values_n["rod"] == pytest.approx(
        -TOTAL * G * math.cos(theta), rel=1e-6
    )


def test_unconnected_actuation_is_unavailable_not_zero() -> None:
    plant, diagram, pctx = _pendulum()
    source = DrakeForceTorqueSource(plant, diagram)
    frame = source.sample(pctx)
    assert frame.by_kind(WrenchKind.JOINT_ACTUATOR) == ()
    assert "joint_actuator:shoulder" in source.unavailable_labels


def test_actuation_available_clears_unavailable() -> None:
    plant, diagram, pctx = _pendulum(actuated_input=1.0)
    source = DrakeForceTorqueSource(plant, diagram)
    source.sample(pctx)
    assert source.unavailable_labels == ()


@pytest.mark.parametrize("hydro", [False, True])
def test_resting_contact_sums_to_weight(hydro: bool) -> None:
    urdf, body = (SYNTHETIC_BOX_URDF, "box") if hydro else (SYNTHETIC_BALL_URDF, "ball")
    plant, diagram, pctx = _resting(urdf, body, hydro)
    source = DrakeForceTorqueSource(plant, diagram)
    frame = source.sample(pctx)
    contacts = [w for w in frame.by_kind(WrenchKind.CONTACT) if w.body == body]
    assert contacts
    total = np.sum([w.force_n for w in contacts], axis=0)
    np.testing.assert_allclose(total, (0, 0, 2.0 * G), rtol=0.02, atol=0.05)
    assert all(w.label.startswith(f"contact:{body}:") for w in contacts)
    if hydro:
        # Centroid of the contact patch sits on the ground plane, under the box.
        assert all(abs(w.point_m[2]) < 0.02 for w in contacts)
    else:
        assert all(abs(w.point_m[2]) < 0.01 for w in contacts)
    # The ground (world) body never receives a wrench.
    assert all(w.body != "world" for w in frame.wrenches)


def test_gravity_opt_in() -> None:
    plant, diagram, pctx = _pendulum()
    source = DrakeForceTorqueSource(plant, diagram)
    assert source.sample(pctx).by_kind(WrenchKind.GRAVITY) == ()
    frame = source.sample(pctx, include_gravity=True)
    rod = next(w for w in frame.by_kind(WrenchKind.GRAVITY) if w.body == "rod")
    np.testing.assert_allclose(rod.force_n, (0, 0, -ROD_MASS * G), atol=1e-9)
    np.testing.assert_allclose(rod.point_m, (0, 0, -1), atol=1e-9)
    assert rod.torque_nm is None


def test_foreign_context_rejected() -> None:
    plant, diagram, _ = _pendulum()
    _, _, other_ctx = _pendulum()
    with pytest.raises(ValueError, match="plant"):
        DrakeForceTorqueSource(plant, diagram).sample(other_ctx)
    with pytest.raises(TypeError):
        DrakeForceTorqueSource(plant, diagram).sample(object())  # type: ignore[arg-type]


def test_constructor_preconditions() -> None:
    with pytest.raises(TypeError):
        DrakeForceTorqueSource(object())  # type: ignore[arg-type]
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, 0.0)
    with pytest.raises(ValueError, match="finalized"):
        DrakeForceTorqueSource(plant)


def test_engine_wiring_capabilities_and_frames(tmp_path) -> None:
    from src.engines.physics_engines.drake.python.drake_physics_engine import (
        DrakePhysicsEngine,
    )
    from src.shared.python.body_part_viz.axial_loads import AxialLoadProvider
    from src.shared.python.engine_core.capabilities import CapabilityLevel
    from src.shared.python.force_overlay import ForceTorqueProvider

    urdf = tmp_path / "synthetic_pendulum.urdf"
    # A link named "world" is welded to the world frame by Drake's URDF parser.
    urdf.write_text(
        SYNTHETIC_PENDULUM_URDF.replace('name="base"', 'name="world"').replace(
            'link="base"', 'link="world"'
        )
    )
    engine = DrakePhysicsEngine()
    assert engine.get_force_torque_frame() is None
    assert engine.get_segment_axial_loads() is None
    engine.load_from_path(str(urdf))
    assert isinstance(engine, ForceTorqueProvider)
    assert isinstance(engine, AxialLoadProvider)
    caps = engine.get_capabilities()
    assert caps.force_visualization == CapabilityLevel.FULL
    assert caps.contact_forces == CapabilityLevel.FULL
    # Discrete-time plants report reactions only once the simulator has stepped.
    engine.step(1e-3)
    frame = engine.get_force_torque_frame()
    assert frame is not None and frame.engine == "drake"
    assert frame.by_kind(WrenchKind.JOINT_REACTION)
    loads = engine.get_segment_axial_loads()
    assert loads is not None and loads.values_n["rod"] > 0.0
    assert engine.compute_contact_forces().shape == (3,)


def test_contact_results_failure_is_reported_unavailable() -> None:
    class _Failing(DrakeForceTorqueSource):
        def _read_contact_results(self, ctx):  # type: ignore[no-untyped-def]
            raise RuntimeError("query port not connected")

    plant, diagram, pctx = _pendulum()
    source = _Failing(plant, diagram)
    frame = source.sample(pctx)
    assert frame.by_kind(WrenchKind.CONTACT) == ()
    assert "contact:results" in source.unavailable_labels
