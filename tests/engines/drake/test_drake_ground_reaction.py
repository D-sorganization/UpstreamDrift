"""Drake ground-reaction breakdown on the force frame (GCV-2, #11708).

A pelvis carrying two sphere feet (``calcn_l`` / ``calcn_r``) stands on a
static ground; after settling for at least 0.2 s the frame must carry per-foot
and net GRF, CoP, free moment and moment about the centre of mass with net
vertical force equal to the weight.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pydrake.multibody.plant")  # skips if pydrake is mocked

from pydrake.math import RigidTransform  # noqa: E402
from pydrake.multibody.parsing import Parser  # noqa: E402
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph  # noqa: E402
from pydrake.systems.analysis import Simulator  # noqa: E402
from pydrake.systems.framework import DiagramBuilder  # noqa: E402

from src.engines.physics_engines.drake.python.drake_force_torque import (  # noqa: E402
    DrakeForceTorqueSource,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_drake]

G = 9.80665
PELVIS_KG = 70.0
FOOT_KG = 1.0
FOOT_Y = 0.15
RADIUS = 0.05
SETTLE_S = 1.0
WEIGHT_RTOL = 0.02
WEIGHT_N = (PELVIS_KG + 2 * FOOT_KG) * G

_INERTIA = '<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>'
STANCE_URDF = f"""<robot name="stance">
<link name="pelvis"><inertial><mass value="{PELVIS_KG}"/>{_INERTIA}</inertial></link>
<link name="calcn_l"><inertial><mass value="{FOOT_KG}"/>{_INERTIA}</inertial>
<collision><geometry><sphere radius="{RADIUS}"/></geometry></collision></link>
<link name="calcn_r"><inertial><mass value="{FOOT_KG}"/>{_INERTIA}</inertial>
<collision><geometry><sphere radius="{RADIUS}"/></geometry></collision></link>
<joint name="wl" type="fixed"><parent link="pelvis"/><child link="calcn_l"/>
<origin xyz="0 {FOOT_Y} -0.8"/></joint>
<joint name="wr" type="fixed"><parent link="pelvis"/><child link="calcn_r"/>
<origin xyz="0 -{FOOT_Y} -0.8"/></joint>
</robot>"""
GROUND_SDF = """<?xml version="1.0"?>
<sdf version="1.7"><model name="ground"><static>true</static><link name="g">
<collision name="c"><pose>0 0 -0.5 0 0 0</pose>
<geometry><box><size>5 5 1</size></box></geometry></collision></link></model></sdf>"""


def build_stance():
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, 1e-3)
    parser = Parser(plant)
    parser.AddModelsFromString(STANCE_URDF, "urdf")
    parser.AddModelsFromString(GROUND_SDF, "sdf")
    plant.Finalize()
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    pctx = plant.GetMyContextFromRoot(ctx)
    plant.SetFreeBodyPose(
        pctx, plant.GetBodyByName("pelvis"), RigidTransform([0, 0, 0.8 + RADIUS])
    )
    Simulator(diagram, ctx).AdvanceTo(SETTLE_S)
    return plant, diagram, pctx


@pytest.fixture(scope="module")
def stance():
    return build_stance()


def _labelled(stance):
    plant, diagram, pctx = stance
    return {
        w.label: w for w in DrakeForceTorqueSource(plant, diagram).sample(pctx).wrenches
    }


def test_static_stance_net_grf_equals_body_weight(stance) -> None:
    w = _labelled(stance)
    net = np.asarray(w["contact:grf_net"].force_n)
    assert abs(net[2] - WEIGHT_N) / WEIGHT_N < WEIGHT_RTOL
    np.testing.assert_allclose(net[:2], 0.0, atol=0.01 * WEIGHT_N)
    left = np.asarray(w["contact:grf_left"].force_n)
    right = np.asarray(w["contact:grf_right"].force_n)
    np.testing.assert_allclose(left + right, net, atol=1e-9)
    assert abs(left[2] - right[2]) / WEIGHT_N < WEIGHT_RTOL


def test_static_stance_cop_is_between_the_feet_on_the_ground(stance) -> None:
    w = _labelled(stance)
    cop = np.asarray(w["contact:grf_net"].point_m)
    assert abs(cop[0]) < RADIUS and abs(cop[1]) < FOOT_Y
    assert abs(cop[2]) < 0.01
    assert w["contact:grf_left"].point_m[1] > 0.0 > w["contact:grf_right"].point_m[1]


def test_static_stance_com_moment_uses_the_plant_com(stance) -> None:
    plant, diagram, pctx = stance
    w = _labelled(stance)
    com = plant.CalcCenterOfMassPositionInWorld(pctx)
    np.testing.assert_allclose(w["contact:moment_com_net"].point_m, com, atol=1e-9)
    for label in ("contact:free_moment_net", "contact:free_moment_left"):
        assert label in w
    m_com = np.asarray(w["contact:moment_com_net"].torque_nm)
    com_height = w["contact:moment_com_net"].point_m[2]
    assert np.linalg.norm(m_com[:2]) < 0.01 * WEIGHT_N * com_height


def test_airborne_has_no_ground_reaction() -> None:
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, 1e-3)
    parser = Parser(plant)
    parser.AddModelsFromString(STANCE_URDF, "urdf")
    parser.AddModelsFromString(GROUND_SDF, "sdf")
    plant.Finalize()
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    pctx = plant.GetMyContextFromRoot(ctx)
    plant.SetFreeBodyPose(
        pctx, plant.GetBodyByName("pelvis"), RigidTransform([0, 0, 3.0])
    )
    labels = {
        w.label for w in DrakeForceTorqueSource(plant, diagram).sample(pctx).wrenches
    }
    assert not any(label.startswith("contact:grf") for label in labels)
