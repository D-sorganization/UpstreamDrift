"""NSX-50: native-engine ground truth for the dual dynamic maps (#12578).

For each engine a floating body with one hinge is driven with a native velocity
``v_n = T v_c`` and the engine's own kinematics must report the canonical base
twist: linear velocity of the body origin in the world frame and angular
velocity in the body frame.  This pins the frame of every native base block
(the classic silent-divergence trap of ``docs/conventions/canonical-v2.md``).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_interchange import dynamic_maps as dm
from src.shared.python.pose_interchange import se3
from src.shared.python.pose_interchange.adapters import (
    DrakeAdapter,
    MujocoAdapter,
    OpenSimAdapter,
    PinocchioAdapter,
)
from src.shared.python.pose_interchange.canonical_state import CanonicalState
from tests.helpers.opensim_support import require_opensim

pytestmark = [pytest.mark.integration]


def _layout_one_joint(adapter, base_offset: int):
    """A one-joint layout (the golfer's first canonical joint)."""
    from dataclasses import replace

    layout = dict(adapter.joint_layout())
    name = dm.canonical_joint_names(layout)[0]
    return {name: replace(layout[name], start_index=base_offset)}


def _state() -> CanonicalState:
    rng = np.random.default_rng(42)
    quat = se3.quat_exp(rng.normal(scale=0.5, size=3))
    q = np.concatenate([rng.normal(size=3), quat, [0.4]])
    return CanonicalState(q=q, v=rng.normal(size=7), a=np.zeros(7))


def _map(adapter, native_q, base_offset):
    layout = _layout_one_joint(adapter, base_offset)
    t_mat, _ = dm.velocity_map_matrices(adapter.dynamic_layout, native_q, layout)
    return t_mat


@pytest.mark.requires_pinocchio
def test_pinocchio_free_flyer_velocity_frames() -> None:
    pin = pytest.importorskip("pinocchio")
    model = pin.Model()
    base = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root")
    model.appendBodyToJoint(base, pin.Inertia.Random(), pin.SE3.Identity())
    model.addJoint(base, pin.JointModelRZ(), pin.SE3.Identity(), "hinge")
    data = model.createData()
    state = _state()
    pos, quat, joints = state.base_position, state.base_quat_wxyz, state.joint_q
    q_n = np.concatenate([pos, quat[[1, 2, 3, 0]], joints])
    t_mat = _map(PinocchioAdapter(), q_n, 7)
    v_n = t_mat @ state.v
    pin.forwardKinematics(model, data, q_n, v_n)
    rot = data.oMi[1].rotation
    np.testing.assert_allclose(rot, se3.quat_to_matrix(quat), atol=1e-12)
    np.testing.assert_allclose(rot @ data.v[1].linear, state.v[0:3], atol=1e-12)
    np.testing.assert_allclose(data.v[1].angular, state.v[3:6], atol=1e-12)
    assert v_n[6] == pytest.approx(state.v[6])


@pytest.mark.requires_drake
def test_drake_quaternion_floating_velocity_frames() -> None:
    plant_mod = pytest.importorskip("pydrake.multibody.plant")
    tree_mod = pytest.importorskip("pydrake.multibody.tree")

    plant = plant_mod.MultibodyPlant(0.0)
    body = plant.AddRigidBody(
        "body",
        tree_mod.SpatialInertia.SolidSphereWithDensity(1000.0, 0.1),
    )
    plant.Finalize()
    context = plant.CreateDefaultContext()
    state = _state()
    pos, quat = state.base_position, state.base_quat_wxyz
    q_n = np.concatenate([quat, pos])
    v_c = state.v[:6]
    adapter = DrakeAdapter()
    t_mat = dm.velocity_map_matrices(adapter.dynamic_layout, q_n, {})[0]
    plant.SetPositions(context, q_n)
    plant.SetVelocities(context, t_mat @ v_c)
    spatial = plant.EvalBodySpatialVelocityInWorld(context, body)
    rot = se3.quat_to_matrix(quat)
    np.testing.assert_allclose(spatial.translational(), v_c[0:3], atol=1e-12)
    np.testing.assert_allclose(rot.T @ spatial.rotational(), v_c[3:6], atol=1e-12)


@pytest.mark.requires_mujoco
def test_mujoco_free_joint_velocity_frames() -> None:
    mj = pytest.importorskip("mujoco")
    xml = (
        "<mujoco><worldbody><body name='b'><freejoint/>"
        "<inertial pos='0 0 0' mass='1' diaginertia='1 1 1'/>"
        "<geom type='sphere' size='0.1' mass='0'/></body></worldbody></mujoco>"
    )
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    state = _state()
    v_c = state.v[:6]
    q_n = np.concatenate([state.base_position, state.base_quat_wxyz])
    t_mat = dm.velocity_map_matrices(MujocoAdapter().dynamic_layout, q_n, {})[0]
    data.qpos[:] = q_n
    data.qvel[:] = t_mat @ v_c
    mj.mj_forward(model, data)
    res = np.zeros(6)
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "b")
    mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, body_id, res, 0)
    rot = se3.quat_to_matrix(state.base_quat_wxyz)
    np.testing.assert_allclose(res[3:6], v_c[0:3], atol=1e-12)
    np.testing.assert_allclose(rot.T @ res[0:3], v_c[3:6], atol=1e-12)


@pytest.mark.requires_opensim
def test_opensim_free_joint_euler_rates_match_engine_rotation() -> None:
    """The adapter's v is dq/dt of the XYZ Euler coordinates (deg/s).

    OpenSim's own ``Coordinate`` speed of a FreeJoint is the Simbody mobilizer
    speed (parent-frame angular velocity), which is *not* dq/dt; this adapter
    maps coordinate rates, so the engine is used here for its rotation
    ``R(q)`` and the body angular velocity is its finite difference.
    """
    osim = require_opensim()
    model = osim.Model()
    body = osim.Body("b", 1.0, osim.Vec3(0), osim.Inertia(1, 1, 1))
    joint = osim.FreeJoint(
        "f",
        model.getGround(),
        osim.Vec3(0),
        osim.Vec3(0),
        body,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    model.addBody(body)
    model.addJoint(joint)
    system_state = model.initSystem()
    coords = model.getCoordinateSet()

    def rotation(euler_deg: np.ndarray) -> np.ndarray:
        for i, angle in enumerate(np.radians(euler_deg)):
            coords.get(i).setValue(system_state, float(angle), False)
        model.realizePosition(system_state)
        mat = body.getTransformInGround(system_state).R()
        return np.array([[mat.get(i, k) for k in range(3)] for i in range(3)])

    euler = np.array([12.0, -25.0, 40.0])
    rate = np.array([30.0, -20.0, 50.0])  # deg/s
    np.testing.assert_allclose(
        rotation(euler), se3.euler_xyz_deg_to_matrix(euler), atol=1e-12
    )
    h = 1e-6
    d_rot = (rotation(euler + h * rate) - rotation(euler - h * rate)) / (2 * h)
    spin = rotation(euler).T @ d_rot
    omega_body = np.array([spin[2, 1], spin[0, 2], spin[1, 0]])
    q_n = np.concatenate([np.zeros(3), euler, [0.0]])
    adapter = OpenSimAdapter()
    layout = _layout_one_joint(adapter, 6)
    t_mat, _ = dm.velocity_map_matrices(adapter.dynamic_layout, q_n, layout)
    v_c = np.concatenate([np.zeros(3), omega_body, [0.0]])
    np.testing.assert_allclose((t_mat @ v_c)[3:6], rate, atol=1e-6)
