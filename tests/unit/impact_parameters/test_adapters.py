"""Tests for the per-engine ClubheadSeries adapters and cross-engine parity (GCV-16).

One kinematic chain (yaw z -> pitch y -> roll x, then a fixed rotated club body)
is built natively in every engine.  The same joint rollout must give the same
face-centre series and the same impact parameters.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from src.shared.python.impact_parameters import TargetFrame, extract_impact_parameters
from src.shared.python.impact_parameters.adapters import (
    NATIVE_CLUB_FACE,
    ClubFaceSpec,
    clubhead_series_from_drake,
    clubhead_series_from_mujoco,
    clubhead_series_from_myosuite,
    clubhead_series_from_opensim,
    clubhead_series_from_pinocchio,
    clubhead_series_from_simscape,
    rigid_body_series,
    select_impact_index,
)

pytestmark = pytest.mark.unit

N = 61
DT = 0.002
CLUB_PITCH = 0.4
SPEC = ClubFaceSpec(face_center_body_m=(0.01, -0.02, 0.03))

MJCF = f"""
<mujoco><compiler angle="radian"/><worldbody>
 <body name="l1" pos="0 0 1.5">
  <joint name="j1" type="hinge" axis="0 0 1"/>
  <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
  <body name="l2" pos="0.1 0 0">
   <joint name="j2" type="hinge" axis="0 1 0"/>
   <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
   <body name="l3" pos="0.3 0 -0.6">
    <joint name="j3" type="hinge" axis="1 0 0"/>
    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    <body name="club" pos="0 0 -0.5" quat="{math.cos(CLUB_PITCH / 2)} 0 {math.sin(CLUB_PITCH / 2)} 0">
     <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
   </body>
  </body>
 </body>
</worldbody></mujoco>
"""


def _link(name: str) -> str:
    return (
        f'<link name="{name}"><inertial><origin xyz="0 0 0"/><mass value="1"/>'
        '<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>'
        "</inertial></link>"
    )


def _joint(name, kind, parent, child, xyz, axis=None, rpy="0 0 0"):
    ax = f'<axis xyz="{axis}"/>' if axis else ""
    return (
        f'<joint name="{name}" type="{kind}"><parent link="{parent}"/>'
        f'<child link="{child}"/><origin xyz="{xyz}" rpy="{rpy}"/>{ax}'
        '<limit lower="-10" upper="10" effort="1" velocity="100"/></joint>'
    )


URDF = (
    '<robot name="chain">'
    + _link("base")
    + "".join(_link(n) for n in ("l1", "l2", "l3", "club"))
    + _joint("j1", "revolute", "base", "l1", "0 0 1.5", "0 0 1")
    + _joint("j2", "revolute", "l1", "l2", "0.1 0 0", "0 1 0")
    + _joint("j3", "revolute", "l2", "l3", "0.3 0 -0.6", "1 0 0")
    + _joint("cj", "fixed", "l3", "club", "0 0 -0.5", rpy=f"0 {CLUB_PITCH} 0")
    + "</robot>"
)


def rollout():
    t = np.arange(N) * DT
    amp = np.array([1.1, 0.7, 0.9])
    freq = np.array([9.0, 7.0, 11.0])
    phase = np.array([0.2, 0.9, 1.7])
    q = amp * np.sin(freq * t[:, None] + phase)
    v = amp * freq * np.cos(freq * t[:, None] + phase)
    return t, q, v


def _real_engine(name: str):
    """Import an engine binding, skipping when it is absent or a test stub.

    Other tests may leave a mock or bare stub in ``sys.modules`` on hosts
    without the engine; a real binding always has a ``__file__``.
    """
    module = pytest.importorskip(name)
    if not isinstance(module, types.ModuleType) or not getattr(
        module, "__file__", None
    ):
        pytest.skip(f"{name} is a test stub, not the real binding")
    return module


def _mujoco_model():
    mujoco = _real_engine("mujoco")
    return mujoco.MjModel.from_xml_string(MJCF)


def _pinocchio_model(tmp_path):
    pin = _real_engine("pinocchio")
    path = tmp_path / "chain.urdf"
    path.write_text(URDF)
    return pin.buildModelFromUrdf(str(path))


def _drake_plant(tmp_path):
    _real_engine("pydrake")
    _real_engine("pydrake.multibody.plant")
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import MultibodyPlant

    plant = MultibodyPlant(0.0)
    path = tmp_path / "chain.urdf"
    path.write_text(URDF)
    Parser(plant).AddModels(str(path))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"))
    plant.Finalize()
    return plant


def _opensim_model():
    osim = _real_engine("opensim")
    model = osim.Model()
    model.setName("chain")
    half = math.pi / 2

    def body(name):
        return osim.Body(name, 1.0, osim.Vec3(0), osim.Inertia(0.01, 0.01, 0.01))

    def pin(name, parent, child_body, loc, orient):
        child = body(child_body)
        model.addBody(child)
        joint = osim.PinJoint(
            name,
            parent,
            osim.Vec3(*loc),
            osim.Vec3(*orient),
            child,
            osim.Vec3(0),
            osim.Vec3(*orient),
        )
        joint.upd_coordinates(0).setName(name)
        model.addJoint(joint)
        return child

    l1 = pin("j1", model.getGround(), "l1", (0, 0, 1.5), (0, 0, 0))
    l2 = pin("j2", l1, "l2", (0.1, 0, 0), (-half, 0, 0))
    l3 = pin("j3", l2, "l3", (0.3, 0, -0.6), (0, half, 0))
    club = body("club")
    model.addBody(club)
    model.addJoint(
        osim.WeldJoint(
            "cj",
            l3,
            osim.Vec3(0, 0, -0.5),
            osim.Vec3(0, CLUB_PITCH, 0),
            club,
            osim.Vec3(0),
            osim.Vec3(0),
        )
    )
    return model


def _all_series(tmp_path, spec=SPEC):
    t, q, v = rollout()
    out = {}
    out["mujoco"] = clubhead_series_from_mujoco(_mujoco_model(), t, q, v, "club", spec)
    out["pinocchio"] = clubhead_series_from_pinocchio(
        _pinocchio_model(tmp_path), t, q, v, "club", spec
    )
    try:
        out["drake"] = clubhead_series_from_drake(
            _drake_plant(tmp_path), t, q, v, "club", spec
        )
    except pytest.skip.Exception:
        pass
    try:
        names = ("j1", "j2", "j3")
        out["opensim"] = clubhead_series_from_opensim(
            _opensim_model(),
            t,
            {n: q[:, i] for i, n in enumerate(names)},
            {n: v[:, i] for i, n in enumerate(names)},
            "club",
            spec,
        )
    except pytest.skip.Exception:
        pass
    return out


def _frame():
    return TargetFrame(target_dir=(0.0, -1.0, 0.0), ball_m=(0.0, 0.0, 0.0))


# ---------------------------------------------------------------- kernel ----
def test_face_spec_rejects_non_orthogonal_axes():
    with pytest.raises(ValueError, match="orthogonal"):
        ClubFaceSpec(toe_body=(1.0, 1.0, 0.0))


def test_face_spec_rejects_bad_offset():
    with pytest.raises(ValueError, match="3-vector"):
        ClubFaceSpec(face_center_body_m=(0.0, float("nan"), 0.0))


def test_rigid_body_series_transports_velocity_to_face_centre():
    n = 5
    t = np.arange(n) * 0.01
    rot = np.tile(np.eye(3), (n, 1, 1))
    spec = ClubFaceSpec(face_center_body_m=(0.1, 0.0, 0.0))
    s = rigid_body_series(
        t,
        np.zeros((n, 3)),
        rot,
        np.tile([1.0, 0, 0], (n, 1)),
        np.tile([0.0, 0, 2.0], (n, 1)),
        spec,
    )
    np.testing.assert_allclose(s.face_center_m[0], [0.1, 0, 0])
    np.testing.assert_allclose(s.velocity_mps[0], [1.0, 0.2, 0.0])
    np.testing.assert_allclose(s.face_normal[0], [1, 0, 0])
    np.testing.assert_allclose(s.toe_axis[0], [0, 0, -1])


def test_rigid_body_series_rejects_bad_rotation_and_shapes():
    t = np.arange(3) * 0.01
    bad = np.tile(2.0 * np.eye(3), (3, 1, 1))
    with pytest.raises(ValueError, match="orthonormal"):
        rigid_body_series(t, np.zeros((3, 3)), bad, np.zeros((3, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="origins_m"):
        rigid_body_series(
            t,
            np.zeros((2, 3)),
            np.tile(np.eye(3), (3, 1, 1)),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
        )


# ------------------------------------------------------- per-engine basics --
def _assert_valid(series):
    assert len(series) == N
    for arr in (
        series.face_center_m,
        series.velocity_mps,
        series.face_normal,
        series.toe_axis,
        series.grip_axis,
    ):
        assert np.all(np.isfinite(arr))
    axes = np.stack([series.face_normal, series.toe_axis, series.grip_axis], axis=1)
    gram = np.einsum("nai,nbi->nab", axes, axes)
    np.testing.assert_allclose(gram, np.tile(np.eye(3), (N, 1, 1)), atol=1e-9)


def test_mujoco_adapter_series_valid():
    t, q, v = rollout()
    _assert_valid(clubhead_series_from_mujoco(_mujoco_model(), t, q, v, "club", SPEC))


def test_pinocchio_adapter_series_valid(tmp_path):
    t, q, v = rollout()
    _assert_valid(
        clubhead_series_from_pinocchio(
            _pinocchio_model(tmp_path), t, q, v, "club", SPEC
        )
    )


def test_drake_adapter_series_valid(tmp_path):
    t, q, v = rollout()
    _assert_valid(
        clubhead_series_from_drake(_drake_plant(tmp_path), t, q, v, "club", SPEC)
    )


def test_myosuite_wrapper_delegates_to_mujoco_model():
    t, q, v = rollout()
    model = _mujoco_model()

    class Sim:
        pass

    sim = Sim()
    sim.model = model
    got = clubhead_series_from_myosuite(sim, t, q, v, "club", SPEC)
    want = clubhead_series_from_mujoco(model, t, q, v, "club", SPEC)
    np.testing.assert_array_equal(got.face_center_m, want.face_center_m)
    with pytest.raises(ValueError, match="model"):
        clubhead_series_from_myosuite(object(), t, q, v, "club", SPEC)


def test_adapter_preconditions(tmp_path):
    t, q, v = rollout()
    model = _mujoco_model()
    with pytest.raises(ValueError, match="no body"):
        clubhead_series_from_mujoco(model, t, q, v, "nope")
    with pytest.raises(ValueError, match="widths"):
        clubhead_series_from_mujoco(model, t, q[:, :2], v[:, :2], "club")
    with pytest.raises(ValueError, match="N="):
        clubhead_series_from_mujoco(model, t[:-1], q, v, "club")
    with pytest.raises(ValueError, match="no frame"):
        clubhead_series_from_pinocchio(_pinocchio_model(tmp_path), t, q, v, "nope")


def test_simscape_without_orientation_marks_face_unavailable():
    t, q, v = rollout()
    pos = np.column_stack([t, np.zeros(N), np.ones(N)])
    vel = np.tile([1.0, 0.0, 0.0], (N, 1))
    s = clubhead_series_from_simscape(t, pos, vel)
    assert s.face_normal is None and "orientation" in s.face_unobservable_reason
    res = extract_impact_parameters(s, _frame(), impact_index=30)
    assert res.face_angle_deg is None and "face_angle_deg" in res.unavailable


def test_simscape_with_orientation_enables_face():
    t = np.arange(4) * 0.01
    rot = np.tile(np.eye(3), (4, 1, 1))
    s = clubhead_series_from_simscape(
        t, np.zeros((4, 3)), np.ones((4, 3)), rot, np.zeros((4, 3))
    )
    assert s.face_normal is not None
    with pytest.raises(ValueError, match="angular_velocity"):
        clubhead_series_from_simscape(t, np.zeros((4, 3)), np.ones((4, 3)), rot)


def test_select_impact_index_default_and_override():
    t = np.arange(21) * 0.002
    pos = np.column_stack([t**3, np.zeros(21), np.zeros(21)])
    vel = np.gradient(pos, t, axis=0)
    s = clubhead_series_from_simscape(t, pos, vel)
    assert select_impact_index(s) >= 17
    assert select_impact_index(s, closest_approach=lambda _s: 7) == 7
    with pytest.raises(ValueError, match="outside"):
        select_impact_index(s, closest_approach=lambda _s: 99)


# ---------------------------------------------------------------- parity ----
def test_cross_engine_face_series_parity(tmp_path):
    series = _all_series(tmp_path)
    assert {"mujoco", "pinocchio"} <= set(series)
    ref = series["mujoco"]
    for name, s in series.items():
        np.testing.assert_allclose(
            s.face_center_m, ref.face_center_m, atol=1e-9, err_msg=name
        )
        np.testing.assert_allclose(
            s.velocity_mps, ref.velocity_mps, atol=1e-8, err_msg=name
        )
        np.testing.assert_allclose(
            s.face_normal, ref.face_normal, atol=1e-9, err_msg=name
        )
        np.testing.assert_allclose(s.toe_axis, ref.toe_axis, atol=1e-9, err_msg=name)


def test_cross_engine_impact_parameter_parity(tmp_path):
    """Clubhead speed within 1 %, AoA/path/face within 0.5 deg (measured ~1e-8)."""
    series = _all_series(tmp_path)
    frame = _frame()
    idx = 30
    results = {
        n: extract_impact_parameters(s, frame, impact_index=idx)
        for n, s in series.items()
    }
    ref = results["mujoco"]
    for name, res in results.items():
        assert res.clubhead_speed_mps == pytest.approx(
            ref.clubhead_speed_mps, rel=1e-2
        ), name
        for field in (
            "attack_angle_deg",
            "club_path_deg",
            "face_angle_deg",
            "face_to_path_deg",
            "dynamic_loft_deg",
        ):
            assert getattr(res, field) == pytest.approx(getattr(ref, field), abs=0.5), (
                name,
                field,
            )
        assert res.clubhead_speed_mps == pytest.approx(
            ref.clubhead_speed_mps, rel=1e-6
        ), name
