"""OSV-8 (#11755): the rendered clubface is square to the target at address.

For every engine the club body is placed by that engine's own forward
kinematics at the captured address pose (capture A: driver, capture B:
7-iron). The world face normal of the rendered head mesh must agree with the
spec ``Clubface Vector`` direction to 1 degree, and the horizontal face angle
against the target axis (native -Y, positive open for a right-handed golfer)
must be square to 2 degrees.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import club_face as cf
from src.shared.python.model_appearance.club_head_mesh import (
    load_club_head,
    measured_face_normal,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = ROOT / "docs/development/full_body_models"
POSES = json.loads(
    (ROOT / "tests/fixtures/club_face/address_poses.json").read_text(encoding="utf-8")
)["poses"]
CAPTURES = {"A": "driver", "B": "iron7"}
SQUARE_TOL_DEG = 2.0
SPEC_VECTOR_TOL_DEG = 1.0


def _spec_bytes(club: str) -> bytes:
    return (SPEC_DIR / f"full_body_spec_anthro_{club}.json").read_bytes()


def _coords(club: str) -> dict[str, float]:
    pose = POSES[club]
    return dict(zip(pose["coordinate_order"], pose["q_rad"], strict=True))


def _unit_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(a @ b) / float(np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


# ----------------------------------------------- engine forward kinematics
# Each returns (R_geom_world, engine_mesh_vertices): the world rotation of the
# frame the engine's head mesh is expressed in, and that mesh's vertices.


def _assembly_head_vertices(club: str) -> np.ndarray:
    assembly = ca.assembly_from_spec(json.loads(_spec_bytes(club)))
    return ca.assembly_meshes(assembly)["head"].vertices


def _opensim(club: str) -> tuple[np.ndarray, np.ndarray]:
    osim = pytest.importorskip("opensim")
    import tempfile

    from src.engines.physics_engines.opensim.python import club_visuals
    from src.engines.physics_engines.opensim.python.full_body_osim import (
        export_full_body_osim,
    )

    xml, _ = export_full_body_osim(_spec_bytes(club), club_geometry_ref="")
    club_visuals.register_geometry_path()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.osim"
        path.write_text(xml, encoding="utf-8")
        model = osim.Model(str(path))
    state = model.initSystem()
    coordinates = model.getCoordinateSet()
    for name, value in _coords(club).items():
        coordinates.get(name).setValue(state, float(value), False)
    model.realizePosition(state)
    rot = model.getBodySet().get("Clubhead").getTransformInGround(state).R()
    matrix = np.array([[rot.get(i, j) for j in range(3)] for i in range(3)])
    stl = club_visuals.asset_paths(club)["head"]
    from src.shared.python.model_appearance.mesh_io import read_stl_bytes

    return matrix, read_stl_bytes(stl.read_bytes()).vertices


def _mujoco_model(club: str, arena: bool = False):  # noqa: ANN202
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, _ = export_full_body_mjcf(_spec_bytes(club), visual=True)
    if arena:
        pytest.importorskip("myosuite")
        from src.tools.native_viewer_export.backends.myosuite_worker import (
            merge_arena,
        )

        xml = merge_arena(xml, 0.0, 320, 240)
    return mujoco, mujoco.MjModel.from_xml_string(xml)


def _mujoco_like(club: str, arena: bool) -> tuple[np.ndarray, np.ndarray]:
    mujoco, model = _mujoco_model(club, arena)
    data = mujoco.MjData(model)
    for name, value in _coords(club).items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "vmesh_club_head")
    assert mesh_id >= 0
    geom = int(np.flatnonzero(model.geom_dataid == mesh_id)[0])
    adr, num = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    return (
        np.array(data.geom_xmat[geom]).reshape(3, 3),
        np.array(model.mesh_vert[adr : adr + num], dtype=float),
    )


def _mujoco(club: str) -> tuple[np.ndarray, np.ndarray]:
    return _mujoco_like(club, arena=False)


def _myosuite(club: str) -> tuple[np.ndarray, np.ndarray]:
    return _mujoco_like(club, arena=True)


def _drake(club: str) -> tuple[np.ndarray, np.ndarray]:
    pytest.importorskip("pydrake")
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import MultibodyPlant

    from src.tools.native_viewer_export.backends._subprocess import export_urdf

    xml, meta = export_urdf(_spec_bytes(club))
    plant = MultibodyPlant(0.0)
    inst = Parser(plant).AddModelsFromString(xml, "urdf")[0]
    links = meta["body_links"]
    plant.WeldFrames(
        plant.world_frame(), plant.GetBodyByName(links["world"], inst).body_frame()
    )
    plant.Finalize()
    ctx = plant.CreateDefaultContext()
    full = np.zeros(plant.num_positions())
    for name, value in _coords(club).items():
        full[plant.GetJointByName(name, inst).position_start()] = value
    plant.SetPositions(ctx, full)
    name = ca.club_body_name(json.loads(_spec_bytes(club)))
    pose = plant.EvalBodyPoseInWorld(ctx, plant.GetBodyByName(links[name], inst))
    return np.array(pose.rotation().matrix()), _assembly_head_vertices(club)


def _pinocchio(club: str) -> tuple[np.ndarray, np.ndarray]:
    pin = pytest.importorskip("pinocchio")
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    spec = json.loads(_spec_bytes(club))
    adapter = FullBodyPinocchioModel(spec)
    q = adapter.configuration(_coords(club))
    data = adapter.model.createData()
    pin.forwardKinematics(adapter.model, data, q)
    joint, body_pose = adapter._bodies[ca.club_body_name(spec)]  # noqa: SLF001
    placement = data.oMi[joint] * body_pose
    return np.array(placement.rotation), _assembly_head_vertices(club)


ENGINES: dict[str, Callable[[str], tuple[np.ndarray, np.ndarray]]] = {
    "opensim": _opensim,
    "mujoco": _mujoco,
    "drake": _drake,
    "pinocchio": _pinocchio,
    "myosuite": _myosuite,
}


def _procrustes(mesh, dst: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Rotation taking ``mesh`` onto the engine vertices ``dst``.

    ``dst`` is either the welded vertex array or an unwelded STL read-back
    (three vertices per face, same face order).
    """
    src = mesh.vertices
    if len(dst) == 3 * len(mesh.faces) and len(dst) != len(src):
        src = src[mesh.faces].reshape(-1, 3)
    assert src.shape == dst.shape
    a = src - src.mean(axis=0)
    b = dst - dst.mean(axis=0)
    u, _, vt = np.linalg.svd(b.T @ a)
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ d @ vt
    np.testing.assert_allclose(a @ rot.T, b, atol=5e-6)  # the engine mesh is rigid
    return rot


def _world_vectors(engine: str, club: str) -> tuple[np.ndarray, np.ndarray]:
    """(world normal of the rendered head mesh, world spec Clubface Vector)."""
    geom_rot, engine_vertices = ENGINES[engine](club)
    assembly = ca.assembly_from_spec(json.loads(_spec_bytes(club)))
    head = load_club_head(assembly.head_alias)
    club_head = ca.assembly_meshes(assembly)["head"]
    to_geom_from_club = _procrustes(club_head, engine_vertices)
    to_geom_from_head = _procrustes(head.mesh, engine_vertices)
    rendered = geom_rot @ to_geom_from_head @ measured_face_normal(head.head_mesh)
    spec = geom_rot @ to_geom_from_club @ ca.clubface_vector(assembly)
    return rendered, spec


@pytest.mark.parametrize("capture", sorted(CAPTURES))
@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_rendered_face_normal_matches_the_spec_clubface_vector(
    engine: str, capture: str
) -> None:
    rendered, spec = _world_vectors(engine, CAPTURES[capture])
    assert _unit_angle_deg(rendered, spec) <= SPEC_VECTOR_TOL_DEG


@pytest.mark.parametrize("capture", sorted(CAPTURES))
@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_face_is_square_to_the_target_at_address(engine: str, capture: str) -> None:
    rendered, spec = _world_vectors(engine, CAPTURES[capture])
    assert abs(cf.horizontal_face_angle_deg(rendered)) <= SQUARE_TOL_DEG
    assert abs(cf.horizontal_face_angle_deg(spec)) <= SQUARE_TOL_DEG


# -------------------------------------------------------------- shared source


def test_face_angle_sign_convention_positive_is_open_for_a_right_hander() -> None:
    assert cf.horizontal_face_angle_deg([0.0, -1.0, 0.2]) == pytest.approx(0.0)
    open_normal = [-math.sin(math.radians(10)), -math.cos(math.radians(10)), 0.0]
    assert cf.horizontal_face_angle_deg(open_normal) == pytest.approx(10.0)
    closed = [math.sin(math.radians(10)), -math.cos(math.radians(10)), 0.0]
    assert cf.horizontal_face_angle_deg(closed) == pytest.approx(-10.0)


def test_face_angle_rejects_a_vertical_normal() -> None:
    with pytest.raises(ValueError, match="horizontal"):
        cf.horizontal_face_angle_deg([0.0, 0.0, 1.0])


@pytest.mark.parametrize("club", sorted(CAPTURES.values()))
def test_roll_is_one_shared_constant_that_rolls_the_head_about_the_shaft(
    club: str,
) -> None:
    assembly = ca.assembly_from_spec(json.loads(_spec_bytes(club)))
    assert assembly.face_roll_deg == ca.ADDRESS_SQUARE_FACE_ROLL_DEG[club]
    flat = ca.assembly_meshes(
        ca.ClubAssembly(
            assembly.head_alias,
            assembly.length_m,
            assembly.grip_length_m,
            assembly.shaft_radius_m,
            assembly.axis_offset_m,
        )
    )
    rolled = ca.assembly_meshes(assembly)
    for part in ("shaft", "grip"):  # axisymmetric parts do not move
        np.testing.assert_allclose(rolled[part].vertices, flat[part].vertices)
    head_flat, head_rolled = flat["head"].vertices, rolled["head"].vertices
    axis = np.array([0.0, 0.0, assembly.axis_offset_m])
    shaft_distance = np.linalg.norm((head_flat - axis)[:, [0, 2]], axis=1)
    rolled_distance = np.linalg.norm((head_rolled - axis)[:, [0, 2]], axis=1)
    np.testing.assert_allclose(rolled_distance, shaft_distance, atol=1e-12)
    np.testing.assert_allclose(head_rolled[:, 1], head_flat[:, 1], atol=1e-12)


def test_unrolled_assembly_keeps_the_legacy_geometry() -> None:
    assembly = ca.ClubAssembly("iron7", 0.94, 0.265, 0.0065)
    assert assembly.face_roll_deg == 0.0
    vec = ca.clubface_vector(assembly)
    head = load_club_head("iron7")
    expected = head_to_club(head) @ np.array(
        [
            math.cos(math.radians(head.loft_deg)),
            math.sin(math.radians(head.loft_deg)),
            0,
        ]
    )
    np.testing.assert_allclose(vec, expected, atol=1e-12)


def head_to_club(head) -> np.ndarray:  # noqa: ANN001
    from src.shared.python.model_appearance.club_head_mesh import head_to_club_rotation

    return head_to_club_rotation(head)


def test_spec_club_block_overrides_the_default_roll() -> None:
    spec = json.loads(_spec_bytes("driver"))
    spec["club"]["face_roll_deg"] = -10.0
    assert ca.assembly_from_spec(spec).face_roll_deg == -10.0
    spec["club"]["face_roll_deg"] = float("nan")
    with pytest.raises(ValueError, match="face_roll_deg"):
        ca.assembly_from_spec(spec)


# ------------------------------------------------------------ impact frame
SWING_DT_S = 0.002  # fixtures hold every second sample of the 1 kHz reference


def _face_centre_series(club: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(time, face-centre world xyz, face horizontal angle) over the swing (MuJoCo FK)."""
    mujoco, model = _mujoco_model(club)
    from src.shared.python.model_appearance.club_head_mesh import face_centre

    q = np.load(ROOT / f"tests/fixtures/club_face/swing_q_{club}.npz")["q"]
    order = POSES[club]["coordinate_order"]
    adr = [model.joint(n).qposadr[0] for n in order]
    data = mujoco.MjData(model)
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "vmesh_club_head")
    geom = int(np.flatnonzero(model.geom_dataid == mesh_id)[0])
    vadr, vnum = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    vertices = np.array(model.mesh_vert[vadr : vadr + vnum], dtype=float)
    assembly = ca.assembly_from_spec(json.loads(_spec_bytes(club)))
    head = load_club_head(assembly.head_alias)
    src = head.mesh.vertices
    rot = _procrustes(head.mesh, vertices)
    trans = vertices.mean(axis=0) - rot @ src.mean(axis=0)
    centre_g = rot @ face_centre(head.head_mesh) + trans
    normal_g = rot @ measured_face_normal(head.head_mesh)
    centres, angles = [], []
    for row in q.astype(float):
        data.qpos[adr] = row
        mujoco.mj_forward(model, data)
        r = np.array(data.geom_xmat[geom]).reshape(3, 3)
        centres.append(np.array(data.geom_xpos[geom]) + r @ centre_g)
        angles.append(cf.horizontal_face_angle_deg(r @ normal_g))
    return np.arange(len(q)) * SWING_DT_S, np.array(centres), np.array(angles)


@pytest.mark.parametrize("capture", sorted(CAPTURES))
def test_impact_frame_puts_the_clubhead_at_the_ball(capture: str) -> None:
    time, head, _ = _face_centre_series(CAPTURES[capture])
    k = cf.impact_frame(time, head)
    assert k > cf.top_of_backswing_index(time, head)
    assert abs(head[k, 2] - head[0, 2]) <= 0.10
    assert np.linalg.norm(head[k] - head[0]) <= cf.IMPACT_BALL_RADIUS_M
    assert 1.2 < time[k] < 1.45  # downswing of the 1.8 s captures, not the finish


@pytest.mark.parametrize("capture", sorted(CAPTURES))
@pytest.mark.xfail(
    strict=True,
    reason="#11755 open: the matched trajectory leaves the face 18-30 degrees "
    "open at impact while the capture head triad is square; needs an IK refit "
    "with head-triad weight, not a constant roll",
)
def test_face_is_square_at_impact(capture: str) -> None:
    time, head, angle = _face_centre_series(CAPTURES[capture])
    assert abs(angle[cf.impact_frame(time, head)]) <= 10.0


def _swing(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0, n)
    head = np.zeros((n, 3))
    head[:, 2] = 0.1 + 1.0 * np.sin(np.pi * t / 2.0) ** 2  # up and back to the ball
    head[:, 1] = 1.0 * np.sin(np.pi * t / 2.0) ** 2
    return t, head


def test_impact_frame_accepts_a_swing_that_returns_to_the_ball() -> None:
    t, head = _swing()
    k = cf.impact_frame(t, head)
    assert abs(head[k, 2] - head[0, 2]) <= 0.10


def test_impact_frame_raises_when_the_head_never_returns_to_the_ball() -> None:
    t, head = _swing()
    head[1:, 2] += 0.5  # leaves the ball and never comes back down
    with pytest.raises(ValueError, match="no valid impact frame"):
        cf.impact_frame(t, head)


def test_impact_frame_rejects_bad_shapes_and_tolerances() -> None:
    t, head = _swing()
    with pytest.raises(ValueError, match="clubhead"):
        cf.impact_frame(t[:-1], head)
    with pytest.raises(ValueError, match="tolerances"):
        cf.impact_frame(t, head, height_tol_m=0.0)


def test_ball_passage_finds_the_sub_sample_crossing() -> None:
    t = np.linspace(0.0, 2.0, 41)  # coarse: 5 cm per sample near the ball
    head = np.zeros((len(t), 3))
    phase = np.pi * t / 2.0
    head[:, 1] = -np.sin(2.0 * phase) * 0.8  # back, through the ball, through
    head[:, 2] = 0.1 + 0.9 * np.sin(phase) ** 2 * (t < 1.0)
    t_imp, k, s = cf.ball_passage(t, head)
    assert t[k] <= t_imp <= t[k + 1] and 0.0 <= s <= 1.0
    point = head[k] + s * (head[k + 1] - head[k])
    assert np.linalg.norm(point - head[0]) < 1e-9
    assert t_imp == pytest.approx(1.0)


def test_ball_passage_contracts() -> None:
    t, head = _swing()
    with pytest.raises(ValueError, match="clubhead"):
        cf.ball_passage(t[:-1], head)
    with pytest.raises(ValueError, match="tolerances"):
        cf.ball_passage(t, head, ball_radius_m=-1.0)
    bad = head.copy()
    bad[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cf.ball_passage(t, bad)
    head[1:, 2] += 0.5
    with pytest.raises(ValueError, match="no valid impact"):
        cf.ball_passage(t, head)


def test_face_angle_at_blends_neighbouring_normals() -> None:
    a, b = math.radians(10.0), math.radians(-10.0)
    normals = np.array(
        [[-math.sin(a), -math.cos(a), 0.0], [-math.sin(b), -math.cos(b), 0]]
    )
    assert cf.face_angle_at(normals, 0, 0.5) == pytest.approx(0.0, abs=1e-9)
    assert cf.face_angle_at(normals, 0, 0.0) == pytest.approx(10.0)
    with pytest.raises(ValueError, match="segment"):
        cf.face_angle_at(normals, 1, 0.5)
    with pytest.raises(ValueError, match="segment"):
        cf.face_angle_at(normals, 0, 1.5)


def _circle_swing(t: np.ndarray) -> np.ndarray:
    """Head on a 1 m circle: slow backswing to the top at t = 1, back to the
    ball at t = 1.4 at full speed, then through."""
    theta = np.where(
        t < 1.0,
        np.pi / 2.0 * (1.0 - np.cos(np.pi * t)),
        np.pi * np.cos(np.pi / 2.0 * (t - 1.0) / 0.4),
    )
    head = np.zeros((len(t), 3))
    head[:, 1] = np.sin(theta)
    head[:, 2] = 0.1 + (1.0 - np.cos(theta))
    return head


def test_face_events_report_address_top_impact_and_peak_speed() -> None:
    t = np.linspace(0.0, 1.6, 801)
    head = _circle_swing(t)
    turn = np.radians(15.0) * t  # face opens 15 degrees per second
    normals = np.column_stack([-np.sin(turn), -np.cos(turn), np.zeros_like(t)])
    events = cf.face_events(t, normals, head)
    assert events.address_deg == pytest.approx(0.0)
    assert events.top_time_s == pytest.approx(1.0, abs=0.01)
    assert events.top_deg == pytest.approx(15.0 * events.top_time_s)
    assert events.impact_time_s == pytest.approx(1.4, abs=1e-3)
    assert events.impact_deg == pytest.approx(21.0, abs=0.02)
    assert events.peak_speed_time_s == pytest.approx(1.4, abs=0.01)
    assert events.peak_speed_gap_to_ball_m < 0.01
    with pytest.raises(ValueError, match="normals"):
        cf.face_events(t, normals[:-1], head)
