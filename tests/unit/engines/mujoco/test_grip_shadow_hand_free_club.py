"""MuJoCo free-club scene with Shadow hand visuals (#15122)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.engines.physics_engines.mujoco.python.grip_shadow_hand_free_club import (
    build_free_club_shadow_hands_mjcf,
)
from src.shared.python.grip_spec import GripSpec
from src.shared.python.model_appearance.club_assembly import ClubAssembly
from src.shared.python.model_appearance.grip_hands import grip_hand_meshes
from src.shared.python.model_appearance.mesh_io import read_stl_bytes

pytestmark = pytest.mark.unit

mujoco = pytest.importorskip("mujoco")

_HANDS = ("lead_hand", "trail_hand")
_TRI_ATOL = 1e-5


def _club() -> ClubAssembly:
    """The free-club scene's default iron, shared by the geometry checks."""
    return ClubAssembly("iron7", 0.95, 0.25, 0.0065)


def _triangle_vertices(mesh) -> np.ndarray:
    """Face corners in write order, the same order binary STL stores."""
    return np.asarray(mesh.vertices, float)[np.asarray(mesh.faces)].reshape(-1, 3)


def _exported_triangles(mesh_dir: Path, label: str) -> np.ndarray:
    """Triangle corners of the STL the scene wrote for ``label``."""
    mesh = read_stl_bytes((mesh_dir / f"{label}.stl").read_bytes())
    return _triangle_vertices(mesh)


def _body_id(model, name: str) -> int:
    """MuJoCo body id for ``name``."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _geom_pose_in_body(data, gid: int, bid: int) -> tuple[np.ndarray, np.ndarray]:
    """Geom origin and rotation expressed in body ``bid``'s frame."""
    body_rot = data.xmat[bid].reshape(3, 3)
    origin = body_rot.T @ (data.geom_xpos[gid] - data.xpos[bid])
    rotation = body_rot.T @ data.geom_xmat[gid].reshape(3, 3)
    return origin, rotation


def _compiled_world_vertices(model, data, gid: int) -> np.ndarray:
    """Mesh vertices of geom ``gid`` in the world frame after ``mj_forward``."""
    mesh_id = int(model.geom_dataid[gid])
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    local = np.asarray(model.mesh_vert[start : start + count], float)
    rotation = data.geom_xmat[gid].reshape(3, 3)
    return local @ rotation.T + data.geom_xpos[gid]


def test_free_club_shadow_hands_mjcf_compiles() -> None:
    """The free-club Shadow hand MJCF compiles in MuJoCo."""
    xml, mesh_dir, geom_names = build_free_club_shadow_hands_mjcf(GripSpec.legacy())
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.ngeom >= 3
    assert set(geom_names) == {"lead_hand_vis", "trail_hand_vis"}
    assert mesh_dir.is_dir()
    for gname in geom_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        assert gid >= 0
        assert model.geom_contype[gid] == 0
        assert model.geom_conaffinity[gid] == 0


def test_free_club_exports_shadow_hand_geometry() -> None:
    """The scene writes Shadow hand meshes, not the capsule grip hands."""
    spec = GripSpec.legacy()
    club = _club()
    _xml, mesh_dir, _names = build_free_club_shadow_hands_mjcf(spec, club=club)
    pose = spec.to_grip_pose()
    shadow = grip_hand_meshes(club, pose, grip_spec=spec, hand_backend="shadow")
    capsule = grip_hand_meshes(club, pose, grip_spec=spec)
    for label in _HANDS:
        exported = _exported_triangles(mesh_dir, label)
        assert exported == pytest.approx(
            _triangle_vertices(shadow[label]), abs=_TRI_ATOL
        )
        assert exported != pytest.approx(
            _triangle_vertices(capsule[label]), abs=_TRI_ATOL
        )


def test_placed_hand_vertices_are_not_framed_again() -> None:
    """Club-frame vertices are not transformed a second time by the hand body.

    MuJoCo may store a mesh in its own inertia frame. With the free joint at
    rest, composing that frame back to the world must recover the exported
    vertices, and the hand body itself stays at the club origin.
    """
    spec = GripSpec.legacy()
    xml, mesh_dir, _names = build_free_club_shadow_hands_mjcf(spec, club=_club())
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for label in _HANDS:
        bid = _body_id(model, f"{label}_body")
        assert model.body_pos[bid] == pytest.approx(0.0, abs=1e-8)
        assert model.body_quat[bid] == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-8)
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{label}_vis")
        world = _compiled_world_vertices(model, data, gid)
        exported = _exported_triangles(mesh_dir, label)
        distance, _index = cKDTree(exported).query(world, k=1)
        assert float(np.max(distance)) < 1e-6


def test_hands_stay_on_the_club_when_the_free_joint_moves() -> None:
    """Hand bodies are children of the free club and move with it."""
    spec = GripSpec.legacy()
    xml, _mesh_dir, _names = build_free_club_shadow_hands_mjcf(spec, club=_club())
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    club_id = _body_id(model, "club")
    poses = {}
    for label in _HANDS:
        bid = _body_id(model, f"{label}_body")
        assert model.body_parentid[bid] == club_id
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{label}_vis")
        poses[label] = _geom_pose_in_body(data, gid, club_id)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "club_free")
    adr = int(model.jnt_qposadr[joint])
    yaw = math.tau / 8.0
    data.qpos[adr : adr + 7] = (
        0.25,
        -0.15,
        0.4,
        math.cos(yaw),
        0.0,
        0.0,
        math.sin(yaw),
    )
    mujoco.mj_forward(model, data)
    for label, (origin, rotation) in poses.items():
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{label}_vis")
        moved_origin, moved_rotation = _geom_pose_in_body(data, gid, club_id)
        assert moved_origin == pytest.approx(origin, abs=1e-8)
        assert moved_rotation == pytest.approx(rotation, abs=1e-8)
