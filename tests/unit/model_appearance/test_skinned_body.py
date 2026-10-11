"""CMB-6 (#11657): skinned humanoid mesh over the spec skeleton, exported as glTF."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from src.shared.python.humanoid_character_builder.presets.loader import (
    list_character_presets,
)
from src.shared.python.humanoid_character_builder.spec_export import (
    compile_character,
    export_character,
)
from src.shared.python.model_appearance.library import classify_body
from src.shared.python.model_appearance.skinned_body import (
    build_skinned_body,
    gltf_up_rotation,
    pose_skinned_vertices,
    quaternion_xyzw,
    skinned_body_gltf,
    skinned_body_gltf_text,
    spec_body_poses,
    spec_motion_bone_poses,
)
from src.shared.python.motion_matching.anthropometric_candidate import (
    zero_pose_joint_positions,
)

pytestmark = pytest.mark.unit

_LIMB_PARTS = ("upper_arm", "forearm", "thigh", "shin")


def _tiny_spec() -> dict:
    """A two-link toy spec: world -> link1 (Rz) -> link2 (Px)."""
    identity = np.eye(4).tolist()
    return {
        "coordinate_order": ["rz", "px"],
        "bodies": [
            {"name": "world"},
            {"name": "link1", "solids": []},
            {"name": "link2", "solids": []},
        ],
        "joints": [
            {
                "name": "j1",
                "parent": "world",
                "child": "link1",
                "parent_to_base": identity,
                "child_to_follower": identity,
                "primitives": [{"primitive": "Rz", "coordinate": "rz"}],
            },
            {
                "name": "j2",
                "parent": "link1",
                "child": "link2",
                "parent_to_base": identity,
                "child_to_follower": identity,
                "primitives": [{"primitive": "Px", "coordinate": "px"}],
            },
        ],
    }


@pytest.fixture(scope="module")
def compiled_character():
    preset_id = list_character_presets()[0]
    return compile_character(preset_id, {})


@pytest.fixture(scope="module")
def spec_doc(compiled_character):
    return compiled_character.document


@pytest.fixture(scope="module")
def skinned(spec_doc, compiled_character):
    return build_skinned_body(spec_doc, spec_sha256=compiled_character.sha256)


# --- bone topology -----------------------------------------------------


def test_bones_are_bijective_with_spec_bodies(skinned, spec_doc) -> None:
    body_names = {b["name"] for b in spec_doc["bodies"] if b["name"] != "world"}
    assert set(skinned.bone_names) == body_names
    assert len(skinned.bone_names) == len(body_names)


def test_bone_parents_match_joint_tree(skinned, spec_doc) -> None:
    parent_of = {j["child"]: j["parent"] for j in spec_doc["joints"]}
    for i, name in enumerate(skinned.bone_names):
        expected_parent = parent_of[name]
        if expected_parent == "world":
            assert skinned.bone_parents[i] == -1
        else:
            assert skinned.bone_names[skinned.bone_parents[i]] == expected_parent


# --- forward kinematics --------------------------------------------------


def test_zero_pose_matches_zero_pose_joint_positions(spec_doc) -> None:
    poses = spec_body_poses(spec_doc)
    anchors = zero_pose_joint_positions(spec_doc)
    for joint in spec_doc["joints"]:
        base = poses[joint["parent"]] @ np.asarray(joint["parent_to_base"], dtype=float)
        assert np.allclose(base[:3, 3], anchors[joint["name"]], atol=1e-12)


def test_rz_rotates_child_x_axis_to_world_y() -> None:
    spec = _tiny_spec()
    poses = spec_body_poses(spec, {"rz": np.pi / 2})
    rotated_x = poses["link1"][:3, :3] @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(rotated_x, [0.0, 1.0, 0.0], atol=1e-12)


def test_px_translates_along_local_x() -> None:
    spec = _tiny_spec()
    poses = spec_body_poses(spec, {"px": 0.3})
    assert np.allclose(poses["link2"][:3, 3], [0.3, 0.0, 0.0], atol=1e-12)


def test_unknown_coordinate_raises(spec_doc) -> None:
    with pytest.raises(ValueError):
        spec_body_poses(spec_doc, {"not_a_real_coordinate": 0.1})


def test_non_finite_coordinate_raises(spec_doc) -> None:
    coordinate = spec_doc["coordinate_order"][0]
    with pytest.raises(ValueError):
        spec_body_poses(spec_doc, {coordinate: float("nan")})


# --- skinning --------------------------------------------------------------


def test_weights_sum_to_one_and_joints_are_valid(skinned) -> None:
    assert np.allclose(skinned.weights.sum(axis=1), 1.0)
    assert skinned.joints.min() >= 0
    assert skinned.joints.max() < len(skinned.bone_names)


def _primitive_of_coordinate(spec_doc: dict) -> dict[str, str]:
    mapping = {}
    for joint in spec_doc["joints"]:
        for primitive in joint["primitives"]:
            mapping[primitive["coordinate"]] = primitive["primitive"]
    return mapping


def test_attachment_error_within_tolerance(skinned, spec_doc) -> None:
    primitive_of = _primitive_of_coordinate(spec_doc)
    q = {
        name: 0.3 * (-1.0) ** i
        for i, name in enumerate(spec_doc["coordinate_order"])
        if primitive_of[name].startswith("R")
    }
    posed_poses = spec_body_poses(spec_doc, q)
    posed_vertices = pose_skinned_vertices(skinned, posed_poses)

    rest_inv = np.linalg.inv(skinned.rest_poses)
    bone_idx = skinned.vertex_bone
    homogeneous = np.concatenate(
        [skinned.vertices, np.ones((skinned.vertices.shape[0], 1))], axis=1
    )
    local = np.einsum("nij,nj->ni", rest_inv[bone_idx], homogeneous)
    posed_bones = np.stack([posed_poses[name] for name in skinned.bone_names])
    expected = np.einsum("nij,nj->ni", posed_bones[bone_idx], local)[:, :3]

    max_error = float(np.linalg.norm(posed_vertices - expected, axis=1).max())
    assert max_error < 1e-3

    limb_candidates = [
        name for name in skinned.bone_names if classify_body(name) in _LIMB_PARTS
    ]
    assert limb_candidates, "expected at least one limb-classified body"
    limb_bone_idx = skinned.bone_names.index(limb_candidates[0])
    mask = skinned.vertex_bone == limb_bone_idx
    moved = np.linalg.norm(posed_vertices[mask] - skinned.vertices[mask], axis=1).max()
    assert moved > 1e-4


# --- glTF export -------------------------------------------------------


def test_gltf_buffer_decodes_to_the_declared_length(skinned) -> None:
    doc = skinned_body_gltf(skinned)
    prefix = "data:application/octet-stream;base64,"
    uri = doc["buffers"][0]["uri"]
    assert uri.startswith(prefix)
    raw = base64.b64decode(uri[len(prefix) :])
    assert len(raw) == doc["buffers"][0]["byteLength"]


def _accessor_array(doc: dict, index: int, dtype: type, width: int) -> np.ndarray:
    prefix = "data:application/octet-stream;base64,"
    raw = base64.b64decode(doc["buffers"][0]["uri"][len(prefix) :])
    view = doc["bufferViews"][doc["accessors"][index]["bufferView"]]
    chunk = raw[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
    return np.frombuffer(chunk, dtype=dtype).reshape(-1, width)


def test_up_axis_follows_gravity(skinned) -> None:
    assert skinned.up_axis == (0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    "up", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.3, 0.4, -0.5)]
)
def test_gltf_up_rotation_maps_up_onto_y(up) -> None:
    rotation = gltf_up_rotation(up)[:3, :3]
    unit = np.asarray(up) / np.linalg.norm(up)
    assert np.allclose(rotation @ unit, [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_gltf_up_rotation_rejects_zero_axis() -> None:
    with pytest.raises(ValueError):
        gltf_up_rotation((0.0, 0.0, 0.0))


def test_gltf_position_accessor_bounds_and_y_up(skinned) -> None:
    doc = skinned_body_gltf(skinned)
    accessor = doc["accessors"][0]
    expected = skinned.vertices @ gltf_up_rotation(skinned.up_axis)[:3, :3].T
    assert accessor["count"] == skinned.vertices.shape[0]
    assert np.allclose(accessor["min"], expected.min(axis=0), atol=1e-4)
    assert np.allclose(accessor["max"], expected.max(axis=0), atol=1e-4)
    extent = np.subtract(accessor["max"], accessor["min"])
    assert extent[1] == max(extent), "glTF scene must stand upright along +Y"


def test_gltf_skin_driven_by_spec_poses_reproduces_posed_mesh(
    skinned, spec_doc
) -> None:
    """A glTF viewer setting joint globals to R @ spec pose shows the posed skin."""
    doc = skinned_body_gltf(skinned)
    positions = _accessor_array(doc, 0, np.float32, 3).astype(float)
    joints = _accessor_array(doc, 2, np.uint16, 4)
    weights = _accessor_array(doc, 3, np.float32, 4).astype(float)
    inverse_bind = np.transpose(
        _accessor_array(doc, 5, np.float32, 16).reshape(-1, 4, 4), (0, 2, 1)
    ).astype(float)
    to_y_up = gltf_up_rotation(skinned.up_axis)
    q = dict.fromkeys(spec_doc["coordinate_order"][:12], 0.25)
    poses = spec_body_poses(spec_doc, q)
    joint_globals = np.stack([to_y_up @ poses[n] for n in skinned.bone_names])
    homogeneous = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
    skinned_gltf = np.zeros((len(positions), 3))
    for k in range(4):
        skin = joint_globals[joints[:, k]] @ inverse_bind[joints[:, k]]
        moved = np.einsum("nij,nj->ni", skin, homogeneous)[:, :3]
        skinned_gltf += weights[:, k : k + 1] * moved
    expected = pose_skinned_vertices(skinned, poses) @ to_y_up[:3, :3].T
    assert np.abs(skinned_gltf - expected).max() < 1e-4


def test_gltf_node_names_match_bones_and_skin_joint_count(skinned) -> None:
    doc = skinned_body_gltf(skinned)
    bone_count = len(skinned.bone_names)
    bone_nodes = doc["nodes"][:bone_count]
    assert [n["name"] for n in bone_nodes] == list(skinned.bone_names)
    assert doc["nodes"][bone_count] == {"mesh": 0, "skin": 0}
    assert len(doc["skins"][0]["joints"]) == bone_count


def test_gltf_inverse_bind_matrices_cancel_node_global_matrix(skinned) -> None:
    doc = skinned_body_gltf(skinned)
    prefix = "data:application/octet-stream;base64,"
    raw = base64.b64decode(doc["buffers"][0]["uri"][len(prefix) :])
    ibm_accessor = doc["accessors"][5]
    ibm_view = doc["bufferViews"][ibm_accessor["bufferView"]]
    ibm_bytes = raw[
        ibm_view["byteOffset"] : ibm_view["byteOffset"] + ibm_view["byteLength"]
    ]
    inverse_bind = np.frombuffer(ibm_bytes, dtype=np.float32).reshape(-1, 4, 4)
    inverse_bind = np.transpose(inverse_bind, (0, 2, 1))  # column-major -> row-major

    locals_ = [
        _trs_matrix(node["translation"], node["rotation"])
        for node in doc["nodes"][: len(skinned.bone_names)]
    ]
    for i, global_matrix in enumerate(_compose_globals(skinned.bone_parents, locals_)):
        product = inverse_bind[i].astype(float) @ global_matrix
        assert np.allclose(product, np.eye(4), atol=1e-4)


# --- glTF animation driven by spec q (slice 2) ----------------------------


def _quaternion_matrix(xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(xyzw, dtype=float) / np.linalg.norm(xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _trs_matrix(translation, rotation) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _quaternion_matrix(rotation)
    m[:3, 3] = translation
    return m


def _compose_globals(parents, locals_) -> list[np.ndarray]:
    globals_: list[np.ndarray] = []
    for i, local in enumerate(locals_):  # parents precede children
        globals_.append(local if parents[i] == -1 else globals_[parents[i]] @ local)
    return globals_


def _swing_frames(spec_doc: dict, n_frames: int) -> list[dict[str, float]]:
    primitive_of = _primitive_of_coordinate(spec_doc)
    rotational = [n for n in spec_doc["coordinate_order"] if primitive_of[n][0] == "R"]
    return [
        {name: 0.6 * np.sin(0.4 * f + i) for i, name in enumerate(rotational)}
        for f in range(n_frames)
    ]


@pytest.fixture(scope="module")
def animated(skinned, spec_doc):
    frames = _swing_frames(spec_doc, 5)
    motion = spec_motion_bone_poses(skinned, spec_doc, frames)
    return frames, motion, skinned_body_gltf(skinned, motion, fps=30.0)


@pytest.mark.parametrize("seed", range(6))
def test_quaternion_round_trips_random_rotations(seed) -> None:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation = q * np.sign(np.linalg.det(q))
    quat = quaternion_xyzw(rotation)
    assert quat[3] >= 0
    assert np.allclose(_quaternion_matrix(quat), rotation, atol=1e-12)


def test_quaternion_handles_half_turns() -> None:
    for rotation in (np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0])):
        assert np.allclose(_quaternion_matrix(quaternion_xyzw(rotation)), rotation)


def test_quaternion_rejects_reflection() -> None:
    with pytest.raises(ValueError):
        quaternion_xyzw(np.diag([1.0, 1.0, -1.0]))


def test_static_export_has_no_animation(skinned) -> None:
    assert "animations" not in skinned_body_gltf(skinned)


def test_motion_poses_match_spec_fk(skinned, spec_doc) -> None:
    frames = _swing_frames(spec_doc, 2)
    motion = spec_motion_bone_poses(skinned, spec_doc, frames)
    assert motion.shape == (2, len(skinned.bone_names), 4, 4)
    poses = spec_body_poses(spec_doc, frames[1])
    for b, name in enumerate(skinned.bone_names):
        assert np.allclose(motion[1, b], poses[name], atol=1e-12)


def test_animation_channels_cover_every_bone(skinned, animated) -> None:
    _, _, doc = animated
    (animation,) = doc["animations"]
    targets = {
        (c["target"]["node"], c["target"]["path"]) for c in animation["channels"]
    }
    expected = {
        (b, p)
        for b in range(len(skinned.bone_names))
        for p in ("translation", "rotation")
    }
    assert targets == expected
    times = _accessor_array(doc, animation["samplers"][0]["input"], np.float32, 1)
    assert np.allclose(times[:, 0], np.arange(5) / 30.0, atol=1e-7)
    accessor = doc["accessors"][animation["samplers"][0]["input"]]
    assert accessor["min"] == [0.0] and np.isclose(accessor["max"][0], 4 / 30.0)


def _keyframe_globals(doc: dict, parents, frame: int) -> list[np.ndarray]:
    (animation,) = doc["animations"]
    trs: dict[int, dict[str, np.ndarray]] = {}
    for channel in animation["channels"]:
        sampler = animation["samplers"][channel["sampler"]]
        width = 3 if channel["target"]["path"] == "translation" else 4
        values = _accessor_array(doc, sampler["output"], np.float32, width)
        trs.setdefault(channel["target"]["node"], {})[channel["target"]["path"]] = (
            values[frame].astype(float)
        )
    locals_ = [
        _trs_matrix(trs[b]["translation"], trs[b]["rotation"])
        for b in range(len(parents))
    ]
    return _compose_globals(parents, locals_)


def test_animation_keys_reproduce_spec_fk_and_skin(skinned, animated) -> None:
    """A viewer playing the animation shows the spec-FK pose and posed skin."""
    frames, motion, doc = animated
    to_y_up = gltf_up_rotation(skinned.up_axis)
    positions = _accessor_array(doc, 0, np.float32, 3).astype(float)
    inverse_bind = np.transpose(
        _accessor_array(doc, 5, np.float32, 16).reshape(-1, 4, 4), (0, 2, 1)
    ).astype(float)
    bone = skinned.vertex_bone
    homogeneous = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
    for f in (0, len(frames) - 1):
        globals_ = np.stack(_keyframe_globals(doc, skinned.bone_parents, f))
        assert np.abs(globals_ - to_y_up @ motion[f]).max() < 1e-5
        skin = globals_[bone] @ inverse_bind[bone]
        viewer = np.einsum("nij,nj->ni", skin, homogeneous)[:, :3]
        posed = dict(zip(skinned.bone_names, motion[f], strict=True))
        expected = pose_skinned_vertices(skinned, posed) @ to_y_up[:3, :3].T
        assert np.abs(viewer - expected).max() < 1e-4


def test_animation_rotations_stay_in_one_hemisphere(skinned, animated) -> None:
    _, _, doc = animated
    (animation,) = doc["animations"]
    for channel in animation["channels"]:
        if channel["target"]["path"] != "rotation":
            continue
        sampler = animation["samplers"][channel["sampler"]]
        quats = _accessor_array(doc, sampler["output"], np.float32, 4)
        assert (np.einsum("ij,ij->i", quats[1:], quats[:-1]) > 0).all()


def test_motion_validation(skinned, spec_doc) -> None:
    with pytest.raises(ValueError):
        spec_motion_bone_poses(skinned, spec_doc, [])
    motion = spec_motion_bone_poses(skinned, spec_doc, [{}])
    with pytest.raises(ValueError):
        skinned_body_gltf(skinned, motion[:, :-1])
    with pytest.raises(ValueError):
        skinned_body_gltf(skinned, motion, fps=0.0)
    bad = motion.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        skinned_body_gltf(skinned, bad)


def test_gltf_text_round_trips_through_json(skinned) -> None:
    text = skinned_body_gltf_text(skinned)
    doc = json.loads(text)
    assert doc["asset"]["version"] == "2.0"
    assert doc["extras"]["spec_sha256"] == skinned.spec_sha256


# --- export_character("gltf") integration --------------------------------


def test_export_character_gltf(compiled_character) -> None:
    text, media_type, filename = export_character(compiled_character, "gltf")
    assert filename.endswith(".gltf")
    assert media_type == "model/gltf+json"
    doc = json.loads(text)
    assert doc["asset"]["version"] == "2.0"
    assert doc["extras"]["spec_sha256"] == compiled_character.sha256
