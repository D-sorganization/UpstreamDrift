"""Bones of the CC0 body skin: group-to-body shares and bind poses (#15299)."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.model_appearance import body_skin_export as export
from src.shared.python.model_appearance import body_skin_fit as fit
from src.shared.python.model_appearance import body_skin_pose as pose

pytestmark = pytest.mark.unit

SKIN = fit.load_body_skin()
POSE = pose.load_pose_settings()
SETTINGS = export.load_export_settings()
WRIST = json.loads((fit.ASSET_DIR / "export_settings.json").read_text())["wrist_limits"]
#: Body origins (metres) of a 1.71 m golfer with the arms hanging; left side.
CENTRAL = {"pelvis": (0, 0, 0), "thorax": (0, 0, 0.138), "head": (0, 0, 0.601)}
LEFT = {
    "girdle": (0, 0, 0.561),
    "upper_arm": (0, 0.195, 0.561),
    "elbow": (0, 0.195, 0.256),
    "forearm": (0, 0.195, 0.111),
    "hand": (0, 0.195, -0.034),
    "thigh": (0, 0.089, 0),
    "shank": (-0.008, 0.086, -0.4),
    "foot": (-0.01, 0.088, -0.83),
}
RESIDUAL_M, BONES_OF_THE_BODY = 1.0e-9, 15  # pelvis, thorax, head, six per side
#: Each skin knee, ankle, elbow and wrist on its own model joint: a micrometre.
SIDE_RESIDUAL_M = 1.0e-6


def _frames(turn: np.ndarray | None = None) -> dict[str, pose.Rigid]:
    """Body frames at the bind pose, each turned about its origin by ``turn``."""
    rotation = np.eye(3) if turn is None else turn
    out = {k: pose.Rigid(rotation, np.array(p, float)) for k, p in CENTRAL.items()}
    for side, sign in fit.SIDES:
        for role, (x, y, z) in LEFT.items():
            out[f"{role}_{side}"] = pose.Rigid(rotation, np.array([x, sign * y, z]))
    return out


FRAMES = _frames()
LENGTHS = export.subject_lengths(FRAMES)
FITTED = fit.fit_body_skin(SKIN, LENGTHS, fit.load_fit_settings())
SHARES = export.group_shares(SKIN.groups, POSE, SETTINGS)


def test_subject_lengths_are_the_distances_between_body_origins() -> None:
    assert LENGTHS.hip_half_spacing == pytest.approx(0.089)
    assert LENGTHS.shoulder_half_spacing == pytest.approx(0.195)
    assert LENGTHS.shoulder_height == pytest.approx(0.561)
    assert LENGTHS.neck_height == pytest.approx(0.601)
    assert LENGTHS.upper_arm == pytest.approx(0.305)
    assert LENGTHS.forearm == pytest.approx(0.290)
    assert LENGTHS.shank == pytest.approx(np.linalg.norm([0.002, 0.002, 0.43]))


def test_every_group_is_shared_out_whole() -> None:
    assert set(SHARES) == set(SKIN.groups)
    for group, shares in SHARES.items():
        assert sum(shares.values()) == pytest.approx(1.0), group
    follow = POSE.shoulder_follow
    assert SHARES["shoulder_l"] == pytest.approx(
        {"girdle_l": 1.0 - follow, "upper_arm_l": follow}
    )
    low = SETTINGS.spine_follow[0]
    assert SHARES["spine05"] == pytest.approx({"pelvis": 1.0 - low, "thorax": low})
    assert SHARES["neck_upper"]["head"] == pytest.approx(POSE.neck_follow[-1])
    assert SHARES["forearm_proximal_r"] == {"elbow_r": 1.0}


def test_an_unknown_group_or_missing_body_is_refused() -> None:
    with pytest.raises(ValueError, match="tail"):
        export.group_shares(("pelvis", "tail"), POSE, SETTINGS)
    partial = {k: v for k, v in FRAMES.items() if k != "shank_r"}
    with pytest.raises(ValueError, match="shank_r"):
        export.subject_lengths(partial)
    with pytest.raises(ValueError, match="spine_follow"):
        dataclasses.replace(SETTINGS, spine_follow=(0.8, 0.1, 0.5))


@pytest.mark.parametrize("turn", [None, Rotation.from_rotvec([0.3, -0.2, 0.5])])
def test_each_bone_carries_its_fitted_joints_onto_the_model(turn: object) -> None:
    rotation = None if turn is None else turn.as_matrix()  # type: ignore[attr-defined]
    frames = _frames(rotation)
    binds = export.bind_poses(frames, FITTED)
    model = export.model_joints(frames)
    rest = export.rest_joints(FITTED, model)
    assert len(binds) == BONES_OF_THE_BODY
    for role, bind in binds.items():
        for joint in export.bone_joints(role):
            carried = frames[role].apply(bind.inverse().apply(rest[joint]))
            assert np.linalg.norm(carried - model[joint]) < 1.0e-3, (role, joint)
        near = export.bone_joints(role)[0]
        carried = frames[role].apply(bind.inverse().apply(rest[near]))
        assert np.linalg.norm(carried - model[near]) < RESIDUAL_M, role


def test_each_side_is_fitted_to_its_own_limb_lengths() -> None:
    frames = dict(FRAMES)
    for role, shift in (("shank_r", 0.012), ("foot_r", 0.005), ("hand_l", -0.007)):
        frames[role] = pose.Rigid(np.eye(3), FRAMES[role].translation + [0, 0, shift])
    lengths = export.subject_lengths(frames)
    assert lengths.limb("thigh", "r") == pytest.approx(lengths.thigh - 0.006, abs=1e-4)
    fitted = fit.fit_body_skin(SKIN, lengths, fit.load_fit_settings())
    model = export.model_joints(frames)
    rest = export.rest_joints(fitted, model)
    for role, bind in export.bind_poses(frames, fitted).items():
        for joint in export.bone_joints(role):
            carried = frames[role].apply(bind.inverse().apply(rest[joint]))
            assert np.linalg.norm(carried - model[joint]) < SIDE_RESIDUAL_M, joint


def test_bones_keep_every_vertex_whole_and_drop_small_weights() -> None:
    binds = export.bind_poses(FRAMES, FITTED)
    bones = export.skin_bones(SKIN.weights, SKIN.groups, SHARES, binds, SETTINGS)
    assert [bone.role for bone in bones] == list(binds)
    total = np.zeros(len(SKIN.vertices))
    for bone in bones:
        assert bone.weights.min() >= SETTINGS.weight_floor
        np.add.at(total, bone.vertex_ids, bone.weights)
    assert np.allclose(total, 1.0)
    with pytest.raises(ValueError, match="groups"):
        export.skin_bones(SKIN.weights[:, :3], SKIN.groups, SHARES, binds, SETTINGS)


def test_the_skin_at_the_bind_pose_spans_the_model_joints() -> None:
    binds = export.bind_poses(FRAMES, FITTED)
    bones = export.skin_bones(SKIN.weights, SKIN.groups, SHARES, binds, SETTINGS)
    posed = export.pose_bones(FITTED.vertices, bones, FRAMES)
    model = export.model_joints(FRAMES)
    edges = pose.mesh_edges(SKIN.faces)
    change = pose.edge_length_change(FITTED.vertices, posed, edges)
    assert np.abs(change).max() < SETTINGS.bind_stretch_limit_m
    for side, sign in fit.SIDES:  # the forearm skin surrounds the model forearm
        ends = [f"forearm_{end}_{side}" for end in ("proximal", "distal")]
        arm = SKIN.weights[:, [SKIN.groups.index(g) for g in ends]].sum(axis=1) > 0.9
        centre = posed[arm].mean(axis=0)
        assert sign * centre[1] == pytest.approx(0.195, abs=0.02)
        assert model[f"wrist_{side}"][2] < centre[2] < model[f"elbow_{side}"][2]


def _end_ring_shift(settings: export.ExportSettings) -> float:
    """Largest distance (m) of a forearm's end off the place its hand carries
    it to, over both sides and the wrist bent four ways."""
    binds = export.bind_poses(FRAMES, FITTED)
    bones = export.skin_bones(SKIN.weights, SKIN.groups, SHARES, binds, settings)
    hands = {
        side: pose.Rigid(np.eye(3), FITTED.joints[f"wrist_{side}"])
        for side, _ in fit.SIDES
    }
    blended = export.blend_wrists(bones, FITTED.vertices, binds, hands, settings)
    if settings.wrist_blend_length_m == 0.0:
        assert blended == bones  # off: the bones of the skin as they were
    rest = {**binds, **{f"hand_{side}": hand for side, hand in hands.items()}}
    assert np.allclose(
        export.pose_bones(FITTED.vertices, blended, rest), FITTED.vertices
    )
    bend, worst = np.radians(WRIST["bend_deg"]), 0.0
    for side, hand in hands.items():
        wrist = hand.translation
        bone = wrist - FITTED.joints[f"elbow_{side}"]
        ids = next(b for b in bones if b.role == f"forearm_{side}").vertex_ids
        along = (FITTED.vertices[ids] - wrist) @ bone / np.linalg.norm(bone)
        ring = ids[along > -SETTINGS.wrist_blend_full_m]  # the end, held whole
        assert len(ring) > 10
        total = np.zeros(len(SKIN.vertices))
        for one in blended:
            np.add.at(total, one.vertex_ids, one.weights)
        assert np.allclose(total, 1.0)
        for rotvec in ((bend, 0, 0), (-bend, 0, 0), (0, bend, 0), (0, -bend, 0)):
            turn = Rotation.from_rotvec(rotvec).as_matrix()
            frames = {**rest, f"hand_{side}": pose.Rigid(turn, wrist)}
            posed = export.pose_bones(FITTED.vertices, blended, frames)
            carried = wrist + (FITTED.vertices[ring] - wrist) @ turn.T
            shift = np.linalg.norm(posed[ring] - carried, axis=1).max()
            worst = max(worst, float(shift))
            moved = np.linalg.norm(posed - FITTED.vertices, axis=1)
            far = (FITTED.vertices - wrist) @ bone / np.linalg.norm(bone)
            assert moved[far < -settings.wrist_blend_length_m - 1e-9].max() < 1e-12
    return worst


def test_the_forearm_end_follows_a_bent_hand() -> None:
    assert _end_ring_shift(SETTINGS) <= WRIST["end_shift_m"]
    off = dataclasses.replace(SETTINGS, wrist_blend_length_m=0.0)
    assert _end_ring_shift(off) > 10.0 * WRIST["end_shift_m"]
    for length, full in ((-0.01, 0.0), (0.05, 0.05), (0.05, -0.01)):
        with pytest.raises(ValueError, match="wrist_blend"):
            dataclasses.replace(
                SETTINGS, wrist_blend_length_m=length, wrist_blend_full_m=full
            )
    with pytest.raises(ValueError, match="side"):
        binds = export.bind_poses(FRAMES, FITTED)
        export.blend_wrists((), FITTED.vertices, binds, {"x": binds["head"]}, SETTINGS)


def test_the_head_base_is_held_round_the_neck_axis() -> None:
    height, side = SETTINGS.head_base_height_m, SETTINGS.head_base_side_m
    front, back = SETTINGS.head_base_front_m, SETTINGS.head_base_back_m
    drop = SETTINGS.head_base_drop_m
    assert height > 0.0 and front >= side > 0.0 and back > 0.0 and drop > 0.0
    ring = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    wide = np.column_stack([3.0 * front * np.cos(ring), 3.0 * side * np.sin(ring)])
    low = np.column_stack([wide, np.full(16, 0.5 * height)])
    inner = np.column_stack([0.1 * wide, np.full(16, -0.01)])
    high = np.column_stack([wide, np.full(16, height)])
    held = export.narrow_head_base(np.vstack([low, inner, high]), SETTINGS)
    depth = np.where(wide[:, 0] > 0.0, front, back)  # forward is +x
    on_limit = np.hypot(held[:16, 0] / depth, held[:16, 1] / side)
    assert np.allclose(on_limit, 1.0)  # an ellipse each way, straight to the axis
    turned = held[:16, 0] * wide[:, 1] - held[:16, 1] * wide[:, 0]
    assert np.allclose(turned, 0.0, atol=1e-12)
    assert (np.einsum("ij,ij->i", held[:16, :2], wide) > 0.0).all()
    assert np.allclose(held[[0, 4, 8], :2], [[front, 0], [0, side], [-back, 0]])
    # the base is drawn down the neck: by half the drop at half the height,
    # by all of it on and under the joint plane, sideways position kept
    assert np.allclose(held[:16, 2], low[:, 2] - 0.5 * drop)
    assert np.allclose(held[16:32, 2], inner[:, 2] - drop)
    assert np.array_equal(held[16:32, :2], inner[:, :2])
    assert np.array_equal(held[32:], high)
    flat = dataclasses.replace(SETTINGS, head_base_drop_m=0.0)
    assert np.array_equal(export.narrow_head_base(inner, flat), inner)
    off = dataclasses.replace(SETTINGS, head_base_height_m=0.0)
    assert np.array_equal(export.narrow_head_base(inner, off), inner)
    with pytest.raises(ValueError, match="vertices"):
        export.narrow_head_base(low[:, :2], SETTINGS)
    names = ("height", "front", "side", "back", "drop")
    for name, bad in zip(names, (-0.01, 0.0, 0.0, -0.03, -0.001), strict=True):
        with pytest.raises(ValueError, match="head_base"):
            dataclasses.replace(SETTINGS, **{f"head_base_{name}_m": bad})
    with pytest.raises(TypeError, match="head_base"):
        dataclasses.replace(SETTINGS, head_base_side_m="0.03")
