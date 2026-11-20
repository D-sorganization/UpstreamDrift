"""Bones of the CC0 body skin: group-to-body shares and bind poses (#15299)."""

from __future__ import annotations

import dataclasses

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
