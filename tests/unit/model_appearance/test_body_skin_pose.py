"""Skinned shoulder of the CC0 body skin on the girdle poses (#15299)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.model_appearance import body_skin_fit as fit
from src.shared.python.model_appearance import body_skin_pose as pose
from tests.unit.model_appearance.test_body_skin_fit import _subject

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "model_appearance"
GIRDLE = json.loads((FIXTURE / "girdle_poses.json").read_text())["poses"]
TURNS = json.loads((FIXTURE / "shoulder_turns.json").read_text())["turns"]
RAW = json.loads((fit.ASSET_DIR / "pose_settings.json").read_text())
SETTINGS = pose.load_pose_settings()
SKIN = fit.load_body_skin()
FITTED = fit.fit_body_skin(SKIN, _subject(), fit.load_fit_settings())
HUB = 0.5 * (FITTED.joints["shoulder_l"] + FITTED.joints["shoulder_r"])
EDGES = pose.acromion_edges(SKIN, FITTED, "r", SETTINGS)
STILL = pose.IDENTITY


def _girdle(angles: dict[str, float]) -> pose.Rigid:
    """The trail girdle link turned about the trunk: forward, then down.

    The fixture's signs are the model's joint conventions; here the magnitudes
    are applied so that the right shoulder joint moves forward and down.
    """
    turn = [abs(angles["RScapInputX"]), abs(angles["RScapInputY"])]
    return pose.Rigid.about(Rotation.from_euler("xz", turn).as_matrix(), HUB)


def _posed(weights: np.ndarray, girdle: pose.Rigid) -> np.ndarray:
    motions = {"thorax": STILL, "girdle_l": STILL, "upper_arm_l": STILL}
    motions |= {"girdle_r": girdle, "upper_arm_r": girdle}
    moves = pose.shoulder_transforms(motions, FITTED, SETTINGS)
    return pose.pose_skin(FITTED.vertices, weights, SKIN.groups, moves)


def _step_mm(weights: np.ndarray, girdle: pose.Rigid) -> float:
    change = pose.edge_length_change(FITTED.vertices, _posed(weights, girdle), EDGES)
    return 1000.0 * float(change.max())


@pytest.mark.parametrize("name", sorted(GIRDLE))
def test_no_step_between_shirt_and_sleeve_cap_on_the_girdle_poses(name: str) -> None:
    girdle = _girdle(GIRDLE[name])
    moved = girdle.apply(FITTED.joints["shoulder_r"]) - FITTED.joints["shoulder_r"]
    assert moved[0] > 0.04 and moved[2] <= 1e-9  # forward, never up
    smooth = pose.smooth_shoulder_weights(SKIN, FITTED, SETTINGS)
    one_body = np.zeros_like(SKIN.weights)
    one_body[np.arange(len(one_body)), SKIN.weights.argmax(axis=1)] = 1.0
    assert len(EDGES) > 50
    assert _step_mm(smooth, girdle) <= RAW["step_limits_mm"][name]
    assert _step_mm(smooth, girdle) < _step_mm(SKIN.weights, girdle)
    assert _step_mm(one_body, girdle) >= RAW["rigid_step_floor_mm"]


def _turn_stretch_mm(settings: pose.PoseSettings, name: str) -> float:
    """Largest change of an edge length round a shoulder whose arm is turned."""
    side, joint = TURNS[name]["side"], FITTED.joints[f"shoulder_{TURNS[name]['side']}"]
    turn = Rotation.from_rotvec(TURNS[name]["rotvec_rad"]).as_matrix()
    motions = dict.fromkeys(("thorax", "girdle_l", "girdle_r"), STILL)
    motions |= {"upper_arm_l": STILL, "upper_arm_r": STILL}
    motions[f"upper_arm_{side}"] = pose.Rigid.about(turn, joint)
    moves = pose.shoulder_transforms(motions, FITTED, settings)
    weights = pose.smooth_shoulder_weights(SKIN, FITTED, settings)
    posed = pose.pose_skin(FITTED.vertices, weights, SKIN.groups, moves)
    near = np.linalg.norm(FITTED.vertices - joint, axis=1)
    edges = pose.mesh_edges(SKIN.faces)
    edges = edges[(near[edges] < RAW["arm_turn_patch_radius_m"]).all(axis=1)]
    assert len(edges) > 500
    return 1000.0 * float(pose.edge_length_change(FITTED.vertices, posed, edges).max())


@pytest.mark.parametrize("name", sorted(TURNS))
def test_a_turned_arm_stretches_the_shoulder_less_with_the_girdle_passes(
    name: str,
) -> None:
    assert SETTINGS.girdle_smoothing_iterations > 0
    plain = dataclasses.replace(SETTINGS, girdle_smoothing_iterations=0)
    stretch = _turn_stretch_mm(SETTINGS, name)
    assert stretch <= RAW["arm_turn_stretch_limits_mm"][name]
    assert stretch < _turn_stretch_mm(plain, name)


def test_the_girdle_passes_leave_the_other_groups_alone() -> None:
    plain = dataclasses.replace(SETTINGS, girdle_smoothing_iterations=0)
    before = pose.smooth_shoulder_weights(SKIN, FITTED, plain)
    after = pose.smooth_shoulder_weights(SKIN, FITTED, SETTINGS)
    girdle = np.array(
        [
            g == "thorax" or g.rpartition("_")[0] in pose.GIRDLE_GROUPS
            for g in SKIN.groups
        ]
    )
    assert np.allclose(after[:, ~girdle], before[:, ~girdle], atol=1e-12)
    assert np.allclose(after[:, girdle].sum(axis=1), before[:, girdle].sum(axis=1))
    assert not np.allclose(after[:, girdle], before[:, girdle], atol=1e-3)


def test_smoothing_is_local_and_keeps_the_weights_normalised() -> None:
    smooth = pose.smooth_shoulder_weights(SKIN, FITTED, SETTINGS)
    assert np.allclose(smooth.sum(axis=1), 1.0) and smooth.min() >= 0.0
    far = FITTED.vertices[:, 2] < FITTED.joints["hip_l"][2]
    assert np.allclose(smooth[far], SKIN.weights[far], atol=1e-12)
    assert not np.array_equal(smooth, SKIN.weights)
    none = dataclasses.replace(
        SETTINGS, smoothing_iterations=0, girdle_smoothing_iterations=0
    )
    assert np.allclose(pose.smooth_shoulder_weights(SKIN, FITTED, none), SKIN.weights)


def test_a_common_motion_moves_the_surface_rigidly() -> None:
    turn = pose.Rigid.about(Rotation.from_euler("y", 0.4).as_matrix(), HUB)
    posed = pose.pose_skin(
        FITTED.vertices, SKIN.weights, SKIN.groups, dict.fromkeys(SKIN.groups, turn)
    )
    assert np.allclose(posed, turn.apply(FITTED.vertices), atol=1e-12)
    rest = pose.pose_skin(FITTED.vertices, SKIN.weights, SKIN.groups, {})
    assert np.allclose(rest, FITTED.vertices, atol=1e-12)


def test_the_arm_stays_on_the_shoulder_joint_and_the_other_side_is_still() -> None:
    girdle = _girdle(GIRDLE["trail_protracted_110_mm"])
    motions = {"thorax": STILL, "girdle_l": STILL, "upper_arm_l": STILL}
    motions |= {"girdle_r": girdle, "upper_arm_r": girdle}
    moves = pose.shoulder_transforms(motions, FITTED, SETTINGS)
    joint = FITTED.joints["shoulder_r"]
    for group in (
        "clavicle_r",
        "shoulder_r",
        "upperarm_proximal_r",
        "forearm_distal_r",
    ):
        assert np.allclose(moves[group].apply(joint), girdle.apply(joint), atol=1e-12)
    posed = _posed(SKIN.weights, girdle)
    left = FITTED.vertices[:, 1] > SETTINGS.smoothing_radius_m
    assert np.allclose(posed[left], FITTED.vertices[left], atol=1e-12)


@pytest.mark.parametrize(
    "change",
    [
        {"shoulder_follow": 1.5},
        {"smoothing_iterations": -1},
        {"girdle_smoothing_iterations": -1},
        {"smoothing_radius_m": 0.0},
        {"acromion_half_width_m": -0.01},
        {"acromion_inner_m": -0.01},
    ],
)
def test_invalid_settings_raise(change: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(SETTINGS, **change)  # type: ignore[arg-type]


def test_missing_motion_and_bad_weights_raise() -> None:
    with pytest.raises(ValueError, match="girdle_l"):
        pose.shoulder_transforms({"thorax": STILL}, FITTED, SETTINGS)
    with pytest.raises(ValueError):
        pose.pose_skin(FITTED.vertices, SKIN.weights[:, :3], SKIN.groups, {})


NECK = FITTED.joints["neck"]
NECK_EDGES = pose.mesh_edges(SKIN.faces)
NECK_EDGES = NECK_EDGES[(FITTED.vertices[NECK_EDGES, 2] > NECK[2]).all(axis=1)]


def _head_turned(settings: pose.PoseSettings, head: pose.Rigid | None) -> np.ndarray:
    motions = {"thorax": STILL} | ({} if head is None else {"head": head})
    for side in ("l", "r"):
        motions |= {f"girdle_{side}": STILL, f"upper_arm_{side}": STILL}
    moves = pose.shoulder_transforms(motions, FITTED, settings)
    return pose.pose_skin(FITTED.vertices, SKIN.weights, SKIN.groups, moves)


_SIDES = np.vstack([SKIN.faces[:, pair] for pair in ([0, 1], [1, 2], [2, 0])])
_OPEN, _COUNT = np.unique(np.sort(_SIDES, axis=1), axis=0, return_counts=True)
#: Vertices of the open rim where the neck meets the separate head model.
NECK_RIM = np.unique(_OPEN[_COUNT == 1])
NECK_RIM = NECK_RIM[FITTED.vertices[NECK_RIM, 2] > NECK[2]]


def _rim_lag_mm(posed: np.ndarray, head: pose.Rigid) -> float:
    """How far the rim falls behind the head it has to meet."""
    lag = posed[NECK_RIM] - head.apply(FITTED.vertices[NECK_RIM])
    return 1000.0 * float(np.linalg.norm(lag, axis=1).max())


def _neck_step_mm(posed: np.ndarray) -> float:
    change = pose.edge_length_change(FITTED.vertices, posed, NECK_EDGES)
    return 1000.0 * float(change.max())


def test_the_neck_shares_a_head_turn_between_thorax_and_head() -> None:
    turn = RAW["neck_test_turn_rad"]
    head = pose.Rigid.about(Rotation.from_euler("zy", [turn, turn]).as_matrix(), NECK)
    posed = _head_turned(SETTINGS, head)
    assert np.allclose(_head_turned(SETTINGS, None), FITTED.vertices, atol=1e-12)
    below = FITTED.vertices[:, 2] < FITTED.joints["shoulder_l"][2] - 0.1
    assert np.allclose(posed[below], FITTED.vertices[below], atol=1e-12)
    stiff = dataclasses.replace(SETTINGS, neck_follow=(0.0, 0.0, 0.0))
    assert len(NECK_EDGES) > 100 and len(NECK_RIM) > 20
    assert _neck_step_mm(posed) <= RAW["neck_step_limit_mm"]
    lag = _rim_lag_mm(posed, head)
    assert lag <= RAW["neck_rim_lag_limit_mm"]
    assert lag < 0.75 * _rim_lag_mm(_head_turned(stiff, head), head)


@pytest.mark.parametrize("follow", [(0.5, 0.2, 0.8), (0.2, 0.5, 1.2), (0.2, 0.5)])
def test_invalid_neck_follow_raises(follow: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="neck_follow"):
        dataclasses.replace(SETTINGS, neck_follow=follow)  # type: ignore[arg-type]
