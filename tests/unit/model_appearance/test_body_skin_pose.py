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


def test_smoothing_is_local_and_keeps_the_weights_normalised() -> None:
    smooth = pose.smooth_shoulder_weights(SKIN, FITTED, SETTINGS)
    assert np.allclose(smooth.sum(axis=1), 1.0) and smooth.min() >= 0.0
    far = FITTED.vertices[:, 2] < FITTED.joints["hip_l"][2]
    assert np.allclose(smooth[far], SKIN.weights[far], atol=1e-12)
    assert not np.array_equal(smooth, SKIN.weights)
    none = dataclasses.replace(SETTINGS, smoothing_iterations=0)
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
