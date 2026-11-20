"""Runtime fit of the CC0 body skin to a subject's joint centres (#15299)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import body_skin_fit as fit

pytestmark = pytest.mark.unit

REFERENCE = json.loads((fit.ASSET_DIR / "skin_settings.json").read_text())[
    "reference_fit"
]["segments_mm"]
#: Hip midpoint to the shoulder line and to the neck joint of the same export.
TRUNK_MM = {"shoulder_height": 528.0, "neck_height": 623.0}
RESIDUAL_M = 1.0e-4  # the fit is exact by construction; 0.1 mm is the contract
SEGMENTS = (
    ("thigh", "hip", "knee", "thigh"),
    ("shank", "knee", "ankle", "shank"),
    ("upperarm", "shoulder", "elbow", "upper_arm"),
    ("forearm", "elbow", "wrist", "forearm"),
)


def _subject() -> fit.SegmentLengths:
    mm = {name: row[2] for name, row in REFERENCE.items()} | TRUNK_MM
    return fit.SegmentLengths(
        hip_half_spacing=mm["hip_spacing"] / 2000.0,
        shoulder_half_spacing=mm["shoulder_spacing"] / 2000.0,
        shoulder_height=mm["shoulder_height"] / 1000.0,
        neck_height=mm["neck_height"] / 1000.0,
        upper_arm=mm["upper_arm"] / 1000.0,
        forearm=mm["forearm"] / 1000.0,
        thigh=mm["thigh"] / 1000.0,
        shank=mm["shank"] / 1000.0,
    )


SKIN = fit.load_body_skin()
PLAIN = dataclasses.replace(
    fit.load_fit_settings(), hip_back_m=0.0, shoulder_anchor_m=(0.0, 0.0, 0.0)
)


def _assert_subject_lengths(joints: dict[str, np.ndarray]) -> None:
    subject = _subject()
    hip_mid = 0.5 * (joints["hip_l"] + joints["hip_r"])
    shoulder_mid = 0.5 * (joints["shoulder_l"] + joints["shoulder_r"])
    got = {
        "hip_half_spacing": np.linalg.norm(joints["hip_l"] - joints["hip_r"]) / 2,
        "shoulder_half_spacing": np.linalg.norm(joints["shoulder_l"] - shoulder_mid),
        "shoulder_height": np.linalg.norm(shoulder_mid - hip_mid),
        "neck_height": np.linalg.norm(joints["neck"] - hip_mid),
    }
    for side in ("l", "r"):
        for _, near, far, name in SEGMENTS:
            got[f"{name}_{side}"] = np.linalg.norm(
                joints[f"{far}_{side}"] - joints[f"{near}_{side}"]
            )
    for name, value in got.items():
        want = getattr(subject, name.removesuffix("_l").removesuffix("_r"))
        assert abs(value - want) <= RESIDUAL_M, name


def test_fitted_joints_have_the_subject_segment_lengths() -> None:
    _assert_subject_lengths(
        fit.fit_body_skin(SKIN, _subject(), fit.load_fit_settings()).joints
    )


def test_each_segment_carries_its_landmarks_onto_the_joints() -> None:
    fitted = fit.fit_body_skin(SKIN, _subject(), PLAIN)
    marks, joints = SKIN.landmarks, fitted.joints
    pairs = [("thorax", "neck")]
    for side in ("l", "r"):
        pairs.append((f"shoulder_{side}", f"shoulder_{side}"))
        for group, near, far, _ in SEGMENTS:
            for half in ("proximal", "distal") if group != "shank" else ("",):
                name = f"{group}_{half}_{side}".replace("__", "_")
                pairs += [(name, f"{near}_{side}"), (name, f"{far}_{side}")]
    for group, mark in pairs:
        image = fitted.map_points(group, marks[mark][None])[0]
        assert np.linalg.norm(image - joints[mark]) <= RESIDUAL_M, (group, mark)


def test_hip_correction_moves_the_pelvis_skin_back_and_nothing_else() -> None:
    back = dataclasses.replace(PLAIN, hip_back_m=0.02)
    base = fit.fit_body_skin(SKIN, _subject(), PLAIN)
    moved = fit.fit_body_skin(SKIN, _subject(), back)
    shift = moved.vertices - base.vertices
    assert np.allclose(shift[:, 1:], 0.0) and np.all(shift[:, 0] <= 1e-12)
    height = base.vertices[:, 2] - base.joints["hip_l"][2]
    arm = SKIN.weights[:, [g.startswith(fit.ARM_GROUPS) for g in SKIN.groups]].sum(1)
    crest = (height > 0.0) & (height < back.hip_back_plateau_m) & (arm == 0.0)
    assert crest.sum() > 20 and np.allclose(shift[crest, 0], -0.02)
    top = back.hip_back_plateau_m + back.hip_back_rise_m
    below_knee = base.vertices[:, 2] < base.joints["knee_l"][2]
    assert np.allclose(shift[(height > top) | below_knee], 0.0)
    for name, point in base.joints.items():
        assert np.array_equal(point, moved.joints[name]), name


def test_shoulder_anchor_lowers_the_skin_over_the_joint() -> None:
    up = 0.02
    base = fit.fit_body_skin(SKIN, _subject(), PLAIN)
    moved = fit.fit_body_skin(
        SKIN, _subject(), dataclasses.replace(PLAIN, shoulder_anchor_m=(0.0, 0.0, up))
    )
    _assert_subject_lengths(moved.joints)
    near = np.linalg.norm(base.vertices - base.joints["shoulder_l"], axis=1) < 0.08
    groups = [g.startswith("upperarm") and g.endswith("_l") for g in SKIN.groups]
    cap = near & (SKIN.weights[:, groups].sum(axis=1) >= 0.8)
    drop = (base.vertices - moved.vertices)[cap, 2]
    assert cap.sum() > 10 and np.allclose(drop, up, atol=3e-3)
    legs = base.vertices[:, 2] < base.joints["hip_l"][2] - PLAIN.trunk_margin_m
    assert np.allclose(base.vertices[legs], moved.vertices[legs], atol=1e-9)


@pytest.mark.parametrize(
    "change",
    [
        {"hip_back_m": -0.001},
        {"hip_back_m": 0.2},
        {"hip_back_rise_m": 0.0},
        {"shoulder_anchor_m": (0.0, 0.0, 0.5)},
        {"shoulder_anchor_m": (0.0, float("nan"), 0.0)},
        {"trunk_margin_m": 0.0},
        {"warp_extent_m": 0.1},
    ],
)
def test_invalid_settings_raise(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(fit.load_fit_settings(), **change)  # type: ignore[arg-type]


def test_invalid_subject_raises() -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(_subject(), thigh=0.0)
    with pytest.raises(ValueError):
        dataclasses.replace(_subject(), neck_height=0.4)  # below the shoulder line


def test_default_settings_ignore_the_profiles() -> None:
    assert fit.load_fit_settings().shoulder_anchor_m == (0.0, 0.0, 0.0)


def test_showpiece_profile_only_raises_the_shoulder_anchor() -> None:
    plain, tuned = fit.load_fit_settings(), fit.load_fit_settings(profile="showpiece")
    assert tuned.shoulder_anchor_m == (0.0, 0.0, 0.025)
    assert dataclasses.replace(tuned, shoulder_anchor_m=(0.0, 0.0, 0.0)) == plain
    _assert_subject_lengths(fit.fit_body_skin(SKIN, _subject(), tuned).joints)


def test_unknown_profile_or_profile_field_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown fit profile"):
        fit.load_fit_settings(profile="missing")
    raw = json.loads((fit.ASSET_DIR / "fit_settings.json").read_text())
    raw["profiles"] = {"bad": {"not_a_setting": 1.0}}
    path = tmp_path / "fit_settings.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="not a fit setting"):
        fit.load_fit_settings(path, profile="bad")
