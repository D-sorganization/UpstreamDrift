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
        {"thorax_front_scale": 1.1},
        {"thorax_front_scale": 0.5},
        {"thorax_front_rise_m": 0.0},
        {"thigh_back_m": 0.2},
        {"thigh_back_rise_fraction": 1.5},
        {"thigh_back_fade_m": 0.0},
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


def test_showpiece_profile_changes_only_its_named_settings() -> None:
    plain, tuned = fit.load_fit_settings(), fit.load_fit_settings(profile="showpiece")
    changes = {"shoulder_anchor_m": (0.0, 0.0, 0.025)}
    changes |= {"thorax_front_scale": 0.92, "thigh_back_m": 0.02, "slim_share": 1.0}
    assert dataclasses.replace(plain, **changes) == tuned  # type: ignore[arg-type]
    assert plain.thorax_front_scale == 1.0 and plain.thigh_back_m == 0.0
    _assert_subject_lengths(fit.fit_body_skin(SKIN, _subject(), tuned).joints)


def test_slim_share_mixes_the_slim_build_in_before_the_fit() -> None:
    assert PLAIN.slim_share == 0.0
    assert SKIN.rest(0.0) is SKIN.vertices
    assert np.allclose(SKIN.rest(0.5), SKIN.vertices + 0.5 * SKIN.slim_offset)
    base = fit.fit_body_skin(SKIN, _subject(), PLAIN)
    lean = fit.fit_body_skin(
        SKIN, _subject(), dataclasses.replace(PLAIN, slim_share=1.0)
    )
    _assert_subject_lengths(lean.joints)
    for name, joint in lean.joints.items():  # the same subject, the same joints
        assert np.array_equal(joint, base.joints[name]), name

    def girth(fitted: fit.FittedSkin, low: float, high: float) -> tuple[float, float]:
        band = (fitted.vertices[:, 2] > low) & (fitted.vertices[:, 2] < high)
        band &= _limb_weight(("upperarm", "forearm")) < 0.5
        box = np.ptp(fitted.vertices[band], axis=0)
        return float(box[0]), float(box[1])

    chest_low = base.joints["shoulder_l"][2] - 0.2
    for low, high in ((chest_low, chest_low + 0.1), (0.05, 0.15)):  # chest, waist
        depth, breadth = girth(base, low, high)
        lean_depth, lean_breadth = girth(lean, low, high)
        assert lean_depth < depth - 0.004 and lean_breadth < breadth - 0.004
    with pytest.raises(ValueError, match="slim_share"):
        dataclasses.replace(PLAIN, slim_share=1.5)
    with pytest.raises(ValueError, match="slim_share"):
        SKIN.rest(-0.1)


def _limb_weight(prefixes: tuple[str, ...]) -> np.ndarray:
    return SKIN.weights[:, [g.startswith(prefixes) for g in SKIN.groups]].sum(axis=1)


def test_thorax_front_scale_flattens_the_chest_and_nothing_else() -> None:
    flat = dataclasses.replace(PLAIN, thorax_front_scale=0.9)
    base = fit.fit_body_skin(SKIN, _subject(), PLAIN)
    moved = fit.fit_body_skin(SKIN, _subject(), flat)
    shift = moved.vertices - base.vertices
    assert np.allclose(shift[:, 1:], 0.0) and np.all(shift[:, 0] <= 1e-12)
    hip_mid = 0.5 * (base.joints["hip_l"] + base.joints["hip_r"])
    axis = (base.joints["neck"] - hip_mid) / np.linalg.norm(
        base.joints["neck"] - hip_mid
    )
    along = (base.vertices - hip_mid) @ axis
    ahead = base.vertices[:, 0] - (hip_mid[0] + along * axis[0])
    trunk = _limb_weight(fit.THORAX_EXEMPT_GROUPS) == 0.0
    full = trunk & (along > flat.thorax_front_start_m + flat.thorax_front_rise_m)
    chest = full & (ahead > 0.05)
    assert chest.sum() > 50 and np.allclose(shift[chest, 0], -0.1 * ahead[chest])
    assert np.allclose(shift[(ahead <= 0.0) | (along < flat.thorax_front_start_m)], 0.0)
    for name, point in base.joints.items():
        assert np.array_equal(point, moved.joints[name]), name


def test_thigh_back_moves_the_thigh_front_back_and_nothing_else() -> None:
    back = dataclasses.replace(PLAIN, thigh_back_m=0.02)
    base = fit.fit_body_skin(SKIN, _subject(), PLAIN)
    moved = fit.fit_body_skin(SKIN, _subject(), back)
    shift = moved.vertices - base.vertices
    assert np.allclose(shift[:, 1:], 0.0) and np.all(shift[:, 0] <= 1e-12)
    hip, knee = base.joints["hip_l"], base.joints["knee_l"]
    height, ahead = base.vertices[:, 2], base.vertices[:, 0] - hip[0]
    middle = knee[2] + back.thigh_back_rise_fraction * (hip[2] - knee[2])
    thigh = _limb_weight(("thigh",)) >= 1.0 - 1e-9
    band = thigh & (height > middle) & (height < hip[2] - back.thigh_back_below_hip_m)
    front = band & (ahead > back.thigh_back_front_m)
    assert front.sum() > 20 and np.allclose(shift[front, 0], -0.02)
    top = hip[2] - back.thigh_back_below_hip_m + back.thigh_back_fade_m
    still = (ahead <= 0.0) | (height < knee[2]) | (height > top)
    assert np.allclose(shift[still | (_limb_weight(("thigh",)) == 0.0)], 0.0)
    for name, point in base.joints.items():
        assert np.array_equal(point, moved.joints[name]), name


def test_unknown_profile_or_profile_field_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown fit profile"):
        fit.load_fit_settings(profile="missing")
    raw = json.loads((fit.ASSET_DIR / "fit_settings.json").read_text())
    raw["profiles"] = {"bad": {"not_a_setting": 1.0}}
    path = tmp_path / "fit_settings.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="not a fit setting"):
        fit.load_fit_settings(path, profile="bad")


def test_the_neck_above_the_joint_rides_the_trunk_and_keeps_its_height() -> None:
    cut = json.loads((fit.ASSET_DIR / "skin_settings.json").read_text())["neck_cut_m"]
    fitted = fit.fit_body_skin(SKIN, _subject(), fit.load_fit_settings())
    for name in ("neck_lower", "neck_middle", "neck_upper", "head"):
        assert fitted.maps[name] is fitted.maps["thorax"]
    stored_top = SKIN.vertices[:, 2].max() - SKIN.landmarks["neck"][2]
    fitted_top = fitted.vertices[:, 2].max() - fitted.joints["neck"][2]
    assert stored_top > cut / 2
    assert (
        abs(fitted_top - stored_top) < 0.005
    )  # a shift above the joint, not a stretch


def test_a_limb_can_be_fitted_to_each_side_on_its_own() -> None:
    sides = {"thigh": (0.431, 0.447), "forearm": (0.280, 0.287)}
    joints = fit.fit_body_skin(
        SKIN, dataclasses.replace(_subject(), sides=sides), PLAIN
    ).joints
    for name, near, far in (("thigh", "hip", "knee"), ("forearm", "elbow", "wrist")):
        for side, want in zip("lr", sides[name], strict=True):
            got = np.linalg.norm(joints[f"{far}_{side}"] - joints[f"{near}_{side}"])
            assert abs(got - want) <= RESIDUAL_M, (name, side)
    with pytest.raises(ValueError, match="tail"):
        dataclasses.replace(_subject(), sides={"tail": (0.1, 0.1)})
    with pytest.raises(ValueError, match="thigh"):
        dataclasses.replace(_subject(), sides={"thigh": (0.4, -0.4)})


def _ring(side: str, fitted: fit.FittedSkin) -> fit.WristRing:
    """A wrist 52 mm wide and 36 mm deep, 5 mm off the joint, askew to the bone."""
    sign = dict(fit.SIDES)[side]
    return fit.WristRing(0.026, 0.018, np.array([0.9, 0.3 * sign, 0.3]), OFF_M)


OFF_M = np.array([0.004, 0.0, 0.003])


def _wrist_radii(
    vertices: np.ndarray, fitted: fit.FittedSkin, side: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per forearm vertex: reach before the wrist, radius and the ring's radius."""
    ring = _ring(side, fitted)
    elbow, wrist = fitted.joints[f"elbow_{side}"], fitted.joints[f"wrist_{side}"]
    axis = (wrist - elbow) / np.linalg.norm(wrist - elbow)
    width = ring.width_axis - (ring.width_axis @ axis) * axis
    width /= np.linalg.norm(width)
    off = ring.centre_m - (ring.centre_m @ axis) * axis
    columns = [g.startswith("forearm") and g.endswith(side) for g in SKIN.groups]
    mine = SKIN.weights[:, columns].sum(axis=1) > 0.5
    local = vertices[mine] - wrist
    radial = local - np.outer(local @ axis, axis) - off
    radius = np.linalg.norm(radial, axis=1)
    cosine = radial @ width / radius
    sine = np.sqrt(np.clip(1.0 - cosine**2, 0.0, None))
    target = 1.0 / np.hypot(cosine / ring.half_width_m, sine / ring.half_depth_m)
    return -(local @ axis), radius, target


def test_the_forearm_ends_just_under_the_hand_wrist_section() -> None:
    settings = fit.load_fit_settings()
    fitted = fit.fit_body_skin(SKIN, _subject(), settings)
    rings = {side: _ring(side, fitted) for side, _ in fit.SIDES}
    matched = fit.match_wrists(SKIN, fitted, rings, settings)
    assert matched.joints is fitted.joints
    for side, _ in fit.SIDES:
        before, plain, target = _wrist_radii(fitted.vertices, fitted, side)
        _, radius, _ = _wrist_radii(matched.vertices, fitted, side)
        end = before < settings.wrist_match_full_m
        assert end.sum() >= 20
        # Radius step from the hand's section down to the forearm skin.
        step = (target - radius)[end]
        assert np.abs(step - settings.wrist_under_m).max() < 2.0e-4, side
        assert np.abs((target - plain)[end]).max() > 2.0e-3  # unmatched: a step
        far = before > settings.wrist_match_from_m
        assert np.abs(radius - plain)[far].max() < 1.0e-12
    moved = np.linalg.norm(matched.vertices - fitted.vertices, axis=1) > 0.0
    forearm = [g.startswith("forearm") for g in SKIN.groups]
    assert (SKIN.weights[moved][:, forearm].sum(axis=1) > 0.5).all()
    assert fit.match_wrists(SKIN, fitted, {}, settings) is fitted


def test_an_invalid_wrist_ring_or_side_raises() -> None:
    settings = fit.load_fit_settings()
    fitted = fit.fit_body_skin(SKIN, _subject(), settings)
    with pytest.raises(ValueError, match="half_depth_m"):
        fit.WristRing(0.02, 0.0, np.array([1.0, 0, 0]), np.zeros(3))
    with pytest.raises(ValueError, match="width_axis"):
        fit.WristRing(0.02, 0.02, np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="side"):
        fit.match_wrists(SKIN, fitted, {"x": _ring("l", fitted)}, settings)
    with pytest.raises(ValueError, match="wrist_match_full_m"):
        dataclasses.replace(settings, wrist_match_full_m=0.2)
