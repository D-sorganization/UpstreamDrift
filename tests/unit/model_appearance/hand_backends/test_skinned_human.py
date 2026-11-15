"""Skinned MakeHuman hand holding the grip (epic #15150, #15148).

Every tolerance comes from ``scripts/config/hand_visual_quality.json``.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends import (
    build_hand_visual as registry_build,
)
from src.shared.python.model_appearance.hand_backends import (
    list_hand_visual_backends,
)
from src.shared.python.model_appearance.hand_backends.hand_asset import (
    load_hand_cc0_asset,
)
from src.shared.python.model_appearance.hand_backends.joint_angles import (
    FINGERS,
    HandJointAngles,
)
from src.shared.python.model_appearance.hand_backends.qa import (
    anthropometric_ratios,
    forearm_spread_deg,
    hand_pair_penetration_m,
    load_hand_visual_quality,
    max_grip_cylinder_penetration_m,
    skinning_defects,
    skinning_residual_m,
)
from src.shared.python.model_appearance.hand_backends.skinned_human import (
    BACKEND_NAME,
    build_grip_hands,
    build_hand_visual,
    default_settings,
    load_settings_mapping,
    settings_from_mapping,
    solve_grip_joint_angles,
)

pytestmark = pytest.mark.unit

_Q = load_hand_visual_quality()
_RADIUS = float(_Q["grip_radius_m"])
_SIDES = ("lead", "trail")
_CACHE: dict[tuple, object] = {}


def _spec(roll_deg: float = 0.0, handedness: str = "right") -> GripSpec:
    base = GripSpec.legacy()
    return dataclasses.replace(
        base,
        handedness=handedness,
        lead=dataclasses.replace(base.lead, shaft_roll_deg=roll_deg),
        trail=dataclasses.replace(base.trail, shaft_roll_deg=roll_deg),
    )


def _hand(side: str, roll_deg: float = 0.0, handedness: str = "right"):
    key = (side, roll_deg, handedness)
    if key not in _CACHE:
        _CACHE[key] = build_hand_visual(
            _spec(roll_deg, handedness), side, grip_radius_m=_RADIUS
        )
    return _CACHE[key]


@pytest.mark.parametrize("side", _SIDES)
def test_every_fingertip_and_middle_phalanx_touches_the_grip(side: str) -> None:
    gaps = _hand(side).contact_gaps_m
    tolerance = float(_Q["contact_tolerance_m"])
    for finger in FINGERS:
        for segment in ("mid", "dist"):
            gap = gaps[f"{finger}_{segment}"]
            assert gap <= tolerance, f"{side} {finger}_{segment} {gap * 1e3:.2f} mm"
    assert gaps["thumb_dist"] <= tolerance, f"{side} thumb {gaps['thumb_dist']}"
    assert gaps["palm"] <= tolerance, f"{side} palm {gaps['palm'] * 1e3:.2f} mm"


@pytest.mark.parametrize("side", _SIDES)
def test_no_skin_penetrates_the_grip_beyond_tolerance(side: str) -> None:
    result = _hand(side)
    depth = max_grip_cylinder_penetration_m(result.mesh.vertices, _RADIUS)
    assert depth <= float(_Q["penetration_tolerance_m"]), f"{depth * 1e3:.2f} mm"
    assert result.contact_gaps_m["hand"] >= -float(_Q["penetration_tolerance_m"])


@pytest.mark.parametrize("side", _SIDES)
def test_no_flipped_or_degenerate_triangles_after_skinning(side: str) -> None:
    result = _hand(side)
    defects = skinning_defects(result, min_area_m2=float(_Q["min_triangle_area_m2"]))
    assert defects["degenerate"] == 0
    # Pure LBS folds a few triangles into the inner crease of a joint bent
    # near a right angle; none may flip anywhere else, and the fold is capped.
    assert defects["flipped"] <= int(_Q["max_crease_fold_faces"]), defects
    assert defects["flipped_area_fraction"] <= float(
        _Q["max_crease_fold_area_fraction"]
    ), defects
    assert result.mesh.volume() > 0.0
    np.testing.assert_allclose(
        np.linalg.norm(result.vertex_normals, axis=1), 1.0, atol=1e-9
    )


@pytest.mark.parametrize("side", _SIDES)
def test_pose_is_joint_rotation_and_linear_blend_skinning_only(side: str) -> None:
    """No vertex is pushed: the mesh is exactly LBS of the reported pose."""
    assert skinning_residual_m(_hand(side)) < 1e-12
    transforms = _hand(side).bone_transforms
    for t in transforms:
        np.testing.assert_allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-9)
        assert np.linalg.det(t[:3, :3]) == pytest.approx(1.0)


def test_anthropometric_ratios_match_the_references() -> None:
    settings = default_settings()
    ratios = anthropometric_ratios(load_hand_cc0_asset().asset, settings.breadth_band_m)
    tolerance = float(_Q["anthropometric_tolerance"])
    for name, reference in _Q["anthropometric_reference"].items():
        error = abs(ratios[name] - reference) / reference
        assert error <= tolerance, f"{name}: {ratios[name]:.3f} vs {reference}"


def test_hand_is_scaled_to_the_spec_hand_size() -> None:
    spec = _spec()
    asset = _hand("trail").rest_asset
    from src.shared.python.model_appearance.hand_backends.skinned_human import (
        asset_hand_breadth_m,
        asset_hand_length_m,
    )

    assert asset_hand_length_m(asset) == pytest.approx(spec.hand_size.length_m)
    assert asset_hand_breadth_m(
        asset, default_settings().breadth_band_m
    ) == pytest.approx(spec.hand_size.breadth_m)


def test_left_handed_golfer_is_the_exact_mirror_image() -> None:
    """A left-hander's hands are the right-hander's reflected in the swing plane."""
    for side in _SIDES:
        right = _hand(side).mesh
        left = _hand(side, handedness="left").mesh
        mirrored = right.vertices * np.array([1.0, -1.0, 1.0])
        np.testing.assert_allclose(
            left.vertices, mirrored, atol=float(_Q["mirror_error_m"])
        )
        np.testing.assert_array_equal(left.faces, right.faces[:, ::-1])
        assert left.volume() > 0.0


def _chirality(result) -> float:
    """Sign of (index→little knuckle) x (wrist→middle knuckle) . palm normal."""
    asset, frame, t = result.rest_asset, result.frame_from_rest, result.bone_transforms

    def head(bone: str) -> np.ndarray:
        i = asset.bone_index(bone)
        local = t[i][:3, :3] @ asset.bone_heads[i] + t[i][:3, 3]
        return frame[:3, :3] @ local + frame[:3, 3]

    across = head("little_prox") - head("index_prox")
    along = head("middle_prox") - head("wrist")
    palm_centre = 0.5 * (head("index_prox") + head("little_prox"))
    toward_grip = -np.array([palm_centre[0], palm_centre[1], 0.0])
    return float(np.sign(np.cross(across, along) @ toward_grip))


def test_lead_and_trail_are_opposite_hands_on_opposite_sides() -> None:
    lead, trail = _hand("lead"), _hand("trail")
    assert _chirality(lead) == -_chirality(trail) != 0.0
    # Right-handed golfer: the trail (right) hand's palm sits away from the
    # target (+y), the lead (left) hand's toward it.
    assert trail.mesh.vertices[:, 1].mean() > 0.0 > lead.mesh.vertices[:, 1].mean()
    assert np.linalg.det(trail.frame_from_rest[:3, :3]) == pytest.approx(1.0)
    assert np.linalg.det(lead.frame_from_rest[:3, :3]) == pytest.approx(-1.0)


@pytest.mark.parametrize("side", _SIDES)
@pytest.mark.parametrize("roll_deg", (-20.0, 20.0))
def test_shaft_roll_turns_the_whole_hand_about_the_shaft(
    side: str, roll_deg: float
) -> None:
    flat = _hand(side, 0.0)
    rolled = _hand(side, roll_deg)
    a = math.radians(roll_deg)
    rot = np.array(
        [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0, 0, 1]]
    )
    np.testing.assert_allclose(
        rolled.mesh.vertices,
        flat.mesh.vertices @ rot.T,
        atol=float(_Q["shaft_roll_error_m"]),
    )
    # Contact is unchanged: the grip is a surface of revolution.
    depth = max_grip_cylinder_penetration_m(rolled.mesh.vertices, _RADIUS)
    assert depth <= float(_Q["penetration_tolerance_m"])
    assert rolled.joint_angles == flat.joint_angles


def test_left_handed_shaft_roll_is_mirrored() -> None:
    right = _hand("lead", 20.0).mesh.vertices
    left = _hand("lead", 20.0, handedness="left").mesh.vertices
    np.testing.assert_allclose(
        left, right * np.array([1.0, -1.0, 1.0]), atol=float(_Q["mirror_error_m"])
    )


@pytest.mark.parametrize("side", _SIDES)
def test_club_lies_on_the_index_first_segment_and_below_the_pinky_base(
    side: str,
) -> None:
    """Dieter's grip spec, measured on the posed hand."""
    result = _hand(side)
    asset, frame, t = result.rest_asset, result.frame_from_rest, result.bone_transforms

    def joint(bone: str, tail: bool = False) -> np.ndarray:
        i = asset.bone_index(bone)
        rest = asset.bone_tails[i] if tail else asset.bone_heads[i]
        return frame[:3, :3] @ (t[i][:3, :3] @ rest + t[i][:3, 3]) + frame[:3, 3]

    # The index finger's first segment spans the grip axis station.
    mcp, pip = joint("index_prox"), joint("index_mid")
    gaps = result.contact_gaps_m
    assert gaps["index_prox"] <= float(_Q["contact_tolerance_m"])
    # The little-finger knuckle is distal of where the club passes it: the
    # club crosses the palm just below the base of the pinky.
    wrist, little = joint("wrist"), joint("little_prox")
    distal = joint("middle_prox") - wrist
    distal /= np.linalg.norm(distal)
    foot = np.array([0.0, 0.0, little[2]])  # nearest grip-axis point
    depth = float((little - foot) @ distal)
    low, high = _Q["pinky_palm_depth_range_m"]
    assert low <= depth <= high, f"{side} pinky depth {depth * 1e3:.1f} mm"
    # Slight angle: index knuckle and little knuckle are at different heights.
    assert abs(mcp[2] - little[2]) > 0.02
    assert np.isfinite(pip).all()


@pytest.mark.parametrize("side", _SIDES)
def test_fingers_wrap_and_the_palm_cups(side: str) -> None:
    angles = _hand(side).joint_angles
    for finger in FINGERS:
        total = sum(
            angles.flexion_deg(f"{finger}_{s}") for s in ("prox", "mid", "dist")
        )
        assert total > float(_Q["min_finger_total_curl_deg"]), (
            f"{side} {finger} only curls {total:.0f} deg"
        )
        for joint, segment in (("mcp", "prox"), ("pip", "mid"), ("dip", "dist")):
            low, high = dict(default_settings().limits_deg)[joint]
            assert (
                low - 1e-6 <= angles.flexion_deg(f"{finger}_{segment}") <= high + 1e-6
            )
    assert angles.flexion_deg("meta_little") > angles.flexion_deg("meta_ring") > 0.0


def test_thumb_lies_along_the_shaft_at_the_spec_azimuth() -> None:
    for side in _SIDES:
        result = _hand(side)
        asset = result.rest_asset
        ids = asset.dominant_vertex_ids(("thumb_dist",))
        pad = result.mesh.vertices[ids].mean(axis=0)
        fingers = result.mesh.vertices[
            asset.dominant_vertex_ids(tuple(f"{f}_dist" for f in FINGERS))
        ].mean(axis=0)
        azimuth = math.degrees(math.atan2(pad[1], pad[0]))
        assert abs(azimuth) < 45.0, f"{side} thumb azimuth {azimuth:.0f}"
        assert pad[2] > fingers[2], "thumb points toward the head"


def test_settings_drive_the_result() -> None:
    raw = load_settings_mapping()
    raw["skin_rgb"] = [0.4, 0.3, 0.2]
    raw["curl"] = {**raw["curl"], "limits_deg": {**raw["curl"]["limits_deg"]}}
    raw["curl"]["limits_deg"]["pip"] = [0.0, 30.0]
    settings = settings_from_mapping(raw)
    result = build_hand_visual(
        _spec(), "trail", grip_radius_m=_RADIUS, settings=settings
    )
    assert result.skin_rgb == (0.4, 0.3, 0.2)
    for finger in FINGERS:
        assert result.joint_angles.flexion_deg(f"{finger}_mid") <= 30.0 + 1e-6
    thicker = build_hand_visual(_spec(), "trail", grip_radius_m=_RADIUS * 1.3)
    assert thicker.joint_angles != _hand("trail").joint_angles
    assert max_grip_cylinder_penetration_m(
        thicker.mesh.vertices, _RADIUS * 1.3
    ) <= float(_Q["penetration_tolerance_m"])


def test_bad_settings_and_inputs_are_rejected() -> None:
    raw = load_settings_mapping()
    with pytest.raises(ValueError):
        settings_from_mapping({**raw, "skin_rgb": [1.2, 0.5, 0.5]})
    with pytest.raises(ValueError):
        settings_from_mapping({**raw, "curl": {**raw["curl"], "arch_passes": 0}})
    with pytest.raises(ValueError):
        build_hand_visual(_spec(), "trail", grip_radius_m=0.0)
    with pytest.raises(ValueError):
        build_hand_visual(_spec(), "middle", grip_radius_m=_RADIUS)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_hand_visual(_spec(), "trail", {"a": 1.0}, grip_radius_m=_RADIUS)  # type: ignore[arg-type]


def test_explicit_joint_angles_are_applied_as_given() -> None:
    solved = solve_grip_joint_angles(_spec(), "trail", grip_radius_m=_RADIUS)
    assert solved == _hand("trail").joint_angles
    again = build_hand_visual(_spec(), "trail", solved, grip_radius_m=_RADIUS)
    np.testing.assert_allclose(
        again.mesh.vertices, _hand("trail").mesh.vertices, atol=1e-9
    )
    flat = build_hand_visual(
        _spec(), "trail", HandJointAngles.from_mapping({}), grip_radius_m=_RADIUS
    )
    assert flat.contact_gaps_m["index_dist"] > 0.01, "flat hand must not wrap"


def test_registry_builds_the_default_backend() -> None:
    assert BACKEND_NAME in list_hand_visual_backends()
    result = registry_build(_spec(), "lead", grip_radius_m=_RADIUS)
    assert result.backend == BACKEND_NAME
    np.testing.assert_allclose(result.mesh.vertices, _hand("lead").mesh.vertices)


def test_fast_posing_kinematics_match_the_reference_skinning() -> None:
    from src.shared.python.model_appearance.hand_backends import skinned_human as sh
    from src.shared.python.model_appearance.hand_backends.skinning import (
        bone_transforms,
        pose_vertices,
    )

    settings = default_settings()
    size = _spec().hand_size
    hand = sh._GripHand(
        sh._sized_asset(size.length_m, size.breadth_m, settings), settings
    )
    angles = _hand("trail").joint_angles.as_mapping()
    reference = bone_transforms(hand.asset, hand.rotations(angles))
    np.testing.assert_allclose(hand.transforms(angles), reference, atol=1e-12)
    np.testing.assert_allclose(
        hand.points(angles), pose_vertices(hand.asset, reference), atol=1e-12
    )


def _pair(roll_deg: float = 0.0):
    key = ("pair", roll_deg)
    if key not in _CACHE:
        _CACHE[key] = build_grip_hands(_spec(roll_deg), grip_radius_m=_RADIUS)
    return _CACHE[key]


def test_two_hands_share_the_club_without_passing_through_each_other() -> None:
    pair = _pair()
    depth = hand_pair_penetration_m(pair["lead"], pair["trail"])
    assert depth <= float(_Q["pair_penetration_tolerance_m"]), f"{depth * 1e3:.2f} mm"
    for side in _SIDES:
        inside = max_grip_cylinder_penetration_m(pair[side].mesh.vertices, _RADIUS)
        assert inside <= float(_Q["penetration_tolerance_m"])
    # On their own the two hands overlap: the pair solve is doing real work.
    alone = build_hand_visual(_spec(), "trail", grip_radius_m=_RADIUS)
    shifted = dataclasses.replace(
        alone,
        mesh=type(alone.mesh)(
            alone.mesh.vertices + [0.0, 0.0, _spec().hand_spacing_m],
            alone.mesh.faces,
        ),
    )
    assert hand_pair_penetration_m(pair["lead"], shifted) > depth + 0.002


def test_trail_hand_of_the_pair_still_wraps_the_grip_or_the_lead_hand() -> None:
    pair = _pair()
    gaps = pair["trail"].contact_gaps_m
    tolerance = float(_Q["contact_tolerance_m"])
    for finger in FINGERS:
        for segment in ("mid", "dist"):
            assert gaps[f"{finger}_{segment}"] <= tolerance, f"{finger}_{segment}"
    assert gaps["hand"] >= -float(_Q["penetration_tolerance_m"])
    np.testing.assert_allclose(
        pair["lead"].mesh.vertices, _hand("lead").mesh.vertices, atol=1e-12
    )
    lead_z = pair["lead"].frame_from_rest[2, 3]
    trail_z = pair["trail"].frame_from_rest[2, 3]
    assert trail_z - lead_z > 0.5 * _spec().hand_spacing_m, "trail is nearer the head"
    for side in _SIDES:
        assert skinning_residual_m(pair[side]) < 1e-9


def test_forearms_leave_the_club_on_the_golfers_side() -> None:
    low, high = (float(v) for v in _Q["forearm_spread_range_deg"])
    pair = _pair()
    spread = forearm_spread_deg(pair["lead"], pair["trail"])
    assert low <= spread <= high, f"forearms {spread:.0f} deg apart"


def test_pair_follows_per_hand_shaft_roll() -> None:
    flat, rolled = _pair(), _pair(20.0)
    turn = math.radians(20.0)
    c, s = math.cos(turn), math.sin(turn)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    for side in _SIDES:
        a, b = flat[side].mesh.vertices, rolled[side].mesh.vertices
        direct = np.abs(a @ rot.T - b).max()
        other = np.abs(a @ rot - b).max()
        assert min(direct, other) < float(_Q["shaft_roll_error_m"]) + 1e-6


def test_wrist_bend_turns_the_forearm_and_leaves_the_hand_alone() -> None:
    raw = load_settings_mapping()
    straight = {"extension_deg": 0.0, "ulnar_deviation_deg": 0.0}
    raw["wrist"] = {"lead": straight, "trail": straight}
    unbent = build_hand_visual(
        _spec(), "trail", grip_radius_m=_RADIUS, settings=settings_from_mapping(raw)
    )
    bent = _hand("trail")
    asset = bent.rest_asset
    digits = asset.dominant_vertex_ids(
        tuple(f"{f}_{s}" for f in FINGERS for s in ("prox", "mid", "dist"))
    )
    np.testing.assert_allclose(
        bent.mesh.vertices[digits], unbent.mesh.vertices[digits], atol=1e-9
    )
    stub = asset.dominant_vertex_ids(("forearm",))
    moved = np.abs(bent.mesh.vertices[stub] - unbent.mesh.vertices[stub]).max()
    assert moved > 0.003, "the forearm did not turn"
    assert bent.joint_angles == unbent.joint_angles
