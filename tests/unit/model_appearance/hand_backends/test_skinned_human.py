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
    finger_side_gaps_m,
    forearm_spread_deg,
    grip_contact_area_m2,
    hand_pair_penetration_m,
    hand_to_finger_gaps_m,
    load_hand_visual_quality,
    max_grip_cylinder_penetration_m,
    median_facing_grip_gap_m,
    min_bone_grip_clearance_m,
    pinched_edge_count,
    skinning_defects,
    skinning_residual_m,
)
from src.shared.python.model_appearance.hand_backends.skinned_human import (
    BACKEND_NAME,
    build_hand_visual,
    default_settings,
    load_settings_mapping,
    settings_from_mapping,
    solve_grip_joint_angles,
)

from src.shared.python.model_appearance.hand_backends.skinned_pair import (
    build_grip_hands,
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
def test_the_thumb_lies_on_the_grip_not_on_its_tip(side: str) -> None:
    """Both thumb phalanges seat: the pad is on the grip, not a ramp to it."""
    gaps = _hand(side).contact_gaps_m
    assert gaps["thumb_dist"] <= float(_Q["contact_tolerance_m"])
    limit = float(_Q["thumb_pad_tolerance_m"])
    assert gaps["thumb_prox"] <= limit, f"{side} {gaps['thumb_prox'] * 1e3:.1f} mm"


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
    # Fold relaxation leaves no triangle inside out (it is capped at zero).
    assert defects["flipped"] <= int(_Q["max_crease_fold_faces"]), defects
    assert defects["flipped_area_fraction"] <= float(
        _Q["max_crease_fold_area_fraction"]
    ), defects
    assert result.mesh.volume() > 0.0
    np.testing.assert_allclose(
        np.linalg.norm(result.vertex_normals, axis=1), 1.0, atol=1e-9
    )


def _without_correctives(side: str):
    key = (side, "no correctives")
    if key not in _CACHE:
        raw = load_settings_mapping()
        raw["correctives"] = {
            "fold_relax": {**raw["correctives"]["fold_relax"], "iterations": 0},
            "palm_pad": {**raw["correctives"]["palm_pad"], "reach_m": 0.0},
            "lift_out": {**raw["correctives"]["lift_out"], "max_depth_m": 0.0},
            "crease_relax": {**raw["correctives"]["crease_relax"], "iterations": 0},
        }
        _CACHE[key] = build_hand_visual(
            _spec(), side, grip_radius_m=_RADIUS, settings=settings_from_mapping(raw)
        )
    return _CACHE[key]


def _unsubdivided(side: str):
    key = (side, "no subdivision")
    if key not in _CACHE:
        raw = load_settings_mapping()
        raw["mesh"] = {**raw["mesh"], "subdivision_levels": 0}
        _CACHE[key] = build_hand_visual(
            _spec(), side, grip_radius_m=_RADIUS, settings=settings_from_mapping(raw)
        )
    return _CACHE[key]


def _mean_edge_m(result) -> float:
    v, f = result.mesh.vertices, result.mesh.faces
    sides = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    return float(np.linalg.norm(v[sides[:, 0]] - v[sides[:, 1]], axis=1).mean())


@pytest.mark.parametrize("side", _SIDES)
def test_the_skin_is_subdivided_and_the_setting_switches_it_off(side: str) -> None:
    """One Loop level by default (#15150): finer skin, same hand, same grip."""
    fine, coarse = _hand(side), _unsubdivided(side)
    source = load_hand_cc0_asset().asset
    assert len(coarse.mesh.faces) == len(source.faces)
    assert len(fine.mesh.faces) == 4 * len(source.faces)
    assert len(fine.vertex_normals) == len(fine.mesh.vertices)
    assert len(fine.frame_focus) == len(fine.mesh.vertices)
    limit = float(_Q["max_mean_edge_length_m"])
    assert _mean_edge_m(fine) <= limit < _mean_edge_m(coarse)
    # The solved pose barely moves: subdivision is a change of skin, not grip.
    for joint, angle in coarse.joint_angles.as_mapping().items():
        other = fine.joint_angles.as_mapping()[joint]
        assert abs(other - angle) <= math.radians(
            float(_Q["max_subdivision_joint_change_deg"])
        ), joint
    tolerance = float(_Q["contact_tolerance_m"])
    for result in (fine, coarse):
        assert result.contact_gaps_m["index_dist"] <= tolerance


def _built(key: str, change):
    if key not in _CACHE:
        raw = load_settings_mapping()
        change(raw)
        _CACHE[key] = build_hand_visual(
            _spec(), "lead", grip_radius_m=_RADIUS, settings=settings_from_mapping(raw)
        )
    return _CACHE[key]


def test_nail_edges_stay_sharp_through_subdivision() -> None:
    """Nails are sharp edges in MakeHuman's mesh; smoothing them blurs the nail."""
    from src.shared.python.model_appearance.hand_backends.subdivision import (
        sharp_edges,
    )

    settings = default_settings()

    def nail_edges(result) -> int:
        asset = result.rest_asset
        mask = np.zeros(len(asset.rest_vertices), dtype=bool)
        mask[asset.dominant_vertex_ids(settings.crease_bones)] = True
        limit = settings.crease_min_fold_deg
        return len(sharp_edges(asset.rest_vertices, asset.faces, limit, mask))

    blurred = _built("no creases", lambda raw: raw["mesh"]["creases"].update(bones=[]))
    coarse = nail_edges(_unsubdivided("lead"))
    assert coarse > 0
    # The sharp source edges are split, not smoothed away.
    assert nail_edges(_hand("lead")) > coarse
    assert nail_edges(blurred) < 0.5 * coarse
    assert len(blurred.mesh.faces) == len(_hand("lead").mesh.faces)


@pytest.mark.parametrize("side", _SIDES)
def test_creases_pinched_between_the_knuckles_are_relaxed(side: str) -> None:
    settings = default_settings()
    bones, limit = settings.crease_relax_bones, settings.crease_relax_max_added_fold_deg
    plain, shaped = _without_correctives(side), _hand(side)
    most = int(_Q["max_pinched_knuckle_edges"])
    assert pinched_edge_count(shaped, bones, limit) <= most
    assert pinched_edge_count(plain, bones, limit) > most


def _no_squeeze(side: str):
    key = (side, "no squeeze")
    if key not in _CACHE:
        raw = load_settings_mapping()
        raw["squeeze"] = {"finger_depth_m": 0.0, "palm_depth_m": 0.0}
        _CACHE[key] = build_hand_visual(
            _spec(), side, grip_radius_m=_RADIUS, settings=settings_from_mapping(raw)
        )
    return _CACHE[key]


@pytest.mark.parametrize("side", _SIDES)
def test_the_hand_squeezes_the_grip_instead_of_resting_on_it(side: str) -> None:
    """Joints close a pad-compression depth past first touch (#15179)."""
    loose, firm = _no_squeeze(side), _hand(side)
    band = float(_Q["contact_band_m"])
    area = grip_contact_area_m2(firm, band)
    assert area >= float(_Q["min_grip_contact_area_m2"])
    assert area > 1.5 * grip_contact_area_m2(loose, band)
    curl = {
        which: sum(
            math.degrees(result.joint_angles.as_mapping()[f"{f}_{s}"])
            for f in FINGERS
            for s in ("prox", "mid", "dist")
        )
        for which, result in (("loose", loose), ("firm", firm))
    }
    assert curl["firm"] > curl["loose"]
    # Soft tissue gives way, bone does not: no finger bone enters the grip.
    assert min_bone_grip_clearance_m(firm) >= float(_Q["min_bone_grip_clearance_m"])
    assert max_grip_cylinder_penetration_m(firm.mesh.vertices, _RADIUS) <= float(
        _Q["penetration_tolerance_m"]
    )


def _no_palm_cup(side: str):
    key = (side, "no palm cup")
    if key not in _CACHE:
        raw = load_settings_mapping()
        raw["thumb"] = {**raw["thumb"], "palm_cup_max_deg": 0.0}
        _CACHE[key] = build_hand_visual(
            _spec(), side, grip_radius_m=_RADIUS, settings=settings_from_mapping(raw)
        )
    return _CACHE[key]


@pytest.mark.parametrize("side", _SIDES)
def test_the_palm_cups_so_the_thumb_base_lies_on_the_grip(side: str) -> None:
    """The thumb ray swings toward the grip before the thumb is aimed (#15181)."""
    flat, cupped = _no_palm_cup(side), _hand(side)
    limit = float(_Q["max_thumb_first_segment_median_gap_m"])
    first = ("thumb_prox",)
    assert median_facing_grip_gap_m(cupped, first) <= limit
    assert median_facing_grip_gap_m(flat, first) > limit
    base = ("thumb_meta",)
    closer = float(_Q["min_palm_cup_thenar_approach_m"])
    assert (
        median_facing_grip_gap_m(flat, base) - median_facing_grip_gap_m(cupped, base)
        >= closer
    )
    # Only the thumb ray moves: the fingers hold the grip as before.
    for finger in FINGERS:
        for segment in ("prox", "mid", "dist"):
            name = f"{finger}_{segment}"
            np.testing.assert_allclose(
                cupped.joint_angles.as_mapping()[name],
                flat.joint_angles.as_mapping()[name],
            )
    assert max_grip_cylinder_penetration_m(cupped.mesh.vertices, _RADIUS) <= float(
        _Q["penetration_tolerance_m"]
    )


def _fingers_apart(side: str):
    key = (side, "fingers apart")
    if key not in _CACHE:
        raw = load_settings_mapping()
        raw["side_close"] = {**raw["side_close"], "max_deg": 0.0}
        _CACHE[key] = build_hand_visual(
            _spec(), side, grip_radius_m=_RADIUS, settings=settings_from_mapping(raw)
        )
    return _CACHE[key]


@pytest.mark.parametrize("side", _SIDES)
def test_neighbouring_fingers_touch_along_the_grip(side: str) -> None:
    """Fingers close side to side; the forefinger may keep a slight gap (#15184)."""
    apart, closed = (
        finger_side_gaps_m(_fingers_apart(side)),
        finger_side_gaps_m(_hand(side)),
    )
    limit = float(_Q["max_finger_side_gap_m"])
    index_limit = float(_Q["max_index_side_gap_m"])
    for pair, gap in closed.items():
        most = index_limit if pair.startswith("index") else limit
        assert gap <= most, f"{side} {pair} {gap * 1e3:.1f} mm"
        assert gap > 0.0
    assert apart["middle-ring"] > limit
    assert apart["index-middle"] > index_limit
    # Closing sideways does not lift a finger off the grip.
    tolerance = float(_Q["contact_tolerance_m"])
    for finger in FINGERS:
        for segment in ("mid", "dist"):
            gap = _hand(side).contact_gaps_m[f"{finger}_{segment}"]
            assert gap <= tolerance, f"{side} {finger}_{segment}"


def _palm_contact_count(result) -> int:
    palm = result.rest_asset.dominant_vertex_ids(
        ("wrist",) + tuple(f"meta_{f}" for f in FINGERS)
    )
    xy = result.mesh.vertices[palm, :2]
    gap = np.hypot(xy[:, 0], xy[:, 1]) - _RADIUS
    return int((gap <= float(_Q["contact_tolerance_m"])).sum())


@pytest.mark.parametrize("side", _SIDES)
def test_with_correctives_off_the_pose_is_pure_linear_blend_skinning(
    side: str,
) -> None:
    """Switched off, no vertex is pushed: the mesh is exactly LBS of the pose."""
    plain = _without_correctives(side)
    assert skinning_residual_m(plain) < 1e-12
    for t in plain.bone_transforms:
        np.testing.assert_allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-9)
        assert np.linalg.det(t[:3, :3]) == pytest.approx(1.0)
    # The correctives change the skin only: the solved pose is the same.
    assert plain.joint_angles == _hand(side).joint_angles
    np.testing.assert_array_equal(plain.bone_transforms, _hand(side).bone_transforms)


@pytest.mark.parametrize("side", _SIDES)
def test_correctives_unfold_the_webs_and_seat_the_palm_pad(side: str) -> None:
    """Owner-approved correctives (#15150): small, local, and they earn it."""
    plain, shaped = _without_correctives(side), _hand(side)
    area = float(_Q["min_triangle_area_m2"])
    assert skinning_defects(plain, min_area_m2=area)["flipped"] > 0
    assert skinning_defects(shaped, min_area_m2=area)["flipped"] == 0
    shift = np.linalg.norm(shaped.mesh.vertices - plain.mesh.vertices, axis=1)
    assert 0.0 < shift.max() <= float(_Q["max_corrective_shift_m"])
    assert (shift > 1e-9).mean() <= float(_Q["max_corrective_vertex_fraction"])
    # Finger skin pressed into the grip is laid on its surface; the skin
    # beside it bulges, by less than the press depth.
    digits = shaped.rest_asset.dominant_vertex_ids(
        [f"{f}_{s}" for f in FINGERS for s in ("mid", "dist")] + ["thumb_dist"]
    )
    assert shift[digits].max() <= default_settings().press.max_depth_m
    radial = np.linalg.norm(shaped.mesh.vertices[digits, :2], axis=1)
    inside = np.linalg.norm(plain.mesh.vertices[digits, :2], axis=1) < _RADIUS
    assert inside.any()
    np.testing.assert_allclose(radial[inside], _RADIUS, atol=1e-6)
    least = int(_Q["min_palm_contact_vertices"])
    # Without the correctives the squeezed palm sits inside the grip, where
    # more of it is near the surface; shaped, it must lie on the surface.
    assert _palm_contact_count(shaped) >= least
    palm = shaped.rest_asset.dominant_vertex_ids(
        ("wrist",) + tuple(f"meta_{f}" for f in FINGERS)
    )
    depth = _RADIUS - np.linalg.norm(plain.mesh.vertices[palm, :2], axis=1)
    assert depth.max() > float(_Q["penetration_tolerance_m"])
    assert shaped.contact_gaps_m["palm"] <= plain.contact_gaps_m["palm"]


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


@pytest.mark.parametrize("side", _SIDES)
def test_the_hand_declares_a_frame_focus_that_leaves_out_the_forearm(
    side: str,
) -> None:
    """A renderer frames on the hand; the forearm stub is context (#15150)."""
    result = _hand(side)
    focus = result.frame_focus
    assert focus.dtype == np.bool_ and focus.shape == (len(result.mesh.vertices),)
    forearm = result.rest_asset.dominant_vertex_ids(("forearm",))
    assert len(forearm) > 0 and not focus[forearm].any()
    assert focus.sum() == len(focus) - len(forearm)
    raw = load_settings_mapping()
    raw["framing"] = {"excluded_bones": []}
    everything = build_hand_visual(
        _spec(), side, grip_radius_m=_RADIUS, settings=settings_from_mapping(raw)
    )
    assert everything.frame_focus.all()
    with pytest.raises(ValueError):
        raw["framing"] = {"excluded_bones": ["no_such_bone"]}
        build_hand_visual(
            _spec(), side, grip_radius_m=_RADIUS, settings=settings_from_mapping(raw)
        )


def test_bad_settings_and_inputs_are_rejected() -> None:
    raw = load_settings_mapping()
    with pytest.raises(ValueError):
        settings_from_mapping({**raw, "skin_rgb": [1.2, 0.5, 0.5]})
    with pytest.raises(ValueError):
        settings_from_mapping({**raw, "curl": {**raw["curl"], "arch_passes": 0}})
    for levels in (-1, 4, 1.5, True):
        with pytest.raises(ValueError):
            mesh = {**raw["mesh"], "subdivision_levels": levels}
            settings_from_mapping({**raw, "mesh": mesh})
    for part, bad in (
        ("lift_out", {"max_depth_m": -0.001}),
        ("lift_out", {"bulge_gain": -0.5}),
        ("crease_relax", {"max_added_fold_deg": 0.0}),
        ("crease_relax", {"bones": ["no_such_bone"]}),
        ("fold_relax", {"step": 0.0}),
        ("fold_relax", {"iterations": -1}),
        ("palm_pad", {"reach_m": -0.01}),
        ("palm_pad", {"bones": ["no_such_bone"]}),
    ):
        changed = {**raw["correctives"]}
        changed[part] = {**changed[part], **bad}
        with pytest.raises(ValueError):
            build_hand_visual(
                _spec(),
                "trail",
                grip_radius_m=_RADIUS,
                settings=settings_from_mapping({**raw, "correctives": changed}),
            )
    for bad in (
        {"palm_cup_max_deg": -1.0},
        {"palm_cup_step_deg": 0.0},
        {"palm_cup_pivot_fraction": -0.1},
    ):
        with pytest.raises(ValueError):
            settings_from_mapping({**raw, "thumb": {**raw["thumb"], **bad}})
    for bad in ({"max_deg": -1.0}, {"gap_m": -0.001}, {"index_gap_m": -0.001}):
        with pytest.raises(ValueError):
            settings_from_mapping({**raw, "side_close": {**raw["side_close"], **bad}})
    for bad in ({"finger_depth_m": -0.001}, {"palm_depth_m": -0.001}):
        with pytest.raises(ValueError):
            settings_from_mapping({**raw, "squeeze": {**raw["squeeze"], **bad}})
    with pytest.raises(ValueError):  # a squeeze as deep as the grip is thick
        deep = {**raw, "squeeze": {**raw["squeeze"], "finger_depth_m": _RADIUS}}
        build_hand_visual(
            _spec(),
            "trail",
            grip_radius_m=_RADIUS,
            settings=settings_from_mapping(deep),
        )
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
        sh._sized_assets(size.length_m, size.breadth_m, settings)[0], settings
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


def test_the_trail_hand_closes_up_against_the_lead_index_finger() -> None:
    """The hands touch: trail little and ring fingers on the lead forefinger (#15183)."""
    pair = _pair()
    gaps = hand_to_finger_gaps_m(pair["trail"], pair["lead"], "index")
    assert gaps["little"] <= float(_Q["max_pair_little_finger_gap_m"]), gaps
    assert gaps["ring"] <= float(_Q["max_pair_ring_finger_gap_m"]), gaps


def test_two_hands_share_the_club_without_passing_through_each_other() -> None:
    pair = _pair()
    depth = hand_pair_penetration_m(pair["lead"], pair["trail"])
    assert depth <= float(_Q["pair_penetration_tolerance_m"]), f"{depth * 1e3:.2f} mm"
    for side in _SIDES:
        inside = max_grip_cylinder_penetration_m(pair[side].mesh.vertices, _RADIUS)
        assert inside <= float(_Q["penetration_tolerance_m"])
        defects = skinning_defects(
            pair[side], min_area_m2=float(_Q["min_triangle_area_m2"])
        )
        assert defects["flipped"] == 0 and defects["degenerate"] == 0, defects
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
    # Only the corrective shapes move skin off the skinned pose, and not far.
    for side in _SIDES:
        assert skinning_residual_m(pair[side]) <= float(_Q["max_corrective_shift_m"])


def test_trail_fingers_seat_on_the_grip_and_only_the_little_overlaps() -> None:
    """Overlap grip: the trail little finger rides on the lead hand; the other
    trail fingers reach the grip itself, not the lead index finger."""
    trail = _pair()["trail"]
    grip = np.hypot(*trail.mesh.vertices[:, :2].T) - _RADIUS
    tolerance = float(_Q["contact_tolerance_m"])
    for finger in ("index", "middle", "ring"):
        for segment in ("mid", "dist"):
            ids = trail.rest_asset.dominant_vertex_ids((f"{finger}_{segment}",))
            gap = float(grip[ids].min())
            assert gap <= tolerance, f"trail {finger}_{segment} {gap * 1e3:.1f} mm"
    # The slide toward the head is bounded (the solo trail hand sits at z = 0).
    moved = trail.frame_from_rest[2, 3] - _hand("trail").frame_from_rest[2, 3]
    moved -= _spec().hand_spacing_m
    assert -1e-9 <= moved <= default_settings().pair_max_spacing_adjust_m + 1e-9


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
