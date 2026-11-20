"""The golfer's trunk has adult proportions and no hump under the neck (#15299)."""

from __future__ import annotations

import copy
import dataclasses
import functools
import math

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.body_meshes import body_meshes
from src.shared.python.model_appearance.defaults import default_document_dict
from src.shared.python.model_appearance.torso_profile import (
    TORSO_ACCEPTANCE as ACCEPT,
)
from src.shared.python.model_appearance.torso_profile import (
    TorsoAcceptance,
    TorsoProfile,
    TorsoStation,
)
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton
from tests.unit.model_appearance.test_neck import ADDRESS, DOC, HEAD_M, SPEC, _scene

pytestmark = pytest.mark.unit

STATURE_M = float(SPEC["subject"]["stature_m"])
POSES = {"hang": {"LSInputY": math.pi / 2, "RSInputY": math.pi / 2}, "address": ADDRESS}
UP = np.array([0.0, 0.0, 1.0])
# Shoulder joints as the spec has them, and as the subject preset moves them
# (#15296): (metres lower below the neck joint, half-spacing scale).
LOWERED_M, SPACING = 0.055, 180.0 / 175.0
JOINTS = {"current": (0.0, 1.0), "lowered": (LOWERED_M, SPACING)}
Solid = tuple[str, str, np.ndarray, np.ndarray]


@functools.cache
def _solids(pose: str) -> tuple[tuple[Solid, ...], np.ndarray, np.ndarray]:
    """(spec body, geom name, vertices, faces) of every visual mesh, in the
    thorax frame (x front, y left, z up the trunk); the neck point; world up."""
    mujoco, model, data = _scene(default_document_dict(), POSES[pose])
    thorax = next(i for i in range(model.nbody) if model.body(i).name[-6:] == "COMRod")
    head = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "GolfSwing3D_Kinetic/Head"
    )
    frame, origin = data.xmat[thorax].reshape(3, 3), data.xpos[thorax]
    out = []
    for geom in range(model.ngeom):
        name, mesh = model.geom(geom).name, model.geom_dataid[geom]
        if mesh < 0 or not name.startswith("visual") or "hair" in name:
            continue
        first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        world = model.mesh_vert[first : first + count].astype(float)
        world = world @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        faces = model.mesh_face[model.mesh_faceadr[mesh] :][: model.mesh_facenum[mesh]]
        body = model.body(model.geom_bodyid[geom]).name.split(":")[-1]
        out.append((body, name, (world - origin) @ frame, faces))
    return tuple(out), (data.xpos[head] - origin) @ frame, UP @ frame


def _verts(pose: str, part: str, label: str = "") -> np.ndarray:
    """Vertices of every ``label`` mesh of the bodies classified as ``part``."""
    return np.vstack(
        [
            verts
            for body, name, verts, _ in _solids(pose)[0]
            if library.classify_body(body) == part and label in name
        ]
    )


def _thorax_rings(pose: str, label: str) -> np.ndarray:
    """The widest torso solid's rings, lowest first (the two poles are last)."""
    return _verts(pose, "torso", label)[-2 - 28 * 31 : -2].reshape(31, 28, 3)


def _cut(
    height_m: float, parts: tuple[str, ...], label: str = "", axis: int = 2
) -> np.ndarray:
    """Where the standing ``label`` solids of ``parts`` cross a height (or x)."""
    points = [np.zeros((0, 3))]
    for body, name, verts, faces in _solids("hang")[0]:
        if library.classify_body(body) not in parts or label not in name:
            continue
        side = verts[:, axis] - height_m
        for a, b in ((0, 1), (1, 2), (2, 0)):
            sa, sb = side[faces[:, a]], side[faces[:, b]]
            hit = sa * sb < 0.0
            t = (sa[hit] / (sa[hit] - sb[hit]))[:, None]
            points.append(verts[faces[hit, a]] * (1 - t) + verts[faces[hit, b]] * t)
    return np.vstack(points)


def _back(height_m: float) -> float:
    """Rearmost point of trunk, yokes, neck and head at a height (standing)."""
    return float(_cut(height_m, ("torso", "shoulder", "head"))[:, 0].min())


@functools.cache
def _back_profile() -> tuple[np.ndarray, np.ndarray]:
    """Heights (head lengths about the neck point) and the back there (m)."""
    neck_z = float(_solids("hang")[1][2])
    low, high = ACCEPT.back_span
    heights = np.arange(low, high + 1e-9, ACCEPT.back_step)
    return heights, np.array([_back(neck_z + h * HEAD_M) for h in heights])


def _band(band: tuple[float, float]) -> tuple[float, float]:
    low, high = band
    return (
        low * (1.0 - ACCEPT.norm_tolerance) * STATURE_M,
        high * (1.0 + ACCEPT.norm_tolerance) * STATURE_M,
    )


def _breadths() -> dict[str, float]:
    ring_width = np.ptp(_thorax_rings("hang", "skin")[:, :, 1], axis=1)
    arms = np.vstack([_verts("hang", "upper_arm"), _verts("hang", "shoulder")])
    return {
        "chest": float(ring_width.max()),
        "waist": float(ring_width[: int(ring_width.argmax())].min()),
        "hip": float(np.ptp(_verts("hang", "pelvis", "skin")[:, 1])),
        "bideltoid": float(np.ptp(arms[:, 1])),
    }


def test_the_trunk_breadths_are_inside_the_anthropometric_bands() -> None:
    got = _breadths()
    for name, value in got.items():
        low, high = _band(getattr(ACCEPT, f"{name}_breadth"))
        assert low <= value <= high, (name, value, low, high)
    depth = float(np.ptp(_verts("hang", "torso", "skin")[:, 0]))
    assert depth >= ACCEPT.min_chest_depth * STATURE_M, depth


def test_the_torso_tapers_from_the_shoulders_to_the_waist() -> None:
    got = _breadths()
    assert got["bideltoid"] > got["chest"] > got["waist"], got
    assert got["waist"] >= ACCEPT.min_waist_over_chest * got["chest"], got
    assert got["chest"] >= ACCEPT.min_chest_over_hip * got["hip"], got


def test_the_waist_is_drawn_in_under_the_chest() -> None:
    got = _breadths()
    assert got["waist"] <= ACCEPT.max_waist_over_chest * got["chest"], got


def _trunk_backs() -> tuple[float, float, float, float]:
    """Skin behind the trunk axis: buttock, waist, chest; and the waist front."""
    pelvis, rings = _verts("hang", "pelvis", "skin"), _thorax_rings("hang", "skin")
    rear = -rings[:, :, 0].min(axis=1)
    waist = int(rear[: int(np.ptp(rings[:, :, 1], axis=1).argmax())].argmin())
    front = float(rings[waist, :, 0].max())
    return float(-pelvis[:, 0].min()), float(rear[waist]), float(rear.max()), front


def test_the_back_hollows_at_the_waist_and_the_belly_stands_forward() -> None:
    buttock, waist, chest, front = _trunk_backs()
    hollow = min(buttock, chest) - waist
    assert hollow >= ACCEPT.min_lumbar_hollow * STATURE_M, (buttock, waist, chest)
    assert front >= ACCEPT.min_waist_front_over_back * waist, (front, waist)


def test_the_buttock_stays_full_below_the_hip_joints() -> None:
    pelvis = _verts("hang", "pelvis", "skin")
    hips = float(pelvis[np.abs(pelvis[:, 1]).argmax(), 2])  # the widest level
    below = _cut(0.5 * (hips + float(pelvis[:, 2].min())), ("pelvis",), "skin")
    level = _cut(hips - 1e-4, ("pelvis",), "skin")
    assert below[:, 0].min() <= ACCEPT.min_buttock_fullness * level[:, 0].min()


@pytest.mark.parametrize("label", ["skin", "garment"])
def test_the_pelvis_meets_the_chest_without_a_ledge(label: str) -> None:
    """Flank, front, back and the two corners agree across the lumbar plane."""
    pelvis = _verts("hang", "pelvis", label)
    top = pelvis[pelvis[:, 2] > pelvis[:, 2].max() - 1e-6]
    ring = _thorax_rings("hang", label)[0]
    assert abs(float(ring[:, 2].mean() - top[:, 2].mean())) <= ACCEPT.max_join_gap_m
    for angle in np.radians(ACCEPT.join_directions_deg):
        way = np.array([math.cos(angle), math.sin(angle), 0.0])
        step = abs(float((ring @ way).max() - (top @ way).max()))
        assert step <= ACCEPT.max_join_step_m, (label, angle, step)


def test_the_chest_stands_forward_of_the_trunk_axis() -> None:
    """The neck leans forward of the axis: a trunk centred on it reads hunched."""
    front_back = _thorax_rings("hang", "skin")[:, :, 0]
    assert front_back.max() >= -ACCEPT.min_chest_ahead * front_back.min()


def _yokes(joints: str) -> tuple[np.ndarray, float, float]:
    """Left yoke vertices in its own frame; neck height there; joint offset."""
    lower, spacing = JOINTS[joints]
    spec, reach = copy.deepcopy(SPEC), 0.0
    for joint in spec["joints"]:
        part = library.classify_body(joint["child"])
        if part == "shoulder":
            joint["parent_to_base"][2][3] -= lower
        elif part == "upper_arm":
            joint["parent_to_base"][1][3] *= spacing
            reach = abs(joint["parent_to_base"][1][3])
    thorax = [
        c.length_m()
        for c in derive_visual_skeleton(spec).capsules
        if c.body.endswith("COMRod")
    ]
    skins = [
        m.mesh_in_body_frame.vertices
        for m in body_meshes(spec, DOC)
        if library.classify_body(m.body) == "shoulder" and m.label.startswith("skin")
    ]
    left = max(skins, key=lambda v: float(v[:, 1].max()))
    return left, max(thorax) - min(thorax), reach


def _yoke_tops(joints: str) -> tuple[float, float]:
    """Top of the yoke beside the neck and at the joint, below the neck point."""
    verts, rise, reach = _yokes(joints)
    inner = verts[verts[:, 1] < ACCEPT.yoke_ends[0] * reach][:, 2].max()
    outer = verts[verts[:, 1] > ACCEPT.yoke_ends[1] * reach][:, 2].max()
    return float(rise - inner), float(rise - outer)


def test_the_yoke_hangs_from_the_neck_wherever_the_shoulder_joints_sit() -> None:
    (inner, outer), (low_inner, low_outer) = (_yoke_tops(j) for j in JOINTS)
    assert abs(low_inner - inner) <= ACCEPT.max_yoke_neck_shift * LOWERED_M
    gain = (low_outer - low_inner) - (outer - inner)
    assert gain >= ACCEPT.min_trapezius_gain * LOWERED_M, gain


@pytest.mark.parametrize("joints", list(JOINTS))
def test_the_yoke_reaches_the_shoulder_joint_and_never_above_the_neck(
    joints: str,
) -> None:
    verts, rise, reach = _yokes(joints)
    assert verts[:, 1].max() == pytest.approx(reach, rel=ACCEPT.norm_tolerance)
    assert verts[:, 2].max() < rise


def test_the_deltoid_climbs_as_the_neck_clears_it() -> None:
    radius = library.PART_RADIUS_M["upper_arm"]
    default = library.deltoid_drop_m(radius, None)
    assert default == pytest.approx(library.DELTOID_DROP * radius)
    drops = [library.deltoid_drop_m(radius, r) for r in np.linspace(0.0, 0.3, 13)]
    assert all(a >= b for a, b in zip(drops, drops[1:], strict=False))
    assert drops[-1] == pytest.approx(library.DELTOID_MIN_DROP * radius)


def test_the_thorax_rounds_off_above_the_chest() -> None:
    rows = library.TORSO_PROFILES["thorax"]
    deepest = max(range(len(rows)), key=lambda i: rows[i][2])
    square = [row[4] or library.TORSO_SQUARENESS for row in rows[deepest + 1 :]]
    assert max(square) <= ACCEPT.max_shoulder_squareness
    assert square == sorted(square, reverse=True)


@pytest.mark.parametrize("axis", [0, 2])
def test_the_yoke_does_not_stand_proud_of_the_shirt(axis: int) -> None:
    """No bar, ridge or bump: cut by depth (0) and by height (2), it stays in."""
    yoke = _verts("hang", "shoulder", "garment")[:, axis]
    high = min(yoke.max(), _verts("hang", "upper_arm")[:, 2].max() if axis else 1.0)
    keep = [i for i in range(3) if i != axis]
    for at in np.linspace(yoke.min(), high, ACCEPT.yoke_cuts + 2)[1:-1]:
        arms = _cut(at, ("upper_arm",), "", axis)
        shirt = np.vstack([_cut(at, ("torso",), "garment", axis), arms])[:, keep]
        hull = ConvexHull(shirt).equations
        cut = _cut(at, ("shoulder",), "garment", axis)[:, keep]
        assert (cut @ hull[:, :2].T + hull[:, 2]).max() <= ACCEPT.max_yoke_proud_m


def test_moving_the_shoulder_line_keeps_the_trunk_ends_and_its_width() -> None:
    rows = library.TORSO_PROFILES["thorax"]
    profile = TorsoProfile(tuple(TorsoStation(*row) for row in rows))
    line = library.THORAX_SHOULDER_ALONG
    assert profile.with_shoulder(line, line) == profile
    moved = profile.with_shoulder(line, line * (1.0 - ACCEPT.min_trapezius_gain))
    ends = [(p.stations[0].along, p.stations[-1].along) for p in (profile, moved)]
    assert ends[0] == pytest.approx(ends[1])
    alongs = [s.along for s in moved.stations]
    assert alongs == sorted(alongs)
    assert max(s.width for s in moved.stations) == max(
        s.width for s in profile.stations
    )
    for bad in (0.0, profile.stations[-1].along):
        with pytest.raises(ValueError, match="shoulder line"):
            profile.with_shoulder(line, bad)


def test_the_back_has_no_hump_between_the_thorax_and_the_neck() -> None:
    heights, back = _back_profile()
    points = np.column_stack([back, heights * HEAD_M])
    chord = points[-1] - points[0]
    length = float(np.linalg.norm(chord))
    behind = (points - points[0]) @ (np.array([-chord[1], chord[0]]) / length)
    assert behind.max() <= ACCEPT.max_back_bump * length, (behind.max(), length)


def test_the_back_has_no_shelf_under_the_neck() -> None:
    _, back = _back_profile()
    span = round(ACCEPT.jump_window / ACCEPT.back_step)
    jump = float((back[span:] - back[:-span]).max())
    assert jump <= ACCEPT.max_back_jump * HEAD_M, jump


def test_the_neck_base_is_not_ahead_of_the_line_of_the_upper_back() -> None:
    neck_z = float(_solids("hang")[1][2])
    low, high = ACCEPT.back_tangent
    back_low, back_high = (_back(neck_z + h * HEAD_M) for h in (low, high))
    line = back_high + (back_high - back_low) * (0.0 - high) / (high - low)
    ahead = _back(neck_z) - line
    assert ahead <= ACCEPT.max_neck_ahead_of_back * HEAD_M, (ahead, line)


@pytest.mark.parametrize("label", ["skin", "garment"])
def test_the_thorax_is_mirror_symmetric(label: str) -> None:
    rings = _thorax_rings("hang", label).reshape(-1, 3)
    for point in (rings * np.array([1.0, -1.0, 1.0]))[::5]:
        assert np.linalg.norm(rings - point, axis=1).min() < 1e-5


def test_the_forearms_are_not_swallowed_by_the_shirt_at_address() -> None:
    """The lead arm lies across the chest: its axis stays outside the shirt."""
    rings = _thorax_rings("address", "garment")
    levels = rings[:, :, 2].mean(axis=1)
    deepest = 0.0
    for point in _verts("address", "forearm", "skin"):
        if not levels[0] < point[2] < levels[-1]:
            continue
        ring = rings[int(np.searchsorted(levels, point[2]))]
        centre = ring[:, :2].mean(axis=0)
        spokes, spoke = ring[:, :2] - centre, point[:2] - centre
        angles = np.arctan2(spokes[:, 1], spokes[:, 0])
        reach = np.interp(
            math.atan2(spoke[1], spoke[0]),
            np.sort(angles),
            np.linalg.norm(spokes, axis=1)[np.argsort(angles)],
            period=2.0 * math.pi,
        )
        deepest = max(deepest, float(reach - np.linalg.norm(spoke)))
    limit = ACCEPT.max_forearm_in_chest * library.PART_RADIUS_M["forearm"]
    assert deepest <= limit, deepest


def test_bad_acceptance_settings_are_refused() -> None:
    with pytest.raises(ValueError, match="bands"):
        dataclasses.replace(ACCEPT, chest_breadth=(0.2, 0.1))
    with pytest.raises(ValueError, match="limits"):
        TorsoAcceptance(max_back_bump=0.0)
