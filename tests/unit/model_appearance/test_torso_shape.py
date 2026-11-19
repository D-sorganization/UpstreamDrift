"""The golfer's trunk has adult proportions and no hump under the neck (#15299)."""

from __future__ import annotations

import dataclasses
import functools
import math

import numpy as np
import pytest

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.defaults import default_document_dict
from src.shared.python.model_appearance.torso_profile import (
    TORSO_ACCEPTANCE as ACCEPT,
)
from src.shared.python.model_appearance.torso_profile import TorsoAcceptance
from tests.unit.model_appearance.test_neck import ADDRESS, HEAD_M, SPEC, _scene

pytestmark = pytest.mark.unit

STATURE_M = float(SPEC["subject"]["stature_m"])
POSES = {"hang": {"LSInputY": math.pi / 2, "RSInputY": math.pi / 2}, "address": ADDRESS}
UP = np.array([0.0, 0.0, 1.0])
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


def _back(height_m: float) -> float:
    """Rearmost point of trunk, yokes, neck and head at a height (standing)."""
    solids, _, _ = _solids("hang")
    rear = []
    for body, _, verts, faces in solids:
        if library.classify_body(body) not in ("torso", "shoulder", "head"):
            continue
        side = verts[:, 2] - height_m
        for a, b in ((0, 1), (1, 2), (2, 0)):
            sa, sb = side[faces[:, a]], side[faces[:, b]]
            hit = sa * sb < 0.0
            t = (sa[hit] / (sa[hit] - sb[hit]))[:, None]
            rear.extend(
                (verts[faces[hit, a]] * (1 - t) + verts[faces[hit, b]] * t)[:, 0]
            )
    return float(min(rear))


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
    assert got["chest"] >= ACCEPT.min_chest_over_hip * got["hip"], got


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
