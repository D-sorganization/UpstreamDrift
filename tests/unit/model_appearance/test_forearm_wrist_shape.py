"""A forearm that tapers to a wrist of human girth at the wrist joint (#15299).

ANSUR II male wrist circumference: mean 175.9 mm, 5th to 95th percentile 162 to
191 mm. Bounds are ``forearm_wrist_shape`` settings. Appearance only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import document_from_dict, library
from src.shared.python.model_appearance import skinned_club_hands as sch
from src.shared.python.model_appearance import skinned_wrist_sections as sws
from src.shared.python.model_appearance.body_meshes import body_meshes
from src.shared.python.model_appearance.hand_backends.qa import (
    load_hand_visual_quality,
)
from src.shared.python.motion_matching.execution.skinned_hand_wrists import (
    apply_skinned_hand_wrists,
)
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton

pytestmark = [pytest.mark.unit, pytest.mark.slow, pytest.mark.timeout(600)]
SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
)
SHAPE: dict[str, Any] = dict(load_hand_visual_quality().get("forearm_wrist_shape", {}))
STATION_M = 0.0025  # spacing of the sections cut along the limb


@pytest.fixture(scope="module")
def skinned() -> dict[str, Any]:
    return apply_skinned_hand_wrists(json.loads(SPEC.read_text(encoding="utf-8")))


def _meshes(spec: dict[str, Any]) -> dict[tuple[str, str], Any]:
    doc = document_from_dict({"schema_version": "appearance-v1"})
    club = ca.assembly_from_spec(spec)
    return {
        (m.body, m.label): m.mesh_in_body_frame for m in body_meshes(spec, doc, club)
    }


def _across(rel: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """``rel`` as (x, y) in the plane across ``axis``."""
    x = np.cross(axis, [1.0, 0.0, 0.0])
    if np.linalg.norm(x) < 0.5:
        x = np.cross(axis, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    return np.c_[rel @ x, rel @ np.cross(axis, x)]


def _cut(mesh: Any, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Where the plane through ``origin`` across ``axis`` cuts the mesh, (x, y)."""
    v, f = mesh.vertices, mesh.faces
    height = (v - origin) @ axis
    points = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ha, hb = height[f[:, a]], height[f[:, b]]
        hit = ha * hb < 0
        t = (ha[hit] / (ha[hit] - hb[hit]))[:, None]
        points.append(v[f[hit, a]] + t * (v[f[hit, b]] - v[f[hit, a]]))
    return _across(np.concatenate(points) - origin, axis)


def _girth(section: np.ndarray) -> float:
    """Tape girth (m): the perimeter of the section's convex hull."""
    ring = section[ConvexHull(section).vertices]
    return float(np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1).sum())


def _forearms(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per hand: its forearm's two capsules, near then far, and their skins."""
    skins = _meshes(spec)
    caps = {c.body: c for c in derive_visual_skeleton(spec).capsules}
    parents = {j["child"]: j["parent"] for j in spec["joints"]}
    out = {}
    wrists = sws.straight_wrists(spec, ca.assembly_from_spec(spec))
    for side, (far, to_forearm) in wrists.items():
        bodies = (parents[far], far)
        out[side] = {
            "caps": [caps[b] for b in bodies],
            "skins": [skins[b, "skin0"] for b in bodies],
            "to_forearm": to_forearm,
        }
    assert set(out) == {"lead", "trail"}
    return out


def _girths(arm: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Distance from the wrist joint (m, negative up the arm) and skin girth (m)."""
    lengths = [cap.length_m() for cap in arm["caps"]]
    where, girths = [], []
    for cap, skin, before in zip(
        arm["caps"], arm["skins"], (0.0, lengths[0]), strict=True
    ):
        axis = np.subtract(cap.end_m, cap.start_m) / cap.length_m()
        for s in np.arange(0.5 * STATION_M, cap.length_m(), STATION_M):
            where.append(before + s - sum(lengths))
            girths.append(_girth(_cut(skin, cap.start_m + s * axis, axis)))
    return np.array(where), np.array(girths)


@pytest.fixture(scope="module")
def profiles(skinned) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {side: _girths(arm) for side, arm in _forearms(skinned).items()}


def test_the_forearm_narrows_all_the_way_to_the_wrist(profiles) -> None:
    flat = float(SHAPE["constant_girth_tolerance_m"])
    for where, girth in profiles.values():
        belly = int(girth.argmax())
        steps = np.diff(girth[belly:])
        assert steps.max() <= flat  # never wider again toward the wrist
        narrowest = where[np.flatnonzero(girth <= girth.min() + flat)[0]]
        assert -float(SHAPE["max_minimum_from_joint_m"]) <= narrowest <= 0.0
        crest = belly + int(np.argmax(girth[belly:] < girth[belly] - flat))
        longest = max(  # stations over which the girth stays within ``flat``
            int(np.argmax(np.abs(girth[i:] - girth[i]) > flat) or len(girth) - i)
            for i in range(crest, len(girth))
        )
        assert longest * STATION_M <= float(SHAPE["max_constant_run_m"])


def test_the_wrist_has_the_girth_of_a_human_wrist(skinned, profiles) -> None:
    # ANSUR II male wrist circumference, scaled to the subject by stature.
    norm = float(SHAPE["norm_girth_m"]) * (
        float(skinned["subject"]["stature_m"]) / float(SHAPE["norm_stature_m"])
    )
    low, high = (float(k) * norm for k in SHAPE["girth_band"])
    for _, girth in profiles.values():
        assert low <= girth.min() <= high


def test_the_forearm_is_fullest_in_its_upper_third(skinned, profiles) -> None:
    # ANSUR II male forearm circumference (flexed), scaled by stature; a
    # relaxed forearm lies between its 5th percentile and its mean.
    norm = float(SHAPE["belly_norm_girth_m"]) * (
        float(skinned["subject"]["stature_m"]) / float(SHAPE["norm_stature_m"])
    )
    low, high = (float(k) * norm for k in SHAPE["belly_girth_band"])
    near, far = (float(k) for k in SHAPE["belly_from_elbow"])
    for where, girth in profiles.values():
        length = STATION_M * len(girth)
        assert near <= 1.0 + where[girth.argmax()] / length <= far
        assert low <= girth.max() <= high
        assert girth[0] < girth.max() - 0.01  # the elbow is thinner than the belly


def _polar(section: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.arctan2(section[:, 1], section[:, 0]))
    section = section[order]
    return np.arctan2(section[:, 1], section[:, 0]), np.hypot(*section.T)


def _rim_gaps(spec: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per hand, wrist straight: how far the forearm's skin stands outside each
    point of the hand's cut rim (m), in the plane of the rim."""
    club = ca.assembly_from_spec(spec)
    hands, reach = sch.skinned_hand_meshes(club), sch.forearm_reach_m(club)
    cuff, out = sch.forearm_cuff_m(), {}
    for side, arm in _forearms(spec).items():
        cap, skin = arm["caps"][1], arm["skins"][1]
        axis = np.subtract(cap.end_m, cap.start_m) / cap.length_m()
        wrist, along = reach[side]
        assert arm["to_forearm"] @ along == pytest.approx(-axis, abs=1e-6)
        mesh = hands[f"{side}_hand"]
        rel = mesh.vertices[np.unique(mesh.faces)] - wrist
        rim = rel[np.abs(rel @ along - cuff) < 1e-9] @ arm["to_forearm"].T + cap.end_m
        origin = cap.end_m - cuff * axis
        flat = _across(rim - origin, axis)
        flat = flat[ConvexHull(flat).vertices]  # not the point closing the rim
        fore = _cut(skin, origin + 1e-7 * axis, axis)
        centre = 0.5 * (fore.max(axis=0) + fore.min(axis=0))
        angle, radius = _polar(fore - centre)
        hand_angle, hand_radius = _polar(flat - centre)
        out[side] = (
            np.interp(hand_angle, angle, radius, period=2.0 * np.pi) - hand_radius
        )
    return out


def test_the_hand_starts_where_the_forearm_ends(skinned) -> None:
    for gap in _rim_gaps(skinned).values():
        assert gap.min() >= 0.0  # the rim never shows outside the forearm
        assert gap.max() <= float(SHAPE["max_junction_step_m"])


def test_only_the_forearm_and_the_hands_change(skinned, monkeypatch) -> None:
    new = _meshes(skinned)
    old_quality = dict(load_hand_visual_quality())
    old_quality["forearm_wrist_margin"] = SHAPE["cover"]
    old_quality["forearm_wrist_shape"] = {
        **SHAPE,
        "taper_end": library.taper_end("forearm"),
        "belly": library.belly("forearm"),
    }
    for module in (sch, sws):
        monkeypatch.setattr(module, "load_hand_visual_quality", lambda: old_quality)
    old = _meshes(skinned)
    assert set(old) == set(new)
    forearms = {c.body for arm in _forearms(skinned).values() for c in arm["caps"]}
    changed = {
        key
        for key in new
        if new[key].vertices.shape != old[key].vertices.shape
        or not np.array_equal(new[key].vertices, old[key].vertices)
    }
    assert {body for body, _ in changed if body in forearms} == forearms
    assert {label for body, label in changed if body not in forearms} == {
        "lead_hand",
        "trail_hand",
    }


def test_bad_wrist_shape_settings_are_refused(monkeypatch) -> None:
    for key, bad in (
        ("taper_end", 0.0),
        ("cover", 0.5),
        ("hand_blend_m", 0.0),
        ("belly", [0.2, 0.0, 0.3]),
    ):
        quality = {"forearm_wrist_margin": 1.2, "forearm_wrist_shape": {**SHAPE}}
        quality["forearm_wrist_shape"][key] = bad
        for module in (sch, sws):
            monkeypatch.setattr(module, "load_hand_visual_quality", lambda q=quality: q)
        with pytest.raises(ValueError, match=key):
            {"taper_end": sws.wrist_taper_end, "belly": sws.forearm_belly}.get(
                key, sch.wrist_blend
            )()
