"""The shirt slopes from the neck to the point of the shoulder (#15299)."""

from __future__ import annotations

import dataclasses
import functools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import document_from_dict, library
from src.shared.python.model_appearance.body_meshes import body_meshes
from src.shared.python.model_appearance.cross_section import body_sections
from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.model_appearance.torso_profile import (
    SHOULDER_SLOPE,
    ShoulderSlope,
    TorsoStation,
)
from src.shared.python.motion_matching.execution import spec_builder, subject_preset
from tests.unit.model_appearance.test_neck import ADDRESS, SPEC

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "model_appearance"
GIRDLE = json.loads((FIXTURES / "girdle_poses.json").read_text(encoding="utf-8"))
DOC = document_from_dict({"schema_version": "appearance-v1"})
HANG = {"LSInputY": math.pi / 2, "RSInputY": math.pi / 2}
SUBJECTS = ("default", "tour_average_driver")
SHIRT = {"torso": ("garment0",), "shoulder": ("garment0",), "upper_arm": ("blend0",)}
BIN_M = 0.01  # the shoulder line is read every centimetre, in bands this wide
NECK_SIDE_M = 0.06  # the line starts here, outside the collar (at 1.82 m)
MAX_LUMP_M = 0.004  # the line never climbs more than this going outwards
MAX_STEP_M = 0.012  # nor falls more than this per bin inside the joint: no cliff
MAX_LINK_PROUD_M = 0.003  # the shoulder link stays under the shirt by the neck
LINK_HIDDEN_TO = 0.6  # ... out to this share of the way to the joint
TOUR_SLOPE_DEG = (15.0, 28.0)  # trapezius slope of the tour-average shoulders
MAX_OFF_LINE_M = 0.008  # and how far the line may leave a straight slope
MAX_CHEST_DEPTH = 0.125  # shirt depth under the shoulders, over the stature
CHEST_BELOW_M = 0.06  # where that depth is read, under the joints (at 1.82 m)
MIN_SET_IN = 0.9  # sleeve cap inside the shirt: share kept when the girdle moves


@functools.cache
def _spec(subject: str) -> dict:
    if subject == "default":
        return SPEC
    kwargs = subject_preset.load_subject_preset(subject).builder_kwargs()
    return spec_builder.compose_anthropometric_document(**kwargs).document


def _dense(verts: np.ndarray, faces: np.ndarray, n: int = 5) -> np.ndarray:
    """Points over every triangle, so a band of the surface is never empty."""
    a, b, c = (verts[faces[:, i]] for i in range(3))
    grid = [(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)]
    return np.vstack([a * (1 - u - v) + b * u + c * v for u, v in grid])


@functools.cache
def _shirt(subject: str, pose: tuple[tuple[str, float], ...] = ()) -> dict:
    """Left shirt surfaces per part in a thorax frame (front, left, up), the
    left shoulder joint in it, and the closed meshes behind them."""
    spec = _spec(subject)
    declared = set(spec["coordinate_order"])
    poses = spec_body_poses(spec, {k: v for k, v in pose if k in declared})
    arms = [
        j for j in spec["joints"] if library.classify_body(j["child"]) == "upper_arm"
    ]
    thorax = arms[0]["parent"]
    while library.classify_body(thorax) != "torso":
        thorax = next(j["parent"] for j in spec["joints"] if j["child"] == thorax)
    section = body_sections(spec)[thorax]
    up, left = section.trunk_up_axis(), np.asarray(section.ml_axis, dtype=float)
    joints = [(np.linalg.inv(poses[thorax]) @ poses[j["child"]])[:3, 3] for j in arms]
    joint = max(joints, key=lambda p: float(p @ left))
    if float(joint @ left) < 0.0:  # the width axis may point either way
        left, joint = -left, min(joints, key=lambda p: float(p @ left))
    frame = np.stack([np.cross(left, up), left, up])
    out: dict = {"joint": frame @ joint, "meshes": {}}
    for mesh in body_meshes(spec, DOC):
        part = library.classify_body(mesh.body)
        if mesh.label not in SHIRT.get(part, ()):
            continue
        if part == "torso" and mesh.body != thorax:
            continue
        to_thorax = np.linalg.inv(poses[thorax]) @ poses[mesh.body]
        verts = mesh.mesh_in_body_frame.vertices @ to_thorax[:3, :3].T
        verts = (verts + to_thorax[:3, 3]) @ frame.T
        faces = mesh.mesh_in_body_frame.faces
        if part != "torso" and verts[:, 1].mean() < 0.0:
            out["meshes"].setdefault(f"right_{part}", []).append((verts, faces))
            continue
        out["meshes"].setdefault(part, []).append((verts, faces))
        points = _dense(verts, faces)
        out[part] = np.vstack([out[part], points]) if part in out else points
    return out


def _scale(subject: str) -> float:
    return float(_spec(subject)["subject"]["stature_m"]) / (
        SHOULDER_SLOPE.reference_stature_m
    )


def _line(subject: str, parts: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Top of ``parts`` above the shoulder joint, outwards from the neck (hang)."""
    shirt = _shirt(subject, tuple(HANG.items()))
    joint = shirt["joint"]
    out = np.arange(NECK_SIDE_M * _scale(subject), joint[1] + 4 * BIN_M, BIN_M)
    points = np.vstack([shirt[part] for part in parts])
    tops = []
    for y in out:
        band = points[np.abs(points[:, 1] - y) < BIN_M / 2]
        tops.append(band[:, 2].max() - joint[2] if len(band) else -np.inf)
    return out, np.asarray(tops)


@pytest.mark.parametrize("subject", SUBJECTS)
def test_the_shoulder_line_has_no_lump_and_no_cliff(subject: str) -> None:
    """The owner's complaint: sharp edges and bumps near the shoulders."""
    out, top = _line(subject, ("torso", "shoulder", "upper_arm"))
    step = np.diff(top)
    assert np.isfinite(top).all(), top
    assert step.max() <= MAX_LUMP_M, np.round(top, 3)
    inside = out[1:] <= _shirt(subject, tuple(HANG.items()))["joint"][1]
    assert -step[inside].min() <= MAX_STEP_M, np.round(top, 3)


@pytest.mark.parametrize("subject", SUBJECTS)
def test_the_shoulder_link_is_no_strap_beside_the_neck(subject: str) -> None:
    out, link = _line(subject, ("shoulder",))
    _, thorax = _line(subject, ("torso",))
    near = out <= LINK_HIDDEN_TO * _shirt(subject, tuple(HANG.items()))["joint"][1]
    assert np.isfinite(thorax[near]).all()
    assert (link[near] - thorax[near]).max() <= MAX_LINK_PROUD_M


def test_the_tour_average_shoulders_slope_like_a_trapezius() -> None:
    subject = "tour_average_driver"
    out, top = _line(subject, ("torso", "shoulder", "upper_arm"))
    joint = _shirt(subject, tuple(HANG.items()))["joint"]
    keep = out <= joint[1] - BIN_M
    slope, level = np.polyfit(out[keep], top[keep], 1)
    low, high = TOUR_SLOPE_DEG
    assert low <= math.degrees(math.atan(-slope)) <= high
    assert np.abs(top[keep] - (slope * out[keep] + level)).max() <= MAX_OFF_LINE_M


@pytest.mark.parametrize("subject", SUBJECTS)
def test_the_upper_chest_is_not_thick(subject: str) -> None:
    shirt = _shirt(subject, tuple(HANG.items()))
    level = shirt["joint"][2] - CHEST_BELOW_M * _scale(subject)
    band = shirt["torso"][np.abs(shirt["torso"][:, 2] - level) < BIN_M / 2]
    stature = float(_spec(subject)["subject"]["stature_m"])
    assert np.ptp(band[:, 0]) <= MAX_CHEST_DEPTH * stature


def test_the_point_of_the_shoulder_follows_the_subject() -> None:
    slope = SHOULDER_SLOPE
    low, high = slope.rise_m
    assert slope.rise(slope.acromion_below_neck_m) == low
    assert slope.rise(1.0) == high
    assert slope.rise(2.0, 2 * slope.reference_stature_m) == 2 * high
    assert slope.lift(2 * slope.reference_stature_m) == 2 * slope.point_lift_m
    with pytest.raises(ValueError, match="drop"):
        slope.rise(-0.01)
    for bad in (
        {"rise_m": (0.02, 0.01)},
        {"min_slope_deg": 90.0},
        {"corner": ((0.08, 0.1), (0.02, 0.2))},
        {"squareness": (1.5, 2.0)},
        {"collar_clear": 1.0},
        {"acromion_below_neck_m": 0.0},
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(slope, **bad)


def test_the_slope_stations_round_the_upper_back_and_keep_the_collar() -> None:
    base = tuple(TorsoStation(*row) for row in library.TORSO_PROFILES["thorax"])
    rows = SHOULDER_SLOPE.stations(base, 0.8, 0.07, 2.9)
    chest = max(range(len(rows)), key=lambda i: rows[i].width)
    widths = [row.width for row in rows[chest:]]
    assert rows[-1].along == base[-1].along and rows[-1].width == base[-1].width
    assert all(a >= b for a, b in zip(widths, widths[1:], strict=False))
    assert [row.along for row in rows] == sorted(row.along for row in rows)
    upper = [row.squareness for row in rows if row.squareness]
    assert upper and max(upper) < library.TORSO_SQUARENESS
    assert upper[-1] == SHOULDER_SLOPE.squareness[1]
    with pytest.raises(ValueError, match="shoulder line"):
        SHOULDER_SLOPE.stations(base, 1.2, 0.0, 2.9)


def _inside(points: np.ndarray, verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Which ``points`` a closed mesh contains (ray parity, a skew ray)."""
    ray = np.array([0.0131, 0.0077, 1.0]) / np.linalg.norm([0.0131, 0.0077, 1.0])
    a, b, c = (verts[faces[:, i]] for i in range(3))
    e1, e2 = b - a, c - a
    h = np.cross(ray, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    out = []
    for p in points:
        s = p - a
        u = np.einsum("ij,ij->i", s, h) * inv
        q = np.cross(s, e1)
        v, t = (q @ ray) * inv, np.einsum("ij,ij->i", q, e2) * inv
        hits = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        out.append(int(hits.sum()) % 2 == 1)
    return np.asarray(out)


def _set_in(subject: str, pose: dict[str, float]) -> float:
    """Share of the trail sleeve cap inside the thorax and its shoulder link."""
    meshes = _shirt(subject, tuple(sorted(pose.items())))["meshes"]
    cap = meshes["right_upper_arm"][0][0]
    inside = np.zeros(len(cap), dtype=bool)
    for verts, faces in meshes["torso"] + meshes["right_shoulder"]:
        inside |= _inside(cap, verts, faces)
    return float(inside.mean())


@pytest.mark.xfail(
    strict=True,
    reason="#15299: the thorax is one rigid solid, so a protracted or lifted "
    "shoulder girdle carries the sleeve cap out of the shirt; needs a surface "
    "that follows the girdle (see the model reference)",
)
@pytest.mark.parametrize("name", sorted(GIRDLE["poses"]))
def test_the_sleeve_cap_stays_set_in_the_shirt_as_the_girdle_moves(name: str) -> None:
    """Repo-only poses that reproduce the detached trail shoulder of a fit."""
    subject = "tour_average_driver"
    moved = {**ADDRESS, **GIRDLE["poses"][name]}
    assert _set_in(subject, moved) >= MIN_SET_IN * _set_in(subject, ADDRESS)
