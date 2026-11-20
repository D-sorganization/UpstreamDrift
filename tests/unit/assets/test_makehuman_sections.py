"""Cross-sections of a skinned mesh along a bone axis (#15299)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ASSETS = Path(__file__).resolve().parents[3] / "assets"
sys.path.insert(0, str(ASSETS / "makehuman_cc0"))

import mh_sections  # noqa: E402
from mh_rig_io import MakeHumanRig  # noqa: E402

pytestmark = pytest.mark.unit

RINGS, SIDES, LENGTH = 41, 96, 4.0
HALF_WIDTH, HALF_DEPTH, FORWARD, EXPONENT = 1.2, 0.8, 0.3, 3.0
FIT = {"hull_samples": 240, "squareness_bounds": [1.2, 8.0]}
SEGMENT = {
    "start": "limb/head",
    "end": "limb/tail",
    "slice_bones": ["limb"],
    "min_weight": 0.5,
    "along": [0.1, 0.9],
    "stations": 5,
    "reach": 1.0,
}


def _taper(t: float) -> float:
    """Radius scale of the tube: a belly at 34 % of its length."""
    return 1.0 - 0.6 * (t - 0.34) ** 2


def _tube(offset: float, bone: str, count: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """A superellipse tube along +y, its section shifted to the front (+z)."""
    angle = np.linspace(0.0, 2.0 * np.pi, SIDES, endpoint=False)
    cos, sin = np.cos(angle), np.sin(angle)
    unit_w = np.sign(cos) * np.abs(cos) ** (2.0 / EXPONENT)
    unit_d = np.sign(sin) * np.abs(sin) ** (2.0 / EXPONENT)
    rows = []
    for t in np.linspace(-0.05, 1.05, RINGS):
        k = _taper(t)
        rows.append(
            np.column_stack(
                [
                    offset + HALF_WIDTH * k * unit_w,
                    np.full(SIDES, t * LENGTH),
                    FORWARD + HALF_DEPTH * k * unit_d,
                ]
            )
        )
    quads = [
        [count + r * SIDES + s, count + r * SIDES + (s + 1) % SIDES]
        + [count + (r + 1) * SIDES + (s + 1) % SIDES, count + (r + 1) * SIDES + s]
        for r in range(RINGS - 1)
        for s in range(SIDES)
    ]
    ids = range(count, count + RINGS * SIDES)
    return np.vstack(rows), np.asarray(quads), {bone: [(i, 1.0) for i in ids]}


def _rig(rotation: np.ndarray | None = None) -> MakeHumanRig:
    verts, quads, weights = _tube(0.0, "limb", 0)
    other_v, other_q, other_w = _tube(2.0, "other", len(verts))
    ends = np.array([[0.0, 0.0, 0.0], [0.0, LENGTH, 0.0]])
    vertices = np.vstack([verts, other_v, ends])
    if rotation is not None:
        vertices = vertices @ rotation.T
    last = len(vertices) - 1
    return MakeHumanRig(
        vertices=vertices,
        body_quads=np.vstack([quads, other_q]),
        bones={"limb": {"head": "a", "tail": "b", "parent": None}},
        joints={"a": [last - 1], "b": [last]},
        weights={**weights, **other_w},
        skeleton_licence="CC0",
        weights_licence="CC0",
    )


def test_sections_recover_the_tube_and_ignore_the_other_limb() -> None:
    table = mh_sections.segment_table(_rig(), SEGMENT, np.array([0.0, 0.0, 1.0]), FIT)
    assert table.length == pytest.approx(LENGTH)
    assert [s.along for s in table.sections] == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])
    for section in table.sections:
        k = _taper(section.along)
        assert section.half_width == pytest.approx(HALF_WIDTH * k, rel=5e-3)
        assert section.depth_front == pytest.approx(FORWARD + HALF_DEPTH * k, rel=5e-3)
        assert section.depth_back == pytest.approx(HALF_DEPTH * k - FORWARD, rel=5e-3)
        assert section.width_centre == pytest.approx(0.0, abs=1e-6)
        assert section.squareness == pytest.approx(EXPONENT, abs=0.1)
        assert 4.0 * k < section.girth / HALF_DEPTH < 2.0 * np.pi * 1.5 * k
    girths = [s.girth for s in table.sections]
    assert table.belly_along == pytest.approx(0.3) and max(girths) == girths[1]


def test_sections_do_not_depend_on_the_pose() -> None:
    axis = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = np.eye(3) + np.sin(0.8) * skew + (1.0 - np.cos(0.8)) * skew @ skew
    front = np.array([0.0, 0.0, 1.0])
    rest = mh_sections.segment_table(_rig(), SEGMENT, front, FIT)
    posed = mh_sections.segment_table(_rig(rotation), SEGMENT, rotation @ front, FIT)
    for a, b in zip(rest.sections, posed.sections, strict=True):
        assert (b.half_width, b.depth_front, b.depth_back, b.girth) == pytest.approx(
            (a.half_width, a.depth_front, a.depth_back, a.girth), rel=1e-9
        )
        assert b.squareness == pytest.approx(a.squareness, abs=1e-4)


def test_joint_reference_and_frame_contracts() -> None:
    rig = _rig()
    np.testing.assert_allclose(
        mh_sections.joint_point(rig, "limb/tail"), [0, LENGTH, 0]
    )
    for bad in ("limb", "limb/middle", "femur/head"):
        with pytest.raises(ValueError):
            mh_sections.joint_point(rig, bad)
    with pytest.raises(ValueError, match="length"):
        mh_sections.slice_frame(np.zeros(3), np.zeros(3), np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="front"):
        mh_sections.slice_frame(np.zeros(3), np.array([0.0, 0.0, 2.0]), np.ones(3) * 0)
    with pytest.raises(ValueError, match="no bone weights"):
        mh_sections.weighted_vertices(rig, ["femur"], 0.5)
    with pytest.raises(ValueError, match="min_weight"):
        mh_sections.weighted_vertices(rig, ["limb"], 0.0)


def test_a_plane_that_misses_the_skin_is_an_error() -> None:
    front = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="does not cut"):
        mh_sections.segment_table(_rig(), {**SEGMENT, "along": [0.5, 3.0]}, front, FIT)
    with pytest.raises(ValueError, match="stations"):
        mh_sections.segment_table(_rig(), {**SEGMENT, "stations": 1}, front, FIT)
    near = {**SEGMENT, "reach": 0.01}
    with pytest.raises(ValueError, match="does not cut"):
        mh_sections.segment_table(_rig(), near, front, FIT)


def test_bone_patterns_expand_in_order_and_must_match() -> None:
    rig = _rig()
    assert mh_sections.expand_bones(rig, ["o*", "limb", "*i*"]) == ["other", "limb"]
    with pytest.raises(ValueError, match="no weighted MakeHuman bone"):
        mh_sections.expand_bones(rig, ["femur*"])


def test_volume_of_a_unit_cube() -> None:
    corners = [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6]]
    faces += [[0, 2, 6, 4], [1, 5, 7, 3]]
    rig = MakeHumanRig(np.array(corners, float), np.array(faces), {}, {}, {}, "", "")
    assert mh_sections.skin_volume(rig) == pytest.approx(1.0)
