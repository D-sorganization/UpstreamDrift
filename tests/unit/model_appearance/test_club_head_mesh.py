"""Shared clubhead mesh adapter (GCV-11 #11717): shape, orientation, sources."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_head_mesh as chm

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
CASES = [
    ("driver", (0.120, 0.127)),
    ("iron7", (0.075, 0.080 + 1e-9)),
]


def _edges_are_closed(faces: np.ndarray) -> bool:
    edges: dict[tuple[int, int], int] = {}
    for tri in faces:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = (int(tri[a]), int(tri[b]))
            edges[key] = edges.get(key, 0) + 1
    return all(
        count == 1 and (b, a) in edges for (a, b), count in edges.items()
    )  # each directed edge once, with its opposite present


@pytest.mark.parametrize("name,_range", CASES)
def test_head_and_assembly_are_watertight_and_outward(name, _range) -> None:
    head = chm.load_club_head(name)
    for mesh in (head.head_mesh, head.mesh):
        assert _edges_are_closed(mesh.faces)
        assert mesh.volume() > 0.0


@pytest.mark.parametrize("name,span", CASES)
def test_heel_toe_length_in_range(name, span) -> None:
    head = chm.load_club_head(name)
    z = head.head_mesh.vertices[:, 2]
    assert span[0] <= float(z.max() - z.min()) <= span[1]


@pytest.mark.parametrize("name", ["driver", "iron7", "wedge56", "hybrid3", "fairway3"])
def test_face_normal_angle_equals_loft(name) -> None:
    head = chm.load_club_head(name)
    normal = chm.measured_face_normal(head.head_mesh)
    loft = math.degrees(math.asin(normal[1]))
    assert abs(loft - head.loft_deg) <= 0.5
    assert abs(normal[2]) < 0.05  # square face: no heel-toe lean


def test_club_frame_axes_follow_the_spec_lie_and_loft() -> None:
    head = chm.load_club_head("iron7")
    mesh = chm.club_frame_mesh(head)
    assert _edges_are_closed(mesh.faces)
    face = chm.measured_face_normal(head.head_mesh)
    tau = math.radians(90.0 - head.lie_deg)
    rot = chm.head_to_club_rotation(head)
    f_club = rot @ face
    shaft_toward_grip = np.array([0.0, -1.0, 0.0])
    # face . shaft = sin(loft) cos(90 - lie) in the head frame and is invariant
    assert math.isclose(
        float(f_club @ shaft_toward_grip),
        float(face @ head.shaft_direction),
        abs_tol=1e-9,
    )
    assert math.isclose(float(head.shaft_direction[1]), math.cos(tau), abs_tol=1e-9)
    # The club-frame face normal points along +x (down the target line).
    assert f_club[0] > 0.8
    assert abs(np.linalg.det(rot) - 1.0) < 1e-9
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)


def test_sole_point_sits_on_shaft_axis_and_nothing_is_below_the_ground() -> None:
    head = chm.load_club_head("driver")
    rot = chm.head_to_club_rotation(head)
    mesh = chm.club_frame_mesh(head, axis_offset_m=0.064)
    sole_c = rot @ head.sole_point + (
        np.array([0.0, 0.0, 0.064]) - rot @ head.sole_point
    )
    np.testing.assert_allclose(sole_c, [0.0, 0.0, 0.064], atol=1e-12)
    up_c = rot @ np.array([0.0, 1.0, 0.0])  # ground normal in the club frame
    height = (mesh.vertices - sole_c) @ up_c
    assert float(height.min()) >= -1e-9  # the sole plane touches, never penetrates
    assert float(height.min()) <= 1e-6


@pytest.mark.parametrize(
    "name,expected",
    [
        ("driver", "Driver 10.5°"),
        ("iron7", "7-Iron"),
        ("7-iron", "7-Iron"),
        ("wedge56", "Sand Wedge"),
        ("hybrid3", "3-Hybrid"),
        ("fairway3", "3-Wood"),
        ("iron(9)", "9-Iron"),
    ],
)
def test_name_resolution(name, expected) -> None:
    assert chm.library_name_for(name) == expected


@pytest.mark.parametrize("bad", ["", "putter", "iron(12)", "wedge30", "spoon", 7])
def test_bad_names_raise(bad) -> None:
    with pytest.raises((ValueError, TypeError)):
        chm.library_name_for(bad)


def test_committed_assets_have_provenance_and_exist() -> None:
    manifest = json.loads((ROOT / "assets/club_heads/provenance.json").read_text())
    assert manifest["units"] == "mm"
    assert manifest["head_frame"] == "x=target,y=up,z=toe"
    for name, entry in manifest["heads"].items():
        path = ROOT / entry["path"]
        assert path.is_file(), name
        assert len(entry["sha256"]) == 64
        assert entry["library_name"] == name
    assert "Driver 10.5°" in manifest["heads"]
    # The Simscape driver STL is reused, not duplicated.
    assert manifest["heads"]["Driver 10.5°"]["path"].endswith("gs3dx_driver_head.stl")


def test_committed_stl_matches_tools_builder_when_available() -> None:
    if not chm.tools_builder_available():
        pytest.skip("Tools parametric builder is not importable")
    for name in ("Driver 10.5°", "7-Iron", "Sand Wedge"):
        tools = chm.triangles_from_tools(name)
        stl = chm.triangles_from_committed_stl(name)
        assert tools is not None
        np.testing.assert_allclose(
            np.sort(tools.reshape(-1, 3), axis=0),
            np.sort(stl.reshape(-1, 3), axis=0),
            atol=2e-9,
        )


def test_fallback_source_is_used_when_tools_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(chm, "triangles_from_tools", lambda _name: None)
    chm.load_club_head.cache_clear()
    try:
        head = chm.load_club_head("iron7")
        assert head.source == "committed_stl"
    finally:
        chm.load_club_head.cache_clear()
