"""Shared grip-hand renderer (#15150): settings, scene, camera rig, engines.

The GL tests skip only when no MuJoCo GL context can be created, and the skip
message carries the GL error text. A host with ``libOSMesa`` or ``libEGL``
runs them.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
)
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hand_mesh import build_grip_hand_mesh
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends import render
from src.shared.python.model_appearance.hand_backends.render_rig import (
    GripDrawing,
    corner_normals,
    focus_vertices_of,
    orthonormal_basis,
    projected_fill,
    result_basis,
    skin_vertices_of,
    vertex_normals,
)
from src.shared.python.model_appearance.hand_backends.render_settings import (
    DEFAULT_SETTINGS_PATH,
    REQUIRED_LIGHTS,
    REQUIRED_VIEWS,
    load_hand_render_settings,
    settings_from_mapping,
)
from src.shared.python.model_appearance.library import MATERIALS

pytestmark = pytest.mark.unit

SETTINGS = load_hand_render_settings()
RAW = json.loads(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))


def _hand(side: str = "lead"):
    return build_grip_hand_mesh(Hand(side), grip_radius_m=SETTINGS.grip_radius_m)


# --- 1. settings loader -----------------------------------------------------


def test_settings_file_meets_the_quality_bar() -> None:
    assert SETTINGS.frame_px >= 1024
    assert SETTINGS.fill_fraction == 0.70
    assert tuple(SETTINGS.views) == REQUIRED_VIEWS
    assert SETTINGS.rolls_deg == (-20.0, 0.0, 20.0)
    assert set(REQUIRED_LIGHTS) <= set(SETTINGS.lights)
    assert SETTINGS.grip_radius_m == 0.014
    assert SETTINGS.grip_rgba == tuple(MATERIALS["grip_rubber"].rgba)
    assert SETTINGS.shaft_radius_m < SETTINGS.grip_radius_m
    assert len(SETTINGS.settings_hash) == 64


def test_missing_settings_file_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        load_hand_render_settings(tmp_path / "absent.json")


@pytest.mark.parametrize("token", ["NaN", "Infinity", "1e999"])
def test_non_finite_number_raises(tmp_path, token: str) -> None:
    text = DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(text.replace('"fov_deg": 28.0', f'"fov_deg": {token}'), "utf-8")
    assert token in bad.read_text("utf-8")
    with pytest.raises(ValueError):
        load_hand_render_settings(bad)


def test_unknown_view_name_raises() -> None:
    raw = json.loads(json.dumps(RAW))
    raw["views"]["knuckles"] = raw["views"]["palm"]
    with pytest.raises(ValueError, match="unknown view"):
        settings_from_mapping(raw)
    with pytest.raises(ValueError, match="unknown view"):
        SETTINGS.view("knuckles")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("frame_px", 0),
        ("fill_fraction", 1.5),
        ("grip_radius_m", -0.01),
        ("shaft_radius_m", 0.02),
        ("engine_order", ["pyrender"]),
        ("grip_material", "no_such_material"),
        ("shadows", "yes"),
    ],
)
def test_bad_setting_values_raise(key: str, value: object) -> None:
    with pytest.raises(ValueError):
        SETTINGS.with_overrides(**{key: value})


def test_missing_key_and_missing_light_raise() -> None:
    raw = json.loads(json.dumps(RAW))
    del raw["lights"]["rim"]
    with pytest.raises(ValueError, match="rim"):
        settings_from_mapping(raw)
    raw = json.loads(json.dumps(RAW))
    del raw["fov_deg"]
    with pytest.raises(ValueError, match="fov_deg"):
        settings_from_mapping(raw)


def test_overrides_change_the_hash_and_reject_unknown_keys() -> None:
    other = SETTINGS.with_overrides(fov_deg=30.0)
    assert other.fov_deg == 30.0
    assert other.settings_hash != SETTINGS.settings_hash
    with pytest.raises(ValueError, match="unknown"):
        SETTINGS.with_overrides(not_a_setting=1)


# --- 2. scene assembly ------------------------------------------------------


@pytest.mark.parametrize("side", ["lead", "trail"])
def test_capsule_scene_has_skin_grip_and_shaft(side: str) -> None:
    parts = render.assemble_scene(_hand(side), settings=SETTINGS)
    assert [p.kind for p in parts] == ["skin", "grip", "shaft"]
    skin, grip, shaft = parts
    assert skin.rgba == SETTINGS.skin_rgba
    assert grip.rgba == SETTINGS.grip_rgba
    radius = np.hypot(grip.mesh.vertices[:, 0], grip.mesh.vertices[:, 1])
    assert radius.max() == pytest.approx(SETTINGS.grip_radius_m)
    z_skin = skin.mesh.vertices[:, 2]
    z_grip = grip.mesh.vertices[:, 2]
    margin = SETTINGS.grip_length_margin_m
    assert z_grip.min() == pytest.approx(z_skin.min() - margin)
    assert z_grip.max() == pytest.approx(z_skin.max() + margin)
    z_shaft = shaft.mesh.vertices[:, 2]
    assert z_shaft.min() == pytest.approx(z_grip.max())
    assert z_shaft.max() == pytest.approx(z_grip.max() + SETTINGS.shaft_extension_m)


def test_shaft_roll_matches_apply_shaft_roll() -> None:
    hand = _hand()
    flat = render.assemble_scene(hand, settings=SETTINGS)[0].mesh.vertices
    np.testing.assert_array_equal(flat, hand.mesh.vertices)
    rolled = render.assemble_scene(hand, shaft_roll_deg=20.0, settings=SETTINGS)
    rotation = apply_shaft_roll(np.eye(3), APPEARANCE_GRIP_SHAFT_AXIS, 20.0)
    column = hand.mesh.vertices[7].reshape(3, 1)
    np.testing.assert_allclose(
        rolled[0].mesh.vertices[7], (rotation @ column).ravel(), atol=1e-12
    )
    np.testing.assert_allclose(rolled[0].mesh.vertices[:, 2], flat[:, 2], atol=1e-12)
    assert not np.allclose(rolled[0].mesh.vertices[:, :2], flat[:, :2])
    np.testing.assert_array_equal(
        rolled[1].mesh.vertices,
        render.assemble_scene(hand, settings=SETTINGS)[1].mesh.vertices,
    )


def test_duck_typed_results_resolve_mesh_and_colour() -> None:
    mesh = _hand().mesh
    shadow = SimpleNamespace(mesh=mesh, material_rgba=(0.2, 0.3, 0.4, 1.0))
    procedural = SimpleNamespace(
        mesh=mesh, material="grip_rubber", grip_radius_m=0.0145
    )
    skinned = SimpleNamespace(mesh=mesh, skin_rgb=(0.5, 0.4, 0.3))
    assert render.assemble_scene(shadow, settings=SETTINGS)[0].rgba == (
        0.2,
        0.3,
        0.4,
        1.0,
    )
    parts = render.assemble_scene(procedural, settings=SETTINGS)
    assert parts[0].rgba == tuple(MATERIALS["grip_rubber"].rgba)
    radius = np.hypot(parts[1].mesh.vertices[:, 0], parts[1].mesh.vertices[:, 1])
    assert radius.max() == pytest.approx(0.0145)
    assert render.assemble_scene(skinned, settings=SETTINGS)[0].rgba == (
        0.5,
        0.4,
        0.3,
        1.0,
    )
    bones = {c.name: c.mesh for c in _hand().components}
    by_bone = render.assemble_scene(bones, settings=SETTINGS)
    assert [p.kind for p in by_bone].count("skin") == len(bones)
    assert by_bone[0].rgba == SETTINGS.skin_rgba
    lists = SimpleNamespace(vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist())
    assert render.assemble_scene(lists, settings=SETTINGS)[0].kind == "skin"


def test_bad_scene_inputs_raise() -> None:
    hand = _hand()
    empty = SimpleNamespace(
        mesh=Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    )
    with pytest.raises(ValueError, match="vertices"):
        render.assemble_scene(empty, settings=SETTINGS)
    with pytest.raises(ValueError):
        render.assemble_scene(hand, shaft_roll_deg=float("nan"), settings=SETTINGS)
    with pytest.raises(ValueError):
        render.assemble_scene(hand, shaft_roll_deg=181.0, settings=SETTINGS)
    with pytest.raises(TypeError):
        render.assemble_scene(hand, shaft_roll_deg="20", settings=SETTINGS)
    with pytest.raises(TypeError):
        render.assemble_scene(object(), settings=SETTINGS)
    with pytest.raises(ValueError):
        render.assemble_scene({}, settings=SETTINGS)
    with pytest.raises(ValueError, match="library material"):
        render.assemble_scene(
            SimpleNamespace(mesh=hand.mesh, material="nope"), settings=SETTINGS
        )
    with pytest.raises(ValueError):
        render.build_club_parts(0.1, -0.1, settings=SETTINGS)
    with pytest.raises(ValueError, match="view"):
        render.render_view(hand, "sideways")
    with pytest.raises(ValueError, match="side"):
        render.render_view(hand, "palm", side="left")
    with pytest.raises(TypeError):
        render.render_scene([object()], "palm")
    with pytest.raises(ValueError, match="no skin"):
        render.render_scene(
            render.build_club_parts(0.0, 0.1, settings=SETTINGS), "palm"
        )


# --- 2b. the real golf grip (tapered rubber, butt cap, vent) ----------------

GRIP_KNOTS = (
    (-0.060, 0.01219),
    (-0.0092, 0.01143),
    (0.0924, 0.00991),
    (0.2067, 0.00889),
)
GRIP = GripDrawing(
    knots=GRIP_KNOTS,
    cap_radius_m=0.0133,
    cap_height_m=0.005,
    cap_round_m=0.002,
    vent_radius_m=0.0012,
)
GRIP_FIELDS = {
    "knots": GRIP_KNOTS,
    "cap_radius_m": 0.0133,
    "cap_height_m": 0.005,
    "cap_round_m": 0.002,
    "vent_radius_m": 0.0012,
}


def _named(parts, name: str):
    return [p for p in parts if p.name == name]


def _radius(vertices: np.ndarray) -> np.ndarray:
    return np.hypot(vertices[:, 0], vertices[:, 1])


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("knots", ((0.0, 0.012),), "two knots"),
        ("knots", ((0.0, 0.012), (0.0, 0.011)), "ascending"),
        ("knots", ((0.1, 0.012), (0.0, 0.011)), "ascending"),
        ("knots", ((0.0, 0.012), (0.1, 0.0)), "positive"),
        ("knots", ((0.0, 0.012), (0.1, float("nan"))), "finite"),
        ("knots", ((float("inf"), 0.012), (0.1, 0.01)), "finite"),
        ("knots", ((0.0, 0.012, 1.0), (0.1, 0.01)), "pairs"),
        ("cap_radius_m", 0.012, "cap_radius_m"),
        ("cap_height_m", -0.001, "cap_height_m"),
        ("cap_round_m", -0.001, "cap_round_m"),
        ("cap_round_m", 0.006, "cap_round_m"),
        ("vent_radius_m", -0.001, "vent_radius_m"),
        ("vent_radius_m", 0.0133, "vent_radius_m"),
        ("vent_radius_m", float("nan"), "finite"),
    ],
)
def test_grip_drawing_contract_errors(field: str, value: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        GripDrawing(**{**GRIP_FIELDS, field: value})


def test_grip_drawing_is_frozen_and_normalises_its_knots() -> None:
    drawing = GripDrawing(**{**GRIP_FIELDS, "knots": [list(k) for k in GRIP_KNOTS]})
    assert drawing.knots == GRIP_KNOTS
    assert drawing == GRIP
    with pytest.raises(AttributeError):
        drawing.cap_radius_m = 0.02  # type: ignore[misc]


def test_club_without_a_drawing_is_the_plain_cylinder() -> None:
    radius = SETTINGS.grip_radius_m
    margin = SETTINGS.grip_length_margin_m
    default = render.build_club_parts(-0.03, 0.07, settings=SETTINGS)
    explicit = render.build_club_parts(-0.03, 0.07, settings=SETTINGS, grip=None)
    assert [p.kind for p in default] == ["grip", "shaft"]
    for a, b in zip(default, explicit, strict=True):
        assert np.array_equal(a.mesh.vertices, b.mesh.vertices)
        assert np.array_equal(a.mesh.faces, b.mesh.faces)
        assert a.rgba == b.rgba
    grip, shaft = default
    sides = SETTINGS.club_sides
    assert grip.mesh.vertices.shape == (2 * sides + 2, 3)
    assert shaft.mesh.vertices.shape == (2 * sides + 2, 3)
    assert np.allclose(_radius(grip.mesh.vertices[: 2 * sides]), radius)
    assert np.isclose(grip.mesh.vertices[:, 2].min(), -0.03 - margin)
    assert np.isclose(grip.mesh.vertices[:, 2].max(), 0.07 + margin)
    assert np.isclose(shaft.mesh.vertices[:, 2].min(), 0.07 + margin)
    assert np.isclose(
        shaft.mesh.vertices[:, 2].max(), 0.07 + margin + SETTINGS.shaft_extension_m
    )
    assert grip.rgba == SETTINGS.grip_rgba and shaft.rgba == SETTINGS.shaft_rgba


def test_drawn_grip_parts_kinds_and_colours() -> None:
    parts = render.build_club_parts(-0.03, 0.07, settings=SETTINGS, grip=GRIP)
    assert [p.name for p in parts] == ["grip_rubber", "grip_cap", "grip_vent", "shaft"]
    assert [p.kind for p in parts] == ["grip", "grip", "grip", "shaft"]
    rubber, cap, vent, shaft = parts
    assert rubber.rgba == SETTINGS.grip_rgba
    assert cap.rgba == SETTINGS.grip_cap_rgba
    assert vent.rgba == SETTINGS.grip_vent_rgba
    assert shaft.rgba == SETTINGS.shaft_rgba
    for part in parts:
        checked = render.validated_part(part)
        assert checked.name == part.name
        assert np.isfinite(part.mesh.vertices).all()
        faces = part.mesh.faces
        assert faces.min() >= 0 and faces.max() < len(part.mesh.vertices)
        assert len(np.unique(faces)) == len(part.mesh.vertices)
        assert part.mesh.volume() > 0.0
        tri = part.mesh.vertices[faces]
        area = np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        )
        assert area.min() > 0.0


def test_drawn_rubber_follows_the_knots_and_tapers_monotonically() -> None:
    parts = render.build_club_parts(-0.03, 0.07, settings=SETTINGS, grip=GRIP)
    (rubber,) = _named(parts, "grip_rubber")
    vertices = rubber.mesh.vertices
    radius = _radius(vertices)
    assert np.isclose(vertices[:, 2].min(), GRIP_KNOTS[0][0], atol=1e-12)
    assert np.isclose(vertices[:, 2].max(), GRIP_KNOTS[-1][0], atol=1e-12)
    for z, r in GRIP_KNOTS:
        at_knot = np.abs(vertices[:, 2] - z) < 1e-9
        assert at_knot.sum() >= SETTINGS.club_sides
        assert abs(radius[at_knot].max() - r) < 1e-6
    wall = radius > GRIP_KNOTS[-1][1] - 1e-9
    order = np.argsort(vertices[wall, 2], kind="stable")
    along = radius[wall][order]
    assert np.all(np.diff(along) <= 1e-9)
    expected = np.interp(vertices[wall, 2], *np.array(GRIP_KNOTS).T)
    assert np.allclose(radius[wall], expected, atol=1e-9)
    spacing = np.diff(np.unique(np.round(vertices[wall, 2], 9)))
    assert spacing.max() <= SETTINGS.grip_ring_spacing_m + 1e-9


def test_drawn_rubber_shades_smoothly_along_the_taper() -> None:
    parts = render.build_club_parts(-0.03, 0.07, settings=SETTINGS, grip=GRIP)
    (rubber,) = _named(parts, "grip_rubber")
    vertices = rubber.mesh.vertices
    radius = _radius(vertices)
    wall = radius > GRIP_KNOTS[-1][1] - 1e-9
    normals = corner_normals(
        rubber.mesh,
        weld_tolerance_m=SETTINGS.normal_weld_tolerance_m,
        crease_deg=SETTINGS.normal_crease_deg,
    )
    faces = rubber.mesh.faces
    on_wall = wall[faces].all(axis=1)
    corner = vertices[faces[on_wall]].reshape(-1, 3)
    outward = corner[:, :2] / np.hypot(corner[:, 0], corner[:, 1])[:, None]
    wall_normals = normals[on_wall].reshape(-1, 3)
    assert np.einsum("ij,ij->i", wall_normals[:, :2], outward).min() > 0.999
    assert np.abs(wall_normals[:, 2]).max() < 0.03


def test_drawn_cap_flares_rounds_and_ends_flat() -> None:
    parts = render.build_club_parts(-0.03, 0.07, settings=SETTINGS, grip=GRIP)
    (cap,) = _named(parts, "grip_cap")
    vertices = cap.mesh.vertices
    radius = _radius(vertices)
    z0, r0 = GRIP_KNOTS[0]
    end = z0 - GRIP.cap_height_m
    assert abs(radius.max() - GRIP.cap_radius_m) < 1e-9
    assert np.isclose(vertices[:, 2].min(), end, atol=1e-12)
    assert np.isclose(vertices[:, 2].max(), z0, atol=1e-12)
    at_edge = np.abs(vertices[:, 2] - z0) < 1e-9
    assert abs(radius[at_edge].max() - r0) < 1e-9
    widest = vertices[np.abs(radius - GRIP.cap_radius_m) < 1e-9, 2]
    assert np.allclose(widest, end + GRIP.cap_round_m, atol=1e-9)
    on_end = np.abs(vertices[:, 2] - end) < 1e-9
    assert abs(radius[on_end].max() - (GRIP.cap_radius_m - GRIP.cap_round_m)) < 1e-9
    assert abs(radius[on_end].min() - GRIP.vent_radius_m) < 1e-9
    rounded = (vertices[:, 2] < end + GRIP.cap_round_m - 1e-9) & ~on_end
    assert rounded.sum() >= SETTINGS.club_sides
    centre = np.array([end + GRIP.cap_round_m, GRIP.cap_radius_m - GRIP.cap_round_m])
    offset = np.stack([vertices[rounded, 2], radius[rounded]], axis=1) - centre
    assert np.allclose(np.linalg.norm(offset, axis=1), GRIP.cap_round_m, atol=1e-9)


def test_drawn_vent_fills_the_cap_hole_and_is_optional() -> None:
    parts = render.build_club_parts(-0.03, 0.07, settings=SETTINGS, grip=GRIP)
    (vent,) = _named(parts, "grip_vent")
    end = GRIP_KNOTS[0][0] - GRIP.cap_height_m
    assert abs(_radius(vent.mesh.vertices).max() - GRIP.vent_radius_m) < 1e-9
    assert np.isclose(vent.mesh.vertices[:, 2].min(), end, atol=1e-12)
    assert vent.mesh.vertices[:, 2].max() <= GRIP_KNOTS[0][0] + 1e-12
    plain = GripDrawing(**{**GRIP_FIELDS, "vent_radius_m": 0.0})
    closed = render.build_club_parts(-0.03, 0.07, settings=SETTINGS, grip=plain)
    assert [p.name for p in closed] == ["grip_rubber", "grip_cap", "shaft"]
    (cap,) = _named(closed, "grip_cap")
    on_end = np.abs(cap.mesh.vertices[:, 2] - end) < 1e-9
    assert _radius(cap.mesh.vertices[on_end]).min() == 0.0


@pytest.mark.parametrize(
    ("z_max", "head"),
    [
        (0.07, GRIP_KNOTS[-1][0] + SETTINGS.shaft_extension_m),
        (
            0.30,
            0.30 + SETTINGS.grip_length_margin_m + SETTINGS.shaft_extension_m,
        ),
    ],
)
def test_drawn_shaft_starts_at_the_mouth_and_reaches_the_head(
    z_max: float, head: float
) -> None:
    parts = render.build_club_parts(-0.03, z_max, settings=SETTINGS, grip=GRIP)
    (shaft,) = _named(parts, "shaft")
    (rubber,) = _named(parts, "grip_rubber")
    mouth_z, mouth_r = GRIP_KNOTS[-1]
    z = shaft.mesh.vertices[:, 2]
    assert z.min() >= mouth_z - 1e-12
    assert np.isclose(z.max(), head, atol=1e-12)
    assert z.max() >= z_max
    assert np.isclose(rubber.mesh.vertices[:, 2].max(), mouth_z, atol=1e-12)
    shaft_r = _radius(shaft.mesh.vertices).max()
    assert abs(shaft_r - (mouth_r - SETTINGS.grip_mouth_step_m)) < 1e-9
    assert 0.0 < shaft_r < mouth_r


def test_drawn_grip_edge_cases_and_bad_inputs() -> None:
    sharp = GripDrawing(**{**GRIP_FIELDS, "cap_round_m": 0.0})
    flush = GripDrawing(
        **{**GRIP_FIELDS, "cap_radius_m": GRIP_KNOTS[0][1], "cap_round_m": 0.005}
    )
    for drawing in (sharp, flush):
        for part in render.build_club_parts(0.0, 0.1, settings=SETTINGS, grip=drawing):
            render.validated_part(part)
            assert part.mesh.volume() > 0.0
    (cap,) = _named(
        render.build_club_parts(0.0, 0.1, settings=SETTINGS, grip=sharp), "grip_cap"
    )
    assert abs(_radius(cap.mesh.vertices).max() - sharp.cap_radius_m) < 1e-9
    with pytest.raises(ValueError):
        render.build_club_parts(0.1, -0.1, settings=SETTINGS, grip=GRIP)
    with pytest.raises(ValueError, match="finite"):
        render.build_club_parts(float("nan"), 0.1, settings=SETTINGS, grip=GRIP)
    with pytest.raises(TypeError, match="GripDrawing"):
        render.build_club_parts(0.0, 0.1, settings=SETTINGS, grip=GRIP_FIELDS)
    thin = SETTINGS.with_overrides(grip_mouth_step_m=0.01)
    with pytest.raises(ValueError, match="grip_mouth_step_m"):
        render.build_club_parts(0.0, 0.1, settings=thin, grip=GRIP)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("grip_mouth_step_m", 0.0),
        ("grip_ring_spacing_m", 0.0),
        ("grip_cap_round_segments", 0),
        ("grip_cap_rgba", [0.1, 0.1, 0.1]),
        ("grip_vent_rgba", [0.1, 0.1, 0.1, 2.0]),
    ],
)
def test_bad_grip_drawing_settings_raise(key: str, value: object) -> None:
    with pytest.raises(ValueError):
        SETTINGS.with_overrides(**{key: value})


def test_vertex_normals_are_smooth_across_duplicated_seam_vertices() -> None:
    quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.3], [0, 1, 0.3]], dtype=float)
    fold = np.array([[1, 0, 0], [2, 0, 0.5], [2, 1, 0.8], [1, 1, 0.3]], dtype=float)
    mesh = Mesh(
        np.vstack([quad, fold]), np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    )
    normals = vertex_normals(mesh, weld_tolerance_m=SETTINGS.normal_weld_tolerance_m)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    np.testing.assert_allclose(normals[1], normals[4])
    np.testing.assert_allclose(normals[2], normals[7])
    sphere_like = _hand().components[0].mesh
    smooth = vertex_normals(
        sphere_like, weld_tolerance_m=SETTINGS.normal_weld_tolerance_m
    )
    outward = sphere_like.vertices - sphere_like.vertices.mean(axis=0)
    assert np.mean(np.einsum("ij,ij->i", smooth, outward) > 0.0) > 0.95


def test_corner_normals_keep_hard_edges_and_smooth_skin() -> None:
    corners = np.array(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float
    )
    quads = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    faces = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    cube = Mesh(corners, faces)
    kwargs = {"weld_tolerance_m": SETTINGS.normal_weld_tolerance_m}
    sharp = corner_normals(cube, crease_deg=SETTINGS.normal_crease_deg, **kwargs)
    tri = corners[faces]
    flat = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flat /= np.linalg.norm(flat, axis=1, keepdims=True)
    np.testing.assert_allclose(
        sharp, np.repeat(flat[:, None, :], 3, axis=1), atol=1e-12
    )
    rounded = corner_normals(cube, crease_deg=120.0, **kwargs)
    diagonal = (corners - 0.5) / np.linalg.norm(corners - 0.5, axis=1, keepdims=True)
    assert np.einsum("fcj,fcj->fc", rounded, diagonal[faces]).min() > 0.9
    finger = _hand().components[0].mesh
    smooth = corner_normals(finger, crease_deg=SETTINGS.normal_crease_deg, **kwargs)
    welded = vertex_normals(finger, **kwargs)
    assert np.einsum("fcj,fcj->fc", smooth, welded[finger.faces]).min() > 0.95
    np.testing.assert_allclose(np.linalg.norm(smooth, axis=2), 1.0)


# --- 3. camera rig ----------------------------------------------------------


def _rig(view: str, side: str = "lead", basis=None):
    skin = skin_vertices_of(render.assemble_scene(_hand(side), settings=SETTINGS))
    basis = basis or render.anatomical_basis(skin, side=side, settings=SETTINGS)
    return render.camera_rig(skin, view, basis, SETTINGS), basis, skin


def test_default_basis_is_orthonormal_with_shaft_on_z() -> None:
    for side in ("lead", "trail"):
        _, basis, _ = _rig("palm", side)
        frame = np.stack([basis.thumb, basis.dorsum, basis.shaft])
        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(basis.shaft, [0.0, 0.0, 1.0])


def test_camera_rig_is_deterministic() -> None:
    first, _, _ = _rig("eye")
    second, _, _ = _rig("eye")
    for name in ("eye", "target", "up", "right", "back"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.distance == second.distance


def test_palm_and_back_are_opposite_and_butt_sits_at_the_butt_end() -> None:
    palm, basis, _ = _rig("palm")
    back, _, _ = _rig("back")
    np.testing.assert_allclose(palm.back, -basis.dorsum, atol=1e-12)
    np.testing.assert_allclose(back.back, basis.dorsum, atol=1e-12)
    np.testing.assert_allclose(palm.back, -back.back, atol=1e-12)
    butt, _, _ = _rig("butt")
    assert butt.eye[2] < butt.target[2]


def test_butt_view_looks_up_the_shaft_from_the_palm_side() -> None:
    """Straight down the shaft the end cap hides the hand; the view is tilted
    to the palm side so the heel pad on the grip is in frame."""
    butt, basis, _ = _rig("butt")
    along = float(butt.back @ basis.shaft)
    palmar = float(-butt.back @ basis.dorsum)
    assert along < -0.7, "butt view no longer looks from the butt end"
    assert 0.3 < palmar < -along, "butt view is not tilted to the palm side"
    assert abs(float(butt.back @ basis.thumb)) < 1e-12
    assert float(butt.up @ basis.dorsum) > 0.7


@pytest.mark.parametrize("side", ["lead", "trail"])
@pytest.mark.parametrize("view", REQUIRED_VIEWS)
def test_projected_skin_fills_the_frame(view: str, side: str) -> None:
    rig, _, skin = _rig(view, side)
    fill = projected_fill(rig, skin)
    assert abs(fill - SETTINGS.fill_fraction) <= SETTINGS.fill_tolerance
    assert fill == pytest.approx(SETTINGS.fill_fraction, abs=0.01)
    ndc = rig.project(skin)
    centre = (ndc.min(axis=0) + ndc.max(axis=0)) / 2.0
    np.testing.assert_allclose(centre, 0.0, atol=0.01)
    assert np.abs(ndc).max() < 1.0


def _stubbed_scene(view_side: str = "lead"):
    """A hand with a forearm-like stub: the stub is skin but not the focus."""
    hand = render.assemble_scene(_hand(view_side), settings=SETTINGS)
    skin = [p for p in hand if p.kind == "skin"]
    club = [p for p in hand if p.kind != "skin"]
    vertices = skin_vertices_of(skin)
    reach = float(np.ptp(vertices, axis=0).max())
    basis = render.anatomical_basis(vertices, side=view_side, settings=SETTINGS)
    corners = 0.01 * np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Toward the butt, so no required view sees the stub shrink into the distance.
    offset = 0.9 * reach * (basis.dorsum + basis.thumb - basis.shaft)
    stub = Mesh(
        vertices.mean(axis=0) + corners + offset,
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )
    focused = [render.ScenePart(p.mesh, p.rgba, "skin") for p in skin]
    stub_part = render.ScenePart(
        stub, SETTINGS.skin_rgba, "skin", focus=np.zeros(4, dtype=bool)
    )
    return [*focused, stub_part, *club], vertices, basis


@pytest.mark.parametrize("view", REQUIRED_VIEWS)
def test_camera_frames_on_the_focus_not_on_every_skin_vertex(view: str) -> None:
    """The hand fills the frame; a forearm stub does not shrink it (#15150)."""
    parts, hand_vertices, basis = _stubbed_scene()
    np.testing.assert_array_equal(focus_vertices_of(parts), hand_vertices)
    assert len(skin_vertices_of(parts)) == len(hand_vertices) + 4
    frame = render.render_scene(parts, view, basis=basis, engine="matplotlib")
    assert projected_fill(frame.camera, hand_vertices) == pytest.approx(
        SETTINGS.fill_fraction, abs=0.01
    )
    assert frame.fill_fraction == pytest.approx(SETTINGS.fill_fraction, abs=0.01)
    whole = projected_fill(frame.camera, skin_vertices_of(parts))
    assert whole > SETTINGS.fill_fraction + SETTINGS.fill_tolerance


def test_a_part_without_focus_frames_on_all_of_its_vertices() -> None:
    parts = render.assemble_scene(_hand(), settings=SETTINGS)
    np.testing.assert_array_equal(focus_vertices_of(parts), skin_vertices_of(parts))


def test_bad_focus_masks_raise() -> None:
    mesh = _hand().mesh
    count = len(mesh.vertices)
    for bad in (np.ones(count - 1, dtype=bool), np.ones(count), "all"):
        with pytest.raises((ValueError, TypeError)):
            render.render_scene(
                [render.ScenePart(mesh, SETTINGS.skin_rgba, "skin", focus=bad)],
                "palm",
                engine="matplotlib",
            )
    nothing = render.ScenePart(
        mesh, SETTINGS.skin_rgba, "skin", focus=np.zeros(count, dtype=bool)
    )
    with pytest.raises(ValueError, match="focus"):
        render.render_scene([nothing], "palm", engine="matplotlib")


def test_render_view_frames_on_the_focus_a_result_declares() -> None:
    hand = _hand()
    mesh = hand.mesh
    focus = mesh.vertices[:, 2] >= np.median(mesh.vertices[:, 2])
    declared = SimpleNamespace(
        mesh=mesh, grip_radius_m=SETTINGS.grip_radius_m, frame_focus=focus
    )
    for roll in (0.0, 20.0):
        frame = render.render_view(
            declared, "back", shaft_roll_deg=roll, engine="matplotlib"
        )
        rotation = apply_shaft_roll(np.eye(3), APPEARANCE_GRIP_SHAFT_AXIS, roll)
        rolled = mesh.vertices[focus] @ rotation.T
        assert projected_fill(frame.camera, rolled) == pytest.approx(
            SETTINGS.fill_fraction, abs=0.01
        )


def test_rig_reads_view_coefficients_from_settings() -> None:
    raw = json.loads(json.dumps(RAW))
    raw["views"]["palm"]["eye"] = [0.0, 1.0, 0.0]
    flipped = settings_from_mapping(raw)
    palm, basis, skin = _rig("palm")
    moved = render.camera_rig(skin, "palm", basis, flipped)
    np.testing.assert_allclose(moved.back, -palm.back, atol=1e-12)


def test_explicit_basis_overrides_the_heuristic() -> None:
    explicit = ([0.0, 1.0, 0.0], [1.0, 0.0, 0.2])
    hand = _hand()
    skin = hand.mesh.vertices
    basis = orthonormal_basis(*explicit, epsilon=SETTINGS.epsilon)
    np.testing.assert_allclose(basis.dorsum, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(basis.thumb, [0.0, 1.0, 0.0], atol=1e-12)
    rig = render.camera_rig(skin, "back", basis, SETTINGS)
    np.testing.assert_allclose(rig.back, [1.0, 0.0, 0.0], atol=1e-12)
    with pytest.raises(ValueError, match="thumb"):
        orthonormal_basis([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], epsilon=SETTINGS.epsilon)
    with pytest.raises(TypeError):
        render.render_view(hand, "palm", basis="dorsum")


@pytest.mark.parametrize("side", ["lead", "trail"])
def test_capsule_components_declare_their_own_basis(side: str) -> None:
    hand = _hand(side)
    basis = result_basis(hand, settings=SETTINGS)
    assert basis is not None
    dorsum_shell = np.vstack(
        [c.mesh.vertices for c in hand.components if c.finger == "dorsum"]
    )
    thumb_shell = np.vstack(
        [c.mesh.vertices for c in hand.components if c.finger == "thumb"]
    )
    assert dorsum_shell.mean(axis=0) @ basis.dorsum > 0.0
    assert thumb_shell.mean(axis=0) @ basis.thumb > 0.0
    np.testing.assert_allclose(basis.shaft, [0.0, 0.0, 1.0])
    declared = SimpleNamespace(mesh=hand.mesh, render_basis=([0, 1, 0], [1, 0, 0]))
    np.testing.assert_allclose(
        result_basis(declared, settings=SETTINGS).dorsum, [1.0, 0.0, 0.0]
    )
    assert result_basis(SimpleNamespace(mesh=hand.mesh), settings=SETTINGS) is None


# --- 4. engines -------------------------------------------------------------


def _luminance(image: np.ndarray) -> np.ndarray:
    return (image.astype(float) / 255.0) @ np.asarray(SETTINGS.luminance_weights)


@pytest.fixture(scope="module")
def gl_frames() -> dict[str, render.RenderedView]:
    backend, errors = render.gl_engine_status(SETTINGS)
    if backend is None:
        pytest.skip(f"no MuJoCo GL context can be created: {errors}")
    hand = _hand()
    return {
        "palm": render.render_view(hand, "palm"),
        "palm_again": render.render_view(hand, "palm"),
        "back": render.render_view(hand, "back"),
        "palm_roll": render.render_view(hand, "palm", shaft_roll_deg=20.0),
    }


def test_gl_frame_is_square_rgb_from_a_gl_engine(gl_frames) -> None:
    palm = gl_frames["palm"]
    assert palm.engine in ("osmesa", "egl")
    assert palm.image.dtype == np.uint8
    assert palm.image.shape == (SETTINGS.frame_px, SETTINGS.frame_px, 3)
    assert palm.kind_mask is not None
    assert palm.kind_mask.shape == palm.image.shape[:2]
    assert palm.settings_hash == SETTINGS.settings_hash


def test_gl_repeat_render_is_byte_identical(gl_frames) -> None:
    assert gl_frames["palm"].image.tobytes() == gl_frames["palm_again"].image.tobytes()


def test_gl_skin_fill_is_within_tolerance(gl_frames) -> None:
    for frame in gl_frames.values():
        assert (
            abs(frame.fill_fraction - SETTINGS.fill_fraction) <= SETTINGS.fill_tolerance
        )


def test_gl_skin_is_shaded_not_a_flat_wash(gl_frames) -> None:
    for name in ("palm", "back"):
        frame = gl_frames[name]
        skin = _luminance(frame.image)[frame.kind_mask == 1]
        assert skin.std() >= SETTINGS.min_luminance_std
        assert skin.max() < 1.0, "the skin must not clip to white"
        assert np.mean(skin > 0.97) < 0.01


def test_gl_background_and_club_colours_are_present(gl_frames) -> None:
    frame = gl_frames["palm"]
    pixels = frame.image.astype(float) / 255.0
    background = pixels[frame.kind_mask == 0]
    np.testing.assert_allclose(
        background.mean(axis=0), SETTINGS.background_rgba[:3], atol=2.0 / 255.0
    )
    grip = pixels[frame.kind_mask == 2]
    assert len(grip) > 0.01 * SETTINGS.frame_px**2, "the grip must be visible"
    grip_rgb = np.asarray(SETTINGS.grip_rgba[:3])
    skin_rgb = np.asarray(SETTINGS.skin_rgba[:3])
    distance_to_grip = np.linalg.norm(np.median(grip, axis=0) - grip_rgb)
    distance_to_skin = np.linalg.norm(np.median(grip, axis=0) - skin_rgb)
    assert distance_to_grip < 0.15 < distance_to_skin
    skin = pixels[frame.kind_mask == 1]
    hue = np.median(skin, axis=0)
    cosine = hue @ skin_rgb / (np.linalg.norm(hue) * np.linalg.norm(skin_rgb))
    assert cosine > 0.995, "lit skin keeps the skin hue"
    assert np.linalg.norm(hue) > 2.0 * np.linalg.norm(np.median(grip, axis=0))


def test_gl_palm_differs_from_back_and_from_rolled_palm(gl_frames) -> None:
    palm = gl_frames["palm"].image.astype(float)
    for other in ("back", "palm_roll"):
        difference = np.abs(palm - gl_frames[other].image.astype(float)).mean() / 255.0
        assert difference > 0.01, other


def test_gl_shaft_is_visible_from_the_eye_view(gl_frames) -> None:
    frame = render.render_view(_hand("trail"), "eye", side="trail")
    assert np.count_nonzero(frame.kind_mask == 3) > 0
    assert np.count_nonzero(frame.kind_mask == 2) > 0


def test_gl_render_scene_frames_two_hands(gl_frames) -> None:
    lead, trail = _hand("lead"), _hand("trail")
    offset = trail.distance_below_butt_m - lead.distance_below_butt_m
    moved = Mesh(trail.mesh.vertices + [0.0, 0.0, offset], trail.mesh.faces)
    z = np.concatenate([lead.mesh.vertices[:, 2], moved.vertices[:, 2]])
    parts = [
        render.ScenePart(lead.mesh, SETTINGS.skin_rgba, "skin"),
        render.ScenePart(moved, SETTINGS.skin_rgba, "skin"),
        *render.build_club_parts(float(z.min()), float(z.max()), settings=SETTINGS),
    ]
    basis = render.anatomical_basis(lead.mesh.vertices, side="lead")
    both = render.render_scene(parts, "back", basis=basis)
    assert abs(both.fill_fraction - SETTINGS.fill_fraction) <= SETTINGS.fill_tolerance
    single = gl_frames["back"]
    assert np.count_nonzero(both.kind_mask == 1) != np.count_nonzero(
        single.kind_mask == 1
    )


def test_gl_measured_fill_follows_the_focus(gl_frames) -> None:
    """The reported fill is measured on the pixels of the focus region."""
    parts, _, basis = _stubbed_scene()
    frame = render.render_scene(parts, "back", basis=basis)
    assert frame.kind_mask is not None
    assert abs(frame.fill_fraction - SETTINGS.fill_fraction) <= SETTINGS.fill_tolerance


def test_matplotlib_fallback_is_framed_and_labelled() -> None:
    frame = render.render_view(_hand(), "palm", engine="matplotlib")
    assert frame.engine == "matplotlib"
    assert frame.image.shape == (SETTINGS.frame_px, SETTINGS.frame_px, 3)
    assert frame.kind_mask is None
    assert abs(frame.fill_fraction - SETTINGS.fill_fraction) <= SETTINGS.fill_tolerance
    background = np.rint(np.asarray(SETTINGS.background_rgba[:3]) * 255)
    foreground = np.abs(frame.image.astype(float) - background).max(axis=2) > 8
    rows = np.flatnonzero(foreground.any(axis=1))
    assert rows.size > 0
    assert _luminance(frame.image)[foreground].std() >= SETTINGS.min_luminance_std
    again = render.render_view(_hand(), "palm", engine="matplotlib")
    assert frame.image.tobytes() == again.image.tobytes()


def test_unknown_engine_raises() -> None:
    with pytest.raises(ValueError, match="engine"):
        render.render_view(_hand(), "palm", engine="pyrender")


# --- 5. sources, PNG, contact sheet ----------------------------------------


def test_capsule_source_is_registered_on_import() -> None:
    assert "capsule" in render.render_sources()
    built = render.build_render_source(
        "capsule", "trail", grip_radius_m=SETTINGS.grip_radius_m
    )
    assert built.hand is Hand.TRAIL
    with pytest.raises(ValueError, match="unknown render source"):
        render.build_render_source("absent", "lead", grip_radius_m=0.014)
    with pytest.raises(ValueError, match="already registered"):
        render.register_render_source("capsule", lambda **_: None)
    with pytest.raises(TypeError):
        render.register_render_source("thing", "not callable")


def test_save_png_round_trips_and_is_deterministic(tmp_path) -> None:
    from PIL import Image

    frame = render.render_view(_hand(), "butt", engine="matplotlib")
    first = render.save_png(frame, tmp_path / "a" / "butt.png")
    second = render.save_png(frame, tmp_path / "b.png")
    assert first.read_bytes() == second.read_bytes()
    loaded = np.asarray(Image.open(io.BytesIO(first.read_bytes())).convert("RGB"))
    np.testing.assert_array_equal(loaded, frame.image)
    with pytest.raises(ValueError):
        render.save_png(np.zeros((4, 4)), tmp_path / "bad.png")


def test_contact_sheet_has_a_cell_per_source_and_roll() -> None:
    small = SETTINGS.with_overrides(frame_px=128)
    sheet = render.render_contact_sheet(
        ("capsule",), view="palm", side="lead", settings=small, engine="matplotlib"
    )
    layout = small.sheet
    cell_w = layout.cell_px + 2 * layout.margin_px
    cell_h = layout.cell_px + layout.label_height_px + 2 * layout.margin_px
    assert sheet.image.shape == (cell_h, cell_w * len(small.rolls_deg), 3)
    assert sheet.row_labels == ("capsule",)
    assert sheet.column_labels == ("-20", "+0", "+20")
    assert len(sheet.cells) == 1
    assert len(sheet.cells[0]) == 3
    label = sheet.image[layout.margin_px : layout.margin_px + layout.label_height_px]
    assert label.max() > 200, "the label text must be drawn"
    again = render.render_contact_sheet(
        ("capsule",), view="palm", side="lead", settings=small, engine="matplotlib"
    )
    assert sheet.image.tobytes() == again.image.tobytes()
    with pytest.raises(ValueError, match="unknown render source"):
        render.render_contact_sheet(("absent",), settings=small, engine="matplotlib")


def test_cli_writes_identical_pngs_and_a_manifest(tmp_path) -> None:
    from scripts import render_hand_backend as cli

    runs = []
    for name in ("first", "second"):
        out = tmp_path / name
        argv = ["--backend", "capsule", "--views", "palm,butt", "--rolls", "-20,20"]
        argv += ["--frame-px", "96", "--engine", "matplotlib", "--both-hands"]
        argv += ["--contact-sheet", "--out", str(out)]
        assert cli.main(argv) == 0
        runs.append(out)
    names = sorted(p.name for p in runs[0].glob("*.png"))
    assert "capsule_lead_palm_rollm20.png" in names
    assert "capsule_trail_butt_rollp20.png" in names
    assert "capsule_both_palm_rollp00.png" in names
    assert "contact_sheet_lead_palm.png" in names
    assert len(names) == 2 * 2 * 2 + 2 + 2
    for name in names:
        assert (runs[0] / name).read_bytes() == (runs[1] / name).read_bytes()
    manifest = json.loads((runs[0] / cli.DEFAULT_MANIFEST).read_text("utf-8"))
    assert manifest["engines"] == ["matplotlib"]
    assert manifest["quality_bar_evidence"] is False
    assert manifest["frame_px"] == 96
    assert len(manifest["settings_hash"]) == 64
    assert {f["file"] for f in manifest["frames"]} | {
        s["file"] for s in manifest["contact_sheets"]
    } == set(names)
    assert str(tmp_path) not in json.dumps(manifest)
