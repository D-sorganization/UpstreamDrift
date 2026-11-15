"""Unit tests for the implicit SDF hand (issue #15155, epic #15150)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
)
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec, HandSize
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.hand_backends.contract import HandJointAngles
from src.shared.python.model_appearance.hand_backends.sdf_field import (
    analytic_grip_penetration,
    evaluate_hand_sdf,
    sd_round_cone,
    smin,
)
from src.shared.python.model_appearance.hand_backends.sdf_hand import build_hand_visual
from src.shared.python.model_appearance.hand_backends.sdf_mesh import mesh_is_watertight
from src.shared.python.model_appearance.hand_backends.sdf_pose import (
    phalanx_contact_points,
    pose_hand_skeleton,
)
from src.shared.python.model_appearance.hand_backends.sdf_render import (
    frame_half_extent,
    project_hand,
    rasterize_triangles,
    render_meshes,
)
from src.shared.python.model_appearance.hand_backends.sdf_settings import (
    load_sdf_hand_settings,
)
from src.shared.python.model_appearance.hand_backends.registry import (
    build_hand_visual as build_registered,
)
from src.shared.python.model_appearance.hand_backends.registry import (
    registered_backend_ids,
)

pytestmark = pytest.mark.unit

SETTINGS = load_sdf_hand_settings()
_CONTACT = (
    "index_proximal",
    "index_middle",
    "index_distal",
    "middle_proximal",
    "middle_middle",
    "middle_distal",
    "ring_proximal",
    "ring_middle",
    "ring_distal",
    "little_proximal",
    "little_middle",
    "little_distal",
    "thumb_proximal",
    "thumb_distal",
)


def _angles(spec: GripSpec, side: str) -> HandJointAngles:
    return HandJointAngles.from_grip_spec(spec, side)  # type: ignore[arg-type]


def _pose(spec: GripSpec, side: str, angles: HandJointAngles | None = None):
    return pose_hand_skeleton(
        spec,
        side,  # type: ignore[arg-type]
        _angles(spec, side) if angles is None else angles,
        grip_radius_m=SETTINGS.default_grip_radius_m,
        settings=SETTINGS,
    )


def _field(skeleton, points: np.ndarray) -> np.ndarray:
    return evaluate_hand_sdf(skeleton.primitives, points, dict(SETTINGS.blend_m))


def test_smin_matches_the_polynomial_identity() -> None:
    k = SETTINGS.blend_m["joint"]
    assert float(smin(np.array([0.0]), np.array([0.0]), k)[0]) == pytest.approx(
        -k / 6.0
    )
    assert float(smin(np.array([0.0]), np.array([1.0]), k)[0]) == pytest.approx(0.0)


def test_round_cone_is_negative_inside_and_zero_on_the_surface() -> None:
    length = 0.04
    radius = 0.01
    centre = np.array([[0.0, length / 2.0, 0.0]])
    surface = np.array([[radius, length / 2.0, 0.0]])
    before = np.array([[0.0, -2.0 * radius, 0.0]])
    assert float(sd_round_cone(centre, radius, radius, length)[0]) == pytest.approx(
        -radius
    )
    assert float(sd_round_cone(surface, radius, radius, length)[0]) == pytest.approx(
        0.0
    )
    assert float(sd_round_cone(before, radius, radius, length)[0]) == pytest.approx(
        radius
    )


def test_analytic_grip_penetration_is_the_overlap_of_the_two_interiors() -> None:
    radius = 0.014
    sdf = np.array([-0.004, 0.002, -0.001])
    points = np.array([[0.01, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    depth = analytic_grip_penetration(sdf, points, radius)
    assert depth[0] == pytest.approx(0.004)
    assert depth[1] == pytest.approx(0.0)
    assert depth[2] == pytest.approx(0.0)


def test_blend_radii_are_loaded_from_the_settings_file() -> None:
    raw = json.loads(Path(SETTINGS.source_path).read_text(encoding="utf-8"))
    assert SETTINGS.blend_m["mcp"] == raw["blend_m"]["mcp"]
    assert SETTINGS.blend_m["web"] == raw["blend_m"]["web"]
    assert SETTINGS.hand_length_m == raw["hand_length_m"]


def test_anthropometric_chain_matches_scaled_settings() -> None:
    spec = GripSpec.legacy()
    skeleton = _pose(spec, "lead")
    scale = spec.hand_size.length_m / SETTINGS.hand_length_m
    assert skeleton.hand_length_m == pytest.approx(
        SETTINGS.hand_length_m * scale, abs=1e-6
    )
    assert skeleton.palm_width_m == pytest.approx(
        SETTINGS.palm_width_m * scale, abs=1e-6
    )
    by_name = {prim.name: prim for prim in skeleton.primitives}
    proximal = by_name["middle_proximal"]
    distal = by_name["middle_distal"]
    assert 2.0 * proximal.radius0 == pytest.approx(0.020 * scale, abs=1e-6)
    assert 2.0 * distal.radius1 == pytest.approx(0.014 * scale, abs=1e-6)
    assert distal.radius1 < proximal.radius0
    for finger in ("index", "middle", "ring", "little"):
        profile = SETTINGS.finger(finger)
        for pair in profile.radii_m:
            assert 0.014 <= 2.0 * pair[1] <= 2.0 * pair[0] <= 0.020


def test_larger_hand_size_scales_the_measured_length() -> None:
    spec = GripSpec.legacy(hand_size=HandSize(length_m=0.22, breadth_m=0.10))
    skeleton = _pose(spec, "lead")
    assert skeleton.hand_length_m == pytest.approx(0.22, abs=1e-6)
    assert skeleton.scale == pytest.approx(0.22 / SETTINGS.hand_length_m)


def test_thumb_at_full_straightness_lies_along_the_shaft() -> None:
    skeleton = _pose(GripSpec.legacy(), "lead")
    distal = next(prim for prim in skeleton.primitives if prim.name == "thumb_distal")
    axis = np.asarray(distal.end) - np.asarray(distal.start)
    axis /= np.linalg.norm(axis)
    assert axis[2] > 0.95


def test_on_grip_phalanges_meet_the_clearance_band() -> None:
    skeleton = _pose(GripSpec.legacy(), "lead")
    radius = skeleton.grip_radius_m
    points = phalanx_contact_points(skeleton, radius)
    for name in _CONTACT:
        sample = points[name][None, :]
        gap = float(_field(skeleton, sample)[0])
        assert gap >= -SETTINGS.penetration_tolerance_m, name
        assert gap <= SETTINGS.contact_tolerance_m, name


def test_inset_grip_samples_do_not_penetrate_past_the_tolerance() -> None:
    skeleton = _pose(GripSpec.legacy(), "lead")
    radius = skeleton.grip_radius_m - SETTINGS.penetration_tolerance_m
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    heights = np.linspace(-0.04, 0.06, 12)
    theta, zed = np.meshgrid(angles, heights, indexing="ij")
    points = np.column_stack(
        (radius * np.cos(theta).ravel(), radius * np.sin(theta).ravel(), zed.ravel())
    )
    gap = _field(skeleton, points)
    assert float(np.min(gap)) >= -SETTINGS.penetration_tolerance_m


def test_opening_the_hand_lifts_the_fingers_off_the_grip() -> None:
    spec = GripSpec.legacy()
    closed = _pose(spec, "lead")
    opened = _pose(
        spec,
        "lead",
        HandJointAngles(
            palm_finger_depth=0.0, thumb_straightness=1.0, club_cross_angle_deg=0.0
        ),
    )
    sample = phalanx_contact_points(closed, closed.grip_radius_m)["middle_middle"][
        None, :
    ]
    lift = float(_field(opened, sample)[0] - _field(closed, sample)[0])
    assert lift > 0.5 * SETTINGS.open_hand_lift_m


def test_left_hand_is_the_y_mirror_of_the_right() -> None:
    right = _pose(GripSpec.legacy(), "lead")
    left = _pose(GripSpec.legacy(handedness="left"), "lead")
    xs = np.linspace(-0.02, 0.06, 4)
    ys = np.linspace(-0.05, 0.05, 4)
    zs = np.linspace(-0.04, 0.05, 3)
    grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
    flipped = grid.copy()
    flipped[:, 1] *= -1.0
    assert np.allclose(_field(left, flipped), _field(right, grid), atol=1e-8)


def test_lead_and_trail_mirror_when_azimuths_match() -> None:
    hand = HandGripSpec(v_azimuth_deg=0.0, thumb_azimuth_deg=0.0)
    spec = GripSpec.legacy(lead=hand, trail=hand)
    lead = _pose(spec, "lead")
    trail = _pose(spec, "trail")
    xs = np.linspace(-0.02, 0.06, 4)
    ys = np.linspace(-0.05, 0.05, 4)
    zs = np.linspace(-0.04, 0.05, 3)
    grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
    flipped = grid.copy()
    flipped[:, 1] *= -1.0
    assert np.allclose(_field(trail, flipped), _field(lead, grid), atol=1e-8)


def test_preconditions_reject_bad_inputs_before_meshing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbid(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("mesher ran during precondition checks")

    monkeypatch.setattr(
        "src.shared.python.model_appearance.hand_backends.sdf_hand.marching_cubes_mesh",
        _forbid,
    )
    spec = GripSpec.legacy()
    angles = _angles(spec, "lead")
    with pytest.raises(ValueError, match="side"):
        build_hand_visual(spec, "nope", angles)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grip_radius"):
        build_hand_visual(spec, "lead", angles, grip_radius_m=0.0)
    with pytest.raises(TypeError, match="HandJointAngles"):
        build_hand_visual(spec, "lead", "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resolution"):
        build_hand_visual(spec, "lead", angles, resolution=4)


def test_registry_dispatches_sdf_hand_and_rejects_unknown_ids() -> None:
    assert "sdf_hand" in registered_backend_ids()
    spec = GripSpec.legacy()
    with pytest.raises(KeyError, match="unknown"):
        build_registered("missing", spec, "lead", _angles(spec, "lead"))


def test_orthographic_frame_fills_seventy_percent() -> None:
    assert SETTINGS.frame_px >= 1024
    assert SETTINGS.frame_fill == pytest.approx(0.7)
    span = 1.0
    half = frame_half_extent(span, SETTINGS.frame_fill)
    assert span / (2.0 * half) == pytest.approx(SETTINGS.frame_fill)
    frame = 64
    corners = np.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]
    )
    pixels, _depth, _half = project_hand(
        corners,
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        SETTINGS.frame_fill,
        frame,
    )
    painted = float(pixels[:, 0].max() - pixels[:, 0].min())
    assert painted / (frame - 1) == pytest.approx(SETTINGS.frame_fill, abs=1e-6)


def test_rasterizer_paints_a_triangle_and_keeps_the_background() -> None:
    pixels = np.array([[8.0, 8.0], [40.0, 10.0], [12.0, 42.0]])
    depth = np.zeros(3)
    faces = np.array([[0, 1, 2]])
    background = np.array(SETTINGS.background_rgb)
    image = rasterize_triangles(
        pixels, depth, faces, np.array([0.2, 0.3, 0.4]), 64, background
    )
    assert image.shape == (64, 64, 3)
    assert np.any(np.linalg.norm(image - background, axis=2) > 0.05)
    assert np.allclose(image[0, 0], background)


def test_render_keeps_a_dark_club_pixel() -> None:
    skin = Mesh(
        np.array([[-0.04, -0.02, 0.0], [0.04, -0.02, 0.0], [0.0, 0.05, 0.0]]),
        np.array([[0, 1, 2]], dtype=np.int64),
    )
    club = Mesh(
        np.array([[-0.005, -0.005, -0.01], [0.005, -0.005, -0.01], [0.0, 0.005, 0.02]]),
        np.array([[0, 1, 2]], dtype=np.int64),
    )
    image = render_meshes(
        [(skin, np.array([0.93, 0.76, 0.66])), (club, np.array(SETTINGS.grip_rgb))],
        skin.vertices,
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        SETTINGS,
        SETTINGS.test_frame_px,
    )
    dark = np.linalg.norm(image, axis=2) < 0.35
    assert np.any(dark)


@pytest.fixture(scope="module")
def meshed_lead():
    pytest.importorskip("skimage")
    spec = GripSpec.legacy()
    return build_hand_visual(
        spec,
        "lead",
        _angles(spec, "lead"),
        settings=SETTINGS,
        resolution=SETTINGS.test_resolution,
    )


def test_test_resolution_mesh_is_watertight(meshed_lead) -> None:
    mesh = meshed_lead.mesh
    assert mesh_is_watertight(mesh.faces)
    assert mesh.volume() > 0.0
    assert np.isfinite(mesh.vertices).all()


def test_shaft_roll_is_a_vertex_rotation() -> None:
    pytest.importorskip("skimage")
    spec = GripSpec.legacy()
    rolled_spec = replace(spec, lead=replace(spec.lead, shaft_roll_deg=20.0))
    plain = build_hand_visual(
        spec,
        "lead",
        _angles(spec, "lead"),
        settings=SETTINGS,
        resolution=SETTINGS.min_resolution,
    )
    rolled = build_hand_visual(
        rolled_spec,
        "lead",
        _angles(rolled_spec, "lead"),
        settings=SETTINGS,
        resolution=SETTINGS.min_resolution,
    )
    rotation = apply_shaft_roll(np.eye(3), APPEARANCE_GRIP_SHAFT_AXIS, 20.0)[:3, :3]
    assert np.array_equal(rolled.mesh.faces, plain.mesh.faces)
    assert np.allclose(
        rolled.mesh.vertices, plain.mesh.vertices @ rotation.T, atol=1e-8
    )
    sample = np.array([[0.04, 0.01, 0.0]])
    direct = evaluate_hand_sdf(
        rolled.skeleton.primitives, sample @ rolled.roll, dict(SETTINGS.blend_m)
    )
    assert np.allclose(rolled.sdf(sample), direct, atol=1e-8)
