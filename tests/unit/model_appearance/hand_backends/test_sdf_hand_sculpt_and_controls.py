"""RED/GREEN TDD tests for SDF hand sculpting, controls, and smooth shading."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec, HandSize
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.hand_backends.contract import HandJointAngles
from src.shared.python.model_appearance.hand_backends.sdf_field import evaluate_hand_sdf
from src.shared.python.model_appearance.hand_backends.sdf_pose import (
    pose_hand_skeleton,
)
from src.shared.python.model_appearance.hand_backends.sdf_render import (
    _scene,
    render_meshes,
    skin_srgb,
)
from src.shared.python.model_appearance.hand_backends.sdf_settings import (
    load_sdf_hand_settings,
)

pytestmark = pytest.mark.unit


def test_clear_of_grip_uses_directional_ellipsoid_support() -> None:
    """Flattened palmar ellipsoid must not stand off by its longest lateral axis."""
    from src.shared.python.model_appearance.hand_backends.sdf_primitives import (
        clear_of_grip_ellipsoid,
    )

    centre = np.array([0.020, 0.0, 0.0])
    along = np.array([0.0, 0.0, 1.0])
    palmar = np.array([-1.0, 0.0, 0.0])
    radii = (0.028, 0.009, 0.028)
    grip_radius_m = 0.014
    clearance_m = 0.001

    cleared = clear_of_grip_ellipsoid(
        centre,
        along,
        palmar,
        radii,
        grip_radius_m,
        clearance_m,
        margin_m=0.0015,
    )
    rho = float(np.hypot(cleared[0], cleared[1]))
    assert grip_radius_m + clearance_m + 0.009 <= rho < grip_radius_m + 0.016


def test_dorsal_mcp_knuckle_line_stands_proud_of_interdigital_valleys() -> None:
    """Dorsal MCP knuckle surface extends further dorsally than the valley between knuckles."""
    settings = load_sdf_hand_settings()
    spec = GripSpec.legacy()
    skeleton = pose_hand_skeleton(
        spec,
        "lead",
        HandJointAngles.from_grip_spec(spec, "lead"),
        grip_radius_m=settings.default_grip_radius_m,
        settings=settings,
    )
    by_name = {prim.name: prim for prim in skeleton.primitives}
    mid_mcp = np.asarray(by_name["middle_proximal"].start, dtype=float)
    ring_mcp = np.asarray(by_name["ring_proximal"].start, dtype=float)
    dorsal = np.array([mid_mcp[0], mid_mcp[1], 0.0])
    dorsal /= float(np.linalg.norm(dorsal))

    knuckle_probe = mid_mcp + dorsal * (
        by_name["middle_proximal"].radius0 + 0.003 * skeleton.scale
    )
    valley_mid = 0.5 * (mid_mcp + ring_mcp)
    valley_probe = valley_mid + dorsal * (
        by_name["middle_proximal"].radius0 + 0.003 * skeleton.scale
    )
    blends = dict(settings.blend_m)
    sdf_knuckle, sdf_valley = evaluate_hand_sdf(
        skeleton.primitives,
        np.vstack((knuckle_probe, valley_probe)),
        blends,
    )
    assert float(sdf_knuckle) < float(sdf_valley) - 0.0015


def test_breadth_cross_angle_and_grip_style_controls_move_skeleton() -> None:
    """HandSize.breadth_m, club_cross_angle_deg, and GripStyle affect the posed SDF skeleton."""
    settings = load_sdf_hand_settings()
    narrow_spec = GripSpec.legacy(hand_size=HandSize(length_m=0.193, breadth_m=0.080))
    wide_spec = GripSpec.legacy(hand_size=HandSize(length_m=0.193, breadth_m=0.094))
    narrow = pose_hand_skeleton(
        narrow_spec,
        "lead",
        HandJointAngles.from_grip_spec(narrow_spec, "lead"),
        grip_radius_m=settings.default_grip_radius_m,
        settings=settings,
    )
    wide = pose_hand_skeleton(
        wide_spec,
        "lead",
        HandJointAngles.from_grip_spec(wide_spec, "lead"),
        grip_radius_m=settings.default_grip_radius_m,
        settings=settings,
    )
    assert wide.palm_width_m > narrow.palm_width_m + 0.008

    low_cross = GripSpec.legacy(lead=HandGripSpec(club_cross_angle_deg=5.0))
    high_cross = GripSpec.legacy(lead=HandGripSpec(club_cross_angle_deg=65.0))
    sk_low = pose_hand_skeleton(
        low_cross,
        "lead",
        HandJointAngles.from_grip_spec(low_cross, "lead"),
        grip_radius_m=settings.default_grip_radius_m,
        settings=settings,
    )
    sk_high = pose_hand_skeleton(
        high_cross,
        "lead",
        HandJointAngles.from_grip_spec(high_cross, "lead"),
        grip_radius_m=settings.default_grip_radius_m,
        settings=settings,
    )
    idx_low = next(p for p in sk_low.primitives if p.name == "index_proximal")
    idx_high = next(p for p in sk_high.primitives if p.name == "index_proximal")
    assert np.linalg.norm(np.array(idx_low.start) - np.array(idx_high.start)) > 0.002

    ten_spec = GripSpec.legacy(style=GripStyle.TEN_FINGER)
    ov_spec = GripSpec.legacy(style=GripStyle.OVERLAP)
    sk_ten = pose_hand_skeleton(
        ten_spec,
        "trail",
        HandJointAngles.from_grip_spec(ten_spec, "trail"),
        grip_radius_m=settings.default_grip_radius_m,
        settings=settings,
    )
    sk_ov = pose_hand_skeleton(
        ov_spec,
        "trail",
        HandJointAngles.from_grip_spec(ov_spec, "trail"),
        grip_radius_m=settings.default_grip_radius_m,
        settings=settings,
    )
    pinky_ten = next(p for p in sk_ten.primitives if p.name == "little_middle")
    pinky_ov = next(p for p in sk_ov.primitives if p.name == "little_middle")
    rho_ten = float(np.hypot(pinky_ten.start[0], pinky_ten.start[1]))
    rho_ov = float(np.hypot(pinky_ov.start[0], pinky_ov.start[1]))
    assert rho_ov > rho_ten + 0.004


def test_scene_places_trail_hand_toward_clubhead_and_shades_smoothly() -> None:
    """In both-hands scene, trail shifts +spacing (+z) toward clubhead, and mesh shading is smooth."""
    pytest.importorskip("skimage")
    settings = load_sdf_hand_settings()
    lead_spec = GripSpec.legacy()
    trail_spec = GripSpec.legacy()
    lead_sk = pose_hand_skeleton(
        lead_spec,
        "lead",
        HandJointAngles.from_grip_spec(lead_spec, "lead"),
        grip_radius_m=settings.default_grip_radius_m,
        settings=settings,
    )
    trail_sk = pose_hand_skeleton(
        trail_spec,
        "trail",
        HandJointAngles.from_grip_spec(trail_spec, "trail"),
        grip_radius_m=settings.default_grip_radius_m,
        settings=settings,
    )
    from src.shared.python.model_appearance.hand_backends.sdf_hand import (
        HandVisualResult,
    )

    tri = Mesh(
        np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.01]]),
        np.array([[0, 1, 2]], dtype=np.int64),
    )
    lead_res = HandVisualResult(
        backend_id="sdf_hand",
        role="lead",
        mesh=tri,
        grip_radius_m=0.014,
        material=settings.skin_material,
        skeleton=lead_sk,
        settings=settings,
        roll=np.eye(3),
    )
    trail_res = HandVisualResult(
        backend_id="sdf_hand",
        role="trail",
        mesh=tri,
        grip_radius_m=0.014,
        material=settings.skin_material,
        skeleton=trail_sk,
        settings=settings,
        roll=np.eye(3),
    )
    _frame, drawables = _scene(
        (lead_res, trail_res), settings, np.array(skin_srgb(settings.skin_material))
    )
    lead_z = float(drawables[0][0].vertices[:, 2].mean())
    trail_z = float(drawables[1][0].vertices[:, 2].mean())
    assert trail_z > lead_z + 0.05

    # Curved icosahedron/octahedron test: Gouraud smooth shading varies within a triangle.
    verts = np.array(
        [
            [0.0, 0.0, 0.04],
            [0.04, 0.0, 0.0],
            [0.0, 0.04, 0.0],
            [-0.04, 0.0, 0.0],
            [0.0, -0.04, 0.0],
            [0.0, 0.0, -0.04],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [5, 2, 1],
            [5, 3, 2],
            [5, 4, 3],
            [5, 1, 4],
        ],
        dtype=np.int64,
    )
    octa = Mesh(verts, faces)
    img = render_meshes(
        [(octa, np.array([0.93, 0.76, 0.66]))],
        octa.vertices,
        np.array([0.4, 0.4, 0.8]),
        np.array([0.0, 0.0, 1.0]),
        settings,
        96,
    )
    # Count unique RGB shades on the lit front pixels (should be smooth, not just <= 4 flat facets).
    fg = img[np.linalg.norm(img - np.array(settings.background_rgb), axis=2) > 0.03]
    unique_shades = len(np.unique(np.round(fg, 3), axis=0))
    assert unique_shades >= 16
