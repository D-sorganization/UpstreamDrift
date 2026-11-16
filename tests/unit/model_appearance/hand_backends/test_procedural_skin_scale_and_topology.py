"""TDD tests for procedural_skin scale, knuckle topology, wrap, and mirroring (#15145)."""

from __future__ import annotations

from collections import Counter
import math

import numpy as np
import pytest

from src.shared.python.grip_spec import GripSpec, HandGripSpec, HandSize
from src.shared.python.hand_size import hand_mesh_scale
from src.shared.python.model_appearance.grip_hand_mesh import (
    FINGER_NAMES,
    HandMeshParams,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import GripStyle, Hand
from src.shared.python.model_appearance.hand_backends.metrics import (
    grip_shaft_penetration_fraction,
)
from src.shared.python.model_appearance.hand_backends.placement import (
    axial_offset_vs_lead_m,
)
from src.shared.python.model_appearance.hand_backends.procedural_skin import (
    build_hand_visual,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_VISUAL_GRIP_RADIUS_M,
    HandVisualSettings,
)
from src.shared.python.model_appearance.hand_backends.skeleton_extract import (
    finger_polyline,
    middle_finger_length,
)

pytestmark = pytest.mark.unit


def _edge_counts(faces: np.ndarray) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edge = (u, v) if u < v else (v, u)
            counts[edge] += 1
    return counts


@pytest.mark.parametrize("role", ("lead", "trail"))
def test_skeleton_extract_finger_polylines_are_coherent_and_unflipped(
    role: str,
) -> None:
    spec = GripSpec.legacy()
    hand = Hand.LEAD if role == "lead" else Hand.TRAIL
    baked = build_grip_hand_mesh(
        hand,
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        pose=spec.to_grip_pose(),
        params=HandMeshParams(scale=hand_mesh_scale(spec.hand_size.length_m)),
    )
    mid = finger_polyline(baked, "middle")
    segs = np.diff(mid, axis=0)
    dirs = segs / np.maximum(np.linalg.norm(segs, axis=1, keepdims=True), 1e-12)
    for i in range(len(dirs) - 1):
        assert float(np.dot(dirs[i], dirs[i + 1])) > -0.25
    mid_len = middle_finger_length(mid)
    assert 0.072 <= mid_len <= 0.105


def test_trail_shared_visual_offset_places_trail_toward_clubhead() -> None:
    spec = GripSpec.legacy()
    dz = axial_offset_vs_lead_m(spec, "trail")
    assert dz == pytest.approx(spec.hand_spacing_m, abs=1e-6)
    assert dz > 0.0


@pytest.mark.parametrize("role", ("lead", "trail"))
def test_procedural_skin_mesh_is_watertight_and_not_over_scaled(role: str) -> None:
    spec = GripSpec.legacy()
    settings = HandVisualSettings()
    vis = build_hand_visual(
        spec,
        role,  # type: ignore[arg-type]
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        settings=settings,
    )
    verts = vis.mesh.vertices
    assert np.isfinite(verts).all()
    assert vis.mesh.volume() > 0.0
    counts = _edge_counts(vis.mesh.faces)
    boundary = [e for e, cnt in counts.items() if cnt != 2]
    assert boundary == [], (
        f"expected closed watertight manifold shells, found {len(boundary)} non-manifold edges"
    )
    aabb_span = float(np.max(verts.max(axis=0) - verts.min(axis=0)))
    assert aabb_span < settings.anthropometry.hand_length_range_m[0] * 0.90


@pytest.mark.parametrize("role", ("lead", "trail"))
def test_procedural_skin_anthropometric_scale_matches_settings_bands(
    role: str,
) -> None:
    from src.shared.python.model_appearance.hand_backends.procedural_skin import (
        measure_procedural_hand_anthropometry,
    )

    spec = GripSpec.legacy()
    settings = HandVisualSettings()
    anthro = settings.anthropometry
    report = measure_procedural_hand_anthropometry(
        spec,
        role,  # type: ignore[arg-type]
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        settings=settings,
    )
    base_lo, base_hi = anthro.finger_base_diameter_range_m
    tip_lo, tip_hi = anthro.finger_tip_diameter_range_m
    for finger in FINGER_NAMES:
        base_d = report["finger_base_diameters_m"][finger]
        tip_d = report["finger_tip_diameters_m"][finger]
        assert base_lo <= base_d <= base_hi, f"{finger} base diameter {base_d}"
        assert tip_lo <= tip_d <= tip_hi, f"{finger} tip diameter {tip_d}"
        assert base_d > tip_d
    pw_lo, pw_hi = anthro.palm_width_range_m
    pl_lo, pl_hi = anthro.palm_length_range_m
    hl_lo, hl_hi = anthro.hand_length_range_m
    assert pw_lo <= report["palm_width_m"] <= pw_hi
    assert pl_lo <= report["palm_length_m"] <= pl_hi
    assert hl_lo <= report["hand_length_m"] <= hl_hi
    mid_target = anthro.middle_finger_length_m
    assert (
        abs(report["middle_finger_length_m"] - mid_target) / mid_target
        <= anthro.ratio_tolerance_rel
    )


def test_procedural_skin_scales_monotonically_with_hand_size() -> None:
    from src.shared.python.model_appearance.hand_backends.procedural_skin import (
        measure_procedural_hand_anthropometry,
    )

    settings = HandVisualSettings()
    small = GripSpec.legacy(hand_size=HandSize(length_m=0.16, breadth_m=0.072))
    large = GripSpec.legacy(hand_size=HandSize(length_m=0.22, breadth_m=0.098))
    rep_small = measure_procedural_hand_anthropometry(
        small, "lead", grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M, settings=settings
    )
    rep_large = measure_procedural_hand_anthropometry(
        large, "lead", grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M, settings=settings
    )
    assert rep_large["hand_length_m"] > rep_small["hand_length_m"] * 1.25
    assert rep_large["palm_width_m"] > rep_small["palm_width_m"] * 1.25


@pytest.mark.parametrize("role", ("lead", "trail"))
def test_procedural_skin_knuckle_line_topology_cupped_palm_and_separation(
    role: str,
) -> None:
    from src.shared.python.model_appearance.hand_backends.procedural_skin import (
        measure_procedural_hand_anthropometry,
    )

    spec = GripSpec.legacy()
    settings = HandVisualSettings()
    anthro = settings.anthropometry
    report = measure_procedural_hand_anthropometry(
        spec,
        role,  # type: ignore[arg-type]
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        settings=settings,
    )
    assert report["max_knuckle_gap_m"] <= anthro.knuckle_attachment_tolerance_m
    mcp_z = [
        report["mcp_positions_m"][f][2] for f in ("little", "ring", "middle", "index")
    ]
    assert mcp_z == sorted(mcp_z)
    assert mcp_z[-1] - mcp_z[0] >= 0.040
    assert report["cup_depth_m"] >= settings.palm_cup_depth_min_m
    assert report["min_inter_finger_gap_m"] >= settings.inter_finger_min_gap_m


def test_open_hand_pose_aligns_finger_tangents_within_30_deg() -> None:
    from src.shared.python.model_appearance.hand_backends.procedural_skin import (
        measure_procedural_hand_anthropometry,
    )

    open_hand = HandGripSpec(palm_finger_depth=0.0)
    spec = GripSpec.legacy(lead=open_hand, trail=open_hand)
    settings = HandVisualSettings()
    rep = measure_procedural_hand_anthropometry(
        spec, "lead", grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M, settings=settings
    )
    dirs = [np.asarray(rep["proximal_directions"][f], float) for f in FINGER_NAMES]
    cos_limit = math.cos(math.radians(30.0))
    for i in range(len(dirs) - 1):
        assert float(np.dot(dirs[i], dirs[i + 1])) >= cos_limit


@pytest.mark.parametrize("grip_radius_m", (0.0105, 0.0120, 0.0135, 0.0140))
@pytest.mark.parametrize("role", ("lead", "trail"))
def test_procedural_skin_wraps_grip_radius_within_contact_tolerance(
    grip_radius_m: float, role: str
) -> None:
    from src.shared.python.model_appearance.hand_backends.procedural_skin import (
        measure_procedural_hand_anthropometry,
    )

    spec = GripSpec.legacy(style=GripStyle.TEN_FINGER)
    settings = HandVisualSettings()
    vis = build_hand_visual(
        spec,
        role,  # type: ignore[arg-type]
        grip_radius_m=grip_radius_m,
        settings=settings,
    )
    pen = grip_shaft_penetration_fraction(
        vis.mesh,
        grip_radius_m=grip_radius_m,
        tolerance_m=settings.shaft_penetration_tolerance_m,
    )
    assert pen <= settings.max_shaft_penetration_fraction
    report = measure_procedural_hand_anthropometry(
        spec,
        role,  # type: ignore[arg-type]
        grip_radius_m=grip_radius_m,
        settings=settings,
    )
    for finger in FINGER_NAMES:
        gap = report["pad_clearances_m"][finger]
        assert (
            -settings.shaft_penetration_tolerance_m
            <= gap
            <= settings.pad_contact_tolerance_m
        )


def test_left_handed_and_roll_mirroring_are_exact() -> None:
    settings = HandVisualSettings()
    right = GripSpec.legacy(lead=HandGripSpec(shaft_roll_deg=20.0))
    left = GripSpec.legacy(handedness="left", lead=HandGripSpec(shaft_roll_deg=20.0))
    vis_r = build_hand_visual(
        right,
        "lead",
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        settings=settings,
    )
    vis_l = build_hand_visual(
        left,
        "lead",
        grip_radius_m=DEFAULT_VISUAL_GRIP_RADIUS_M,
        settings=settings,
    )
    assert vis_l.mesh.volume() > 0.0
    rolled_r = vis_r.mesh
    rolled_l = vis_l.mesh
    mean_r = rolled_r.vertices.mean(axis=0)
    mean_l = rolled_l.vertices.mean(axis=0)
    assert mean_l[0] == pytest.approx(mean_r[0], abs=1e-4)
    assert mean_l[1] == pytest.approx(-mean_r[1], abs=1e-4)
    assert mean_l[2] == pytest.approx(mean_r[2], abs=1e-4)
