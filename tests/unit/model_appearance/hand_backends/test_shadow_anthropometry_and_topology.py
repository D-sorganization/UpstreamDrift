"""TDD contracts for Shadow hand anthropometry, topology, contact curl, and mirroring."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.grip_spec.types import GripSpec, HandGripSpec, HandSize
from src.shared.python.model_appearance.hand_backends import shadow

pytestmark = pytest.mark.unit


def test_settings_expose_humanized_ranges_and_hidden_hardware() -> None:
    """Every size, tolerance, and hardware mask comes from settings."""
    cfg = shadow.visual_settings()
    assert cfg.grip_radius_range_m == (0.0105, 0.0135)
    assert cfg.render_frame_px >= 1024
    assert cfg.render_frame_fill >= 0.65
    assert "f_knuckle" in cfg.hidden_mesh_substrings
    assert "metacarpal" not in cfg.hidden_mesh_substrings
    assert cfg.club_penetration_tolerance_m <= 0.005


@pytest.mark.parametrize("side", ("lead", "trail"))
def test_anthropometric_scale_uncurled_and_hand_size_responsive(side: str) -> None:
    """Default Shadow hand matches human ANSUR bands and tracks HandSize inputs."""
    cfg = shadow.visual_settings()
    spec = GripSpec.legacy()
    parts = shadow.extract_shadow_hand_parts(
        spec,
        side,
        zero_qpos=True,  # type: ignore[arg-type]
    )
    all_verts = np.vstack([m.vertices for m in parts.values()])
    # Anatomical palm axes in world frame before curl: measure principal spans along
    # local distal and breadth directions exposed by the part bounding boxes.
    palm = parts["palm"].vertices
    lf_meta = parts["lf_metacarpal"].vertices
    palm_plus_meta = np.vstack([palm, lf_meta])
    # Measure distal extent (wrist -> middle tip) and palm breadth using part centroids.
    wrist_to_tip = float(np.linalg.norm(np.ptp(all_verts, axis=0)))
    # Better: use PCA / explicit span along anatomical axes returned by helper.
    dims = shadow.measure_shadow_anthropometry(spec, side)  # type: ignore[arg-type]
    lo_l, hi_l = cfg.hand_length_range_m
    lo_b, hi_b = cfg.palm_breadth_range_m
    lo_d, hi_d = cfg.finger_diameter_range_m
    assert lo_l <= dims["hand_length_m"] <= hi_l
    assert lo_b <= dims["palm_breadth_m"] <= hi_b
    for finger in ("index", "middle", "ring", "little"):
        assert lo_d <= dims[f"{finger}_proximal_width_m"] <= hi_d
    assert len(palm_plus_meta) > 0

    small = shadow.measure_shadow_anthropometry(
        GripSpec.legacy(hand_size=HandSize(length_m=0.170, breadth_m=0.075)),
        side,  # type: ignore[arg-type]
    )
    large = shadow.measure_shadow_anthropometry(
        GripSpec.legacy(hand_size=HandSize(length_m=0.210, breadth_m=0.095)),
        side,  # type: ignore[arg-type]
    )
    assert large["hand_length_m"] > dims["hand_length_m"] > small["hand_length_m"]
    assert large["palm_breadth_m"] > dims["palm_breadth_m"] > small["palm_breadth_m"]
    assert wrist_to_tip > 0.15


@pytest.mark.parametrize("handedness", ("right", "left"))
@pytest.mark.parametrize("side", ("lead", "trail"))
def test_topology_metacarpal_knuckle_line_and_axial_ordering(
    handedness: str, side: str
) -> None:
    """Knuckle line stays continuous and fingers stack little->index toward +Z_grip."""
    cfg = shadow.visual_settings()
    spec = GripSpec.legacy(handedness=handedness)  # type: ignore[arg-type]
    parts = shadow.extract_shadow_hand_parts(
        spec,
        side,
        grip_radius_m=0.012,  # type: ignore[arg-type]
    )
    assert "lf_metacarpal" in parts
    for forbidden in ("f_knuckle", "forearm", "wrist", "mounting_plate"):
        assert not any(forbidden in k for k in parts)

    palm_hull = np.vstack([parts["palm"].vertices, parts["lf_metacarpal"].vertices])
    max_gap = cfg.knuckle_line_max_gap_m
    z_centers: list[float] = []
    for finger in ("little", "ring", "middle", "index"):
        prox = parts[f"{finger}_proximal"].vertices
        dists = np.linalg.norm(prox[:, None, :] - palm_hull[None, :, :], axis=-1).min(
            axis=1
        )
        assert float(dists.min()) <= max_gap, f"{finger} detached from knuckle line"
        z_centers.append(float(prox[:, 2].mean()))

    assert z_centers[0] < z_centers[1] < z_centers[2] < z_centers[3], (
        f"Expected little < ring < middle < index along +Z_grip, got {z_centers}"
    )


@pytest.mark.parametrize("grip_radius_m", (0.0105, 0.0120, 0.0135))
@pytest.mark.parametrize("side", ("lead", "trail"))
def test_curl_to_contact_middle_and_distal_without_penetration(
    side: str, grip_radius_m: float
) -> None:
    """Proximal-to-distal curl seats middle and distal phalanges onto the grip."""
    cfg = shadow.visual_settings()
    spec = GripSpec.legacy()
    parts = shadow.extract_shadow_hand_parts(
        spec,
        side,
        grip_radius_m=grip_radius_m,  # type: ignore[arg-type]
    )
    target = grip_radius_m + cfg.pad_clearance_m
    tol = cfg.finger_contact_tolerance_m
    for finger in ("index", "middle", "ring", "little"):
        for seg in ("proximal", "middle", "distal"):
            pts = parts[f"{finger}_{seg}"].vertices
            radial = np.hypot(pts[:, 0], pts[:, 1])
            min_r = float(radial.min())
            assert min_r >= grip_radius_m - cfg.club_penetration_tolerance_m
            if seg in ("middle", "distal"):
                assert abs(min_r - target) <= tol, (
                    f"{side} {finger}_{seg} gap: min_r={min_r:.4f} vs target={target:.4f}"
                )

    result = shadow.build_hand_visual(
        spec,
        side,
        None,
        grip_radius_m=grip_radius_m,  # type: ignore[arg-type]
    )
    assert result.mesh.volume() > 0.0
    hand = shadow.Hand.TRAIL if side == "trail" else shadow.Hand.LEAD
    spec_hand = spec.trail if side == "trail" else spec.lead
    solved = shadow.solve_shadow_hand_qpos(
        hand, spec_hand, grip_spec=spec, grip_radius_m=grip_radius_m
    )
    metrics = shadow.penetration_metrics(
        hand,
        solved.qpos,
        grip_radius_m=grip_radius_m,
        mount_matrix=solved.mount_matrix,
        handedness=spec.handedness,
    )
    assert metrics["club_penetration_m"] <= cfg.club_penetration_tolerance_m
    mesh_min_r = float(
        np.hypot(result.mesh.vertices[:, 0], result.mesh.vertices[:, 1]).min()
    )
    assert mesh_min_r >= grip_radius_m - cfg.club_penetration_tolerance_m


def test_left_handed_mirroring_and_dbc_inputs() -> None:
    """Left-handed spec mirrors Y with outward winding; invalid inputs fail cleanly."""
    r_lead = shadow.build_hand_visual(
        GripSpec.legacy(handedness="right"), "lead", None, grip_radius_m=0.012
    ).mesh
    l_lead = shadow.build_hand_visual(
        GripSpec.legacy(handedness="left"), "lead", None, grip_radius_m=0.012
    ).mesh
    assert r_lead.volume() > 0.0
    assert l_lead.volume() > 0.0
    assert (
        float(r_lead.vertices[:, 1].mean()) * float(l_lead.vertices[:, 1].mean()) < 0.0
    )

    with pytest.raises(ValueError):
        shadow.build_hand_visual(GripSpec.legacy(), "lead", None, grip_radius_m=0.005)
    with pytest.raises(ValueError):
        shadow.build_hand_visual(GripSpec.legacy(), "lead", None, grip_radius_m=0.030)
    with pytest.raises(ValueError):
        shadow.build_hand_visual(
            GripSpec.legacy(),
            "invalid",
            None,
            grip_radius_m=0.012,  # type: ignore[arg-type]
        )
    # Smoke check shaft_roll_deg does not crash build_hand_visual
    rolled = shadow.build_hand_visual(
        GripSpec.legacy(lead=HandGripSpec(shaft_roll_deg=20.0)),
        "lead",
        None,
        grip_radius_m=0.012,
    )
    assert rolled.mesh.vertices.shape == r_lead.vertices.shape
