"""Tests for finger pose driven by contact topology with per-hand roll (HRG-GS-G4, #13302)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact.contact_topology import (
    ContactTopology,
    GripStyle,
    contact_topology,
)
from src.shared.python.model_appearance import grip_pose
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hand_mesh import (
    DEFAULT_HAND_MESH_PARAMS,
    FINGER_NAMES,
    GripHandMesh,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)
from src.shared.python.motion_matching.grip_contract import (
    GripSpec,
    HandAngles,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = ROOT / "tests/fixtures/model_appearance/grip_finger_mesh_summary.json"


def _mesh_centroid_in_grip(hand_mesh: GripHandMesh, component_name: str) -> np.ndarray:
    """Compute the 3D vertex centroid of a hand component in the shared grip frame."""
    part = next(p for p in hand_mesh.components if p.name == component_name)
    v = part.mesh.vertices
    mean_v = v.mean(axis=0)
    # Hand-grip frame: x = rho*cos(theta), y = rho*sin(theta), z = axial down shaft from hand center
    # Shared grip frame: x = distance_below_butt + z, y = x_hand, z = y_hand
    return np.array(
        [
            hand_mesh.distance_below_butt_m + mean_v[2],
            mean_v[0],
            mean_v[1],
        ]
    )


def _azimuth_deg(pts: np.ndarray) -> float:
    mean_xy = pts[:, :2].mean(axis=0)
    return float(np.degrees(np.arctan2(mean_xy[1], mean_xy[0])))


def _wrap_deg(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


# =====================================================================
# 1. Duplicate _FINGER_LINKS tag table is removed
# =====================================================================


def test_finger_links_tag_table_removed_and_delegates_to_topology() -> None:
    """The duplicate _FINGER_LINKS string tag table is removed from grip_pose."""
    assert not hasattr(grip_pose, "_FINGER_LINKS")
    # linked_fingers() delegates to ContactTopology hand-hand pair names
    assert DEFAULT_GRIP_POSE.linked_fingers() == ("overlap_little_finger_lead_index",)
    interlock = GripPose(style=GripStyle.INTERLOCK)
    assert interlock.linked_fingers() == (
        "interlock_little_finger_lead_middle",
        "interlock_lead_index_wrap",
    )
    ten = GripPose(style=GripStyle.TEN_FINGER)
    assert ten.linked_fingers() == ()


# =====================================================================
# 2. Rendered fingertip positions match topology site centres <= 1 mm
# =====================================================================


@pytest.mark.parametrize("style", list(GripStyle))
def test_rendered_fingertip_positions_match_topology_site_centres_per_style(
    style: GripStyle,
) -> None:
    """For each style the rendered fingertip positions match topology site centres to <= 1 mm."""
    radius = 0.014
    spec = GripSpec(style=style)
    pose = spec.pose()
    topo = contact_topology(spec, grip_radius_m=radius)

    lead_hand = build_grip_hand_mesh(Hand.LEAD, grip_radius_m=radius, pose=pose)
    trail_hand = build_grip_hand_mesh(Hand.TRAIL, grip_radius_m=radius, pose=pose)

    # 1. Lead index finger matches lead_index site centre <= 1 mm
    lead_idx_site = topo.site("lead_index")
    lead_idx_tip = lead_hand.fingertip_in_grip_m("index")
    err_lead_idx = float(
        np.linalg.norm(lead_idx_tip - np.asarray(lead_idx_site.centre_m))
    )
    assert err_lead_idx <= 1.0e-3, (
        f"style {style.value} lead_index tip {lead_idx_tip} vs site {lead_idx_site.centre_m}, err={err_lead_idx * 1000:.3f} mm"
    )

    # Distal phalanx mesh vertex centroid also matches <= 1 mm
    lead_idx_centroid = _mesh_centroid_in_grip(lead_hand, "index_distal")
    err_lead_centroid = float(
        np.linalg.norm(lead_idx_centroid - np.asarray(lead_idx_site.centre_m))
    )
    assert err_lead_centroid <= 1.0e-3, (
        f"style {style.value} lead_index centroid err={err_lead_centroid * 1000:.3f} mm"
    )

    # 2. Trail little finger matches trail_little_finger site centre <= 1 mm
    trail_lit_site = topo.site("trail_little_finger")
    trail_lit_tip = trail_hand.fingertip_in_grip_m("little")
    err_trail_lit = float(
        np.linalg.norm(trail_lit_tip - np.asarray(trail_lit_site.centre_m))
    )
    assert err_trail_lit <= 1.0e-3, (
        f"style {style.value} trail_little tip {trail_lit_tip} vs site {trail_lit_site.centre_m}, err={err_trail_lit * 1000:.3f} mm"
    )

    # Distal phalanx mesh vertex centroid also matches <= 1 mm
    trail_lit_centroid = _mesh_centroid_in_grip(trail_hand, "little_distal")
    err_trail_centroid = float(
        np.linalg.norm(trail_lit_centroid - np.asarray(trail_lit_site.centre_m))
    )
    assert err_trail_centroid <= 1.0e-3, (
        f"style {style.value} trail_little centroid err={err_trail_centroid * 1000:.3f} mm"
    )


def test_trail_index_trigger_rendered_position_matches_topology_site_centre() -> None:
    """When trigger_finger=True, trail index matches trail_index_trigger site centre <= 1 mm."""
    radius = 0.014
    spec = GripSpec(style=GripStyle.OVERLAP, trigger_finger=True)
    pose = spec.pose()
    assert pose.trigger_finger is True
    topo = contact_topology(spec, grip_radius_m=radius)

    trail_hand = build_grip_hand_mesh(Hand.TRAIL, grip_radius_m=radius, pose=pose)
    trig_site = topo.site("trail_index_trigger")
    trig_tip = trail_hand.fingertip_in_grip_m("index")
    err_trig = float(np.linalg.norm(trig_tip - np.asarray(trig_site.centre_m)))
    assert err_trig <= 1.0e-3, (
        f"trigger tip {trig_tip} vs site {trig_site.centre_m}, err={err_trig * 1000:.3f} mm"
    )

    trig_centroid = _mesh_centroid_in_grip(trail_hand, "index_distal")
    err_centroid = float(np.linalg.norm(trig_centroid - np.asarray(trig_site.centre_m)))
    assert err_centroid <= 1.0e-3, f"trigger centroid err={err_centroid * 1000:.3f} mm"


# =====================================================================
# 3. Overlap, interlock and ten-finger produce visibly distinct fingers
# =====================================================================


def test_styles_produce_visibly_distinct_fingers() -> None:
    """Overlap, interlock and ten-finger produce distinct finger placement and on_grip flags."""
    radius = 0.014
    overlap_lead = build_grip_hand_mesh(
        Hand.LEAD, grip_radius_m=radius, style=GripStyle.OVERLAP
    )
    overlap_trail = build_grip_hand_mesh(
        Hand.TRAIL, grip_radius_m=radius, style=GripStyle.OVERLAP
    )

    interlock_lead = build_grip_hand_mesh(
        Hand.LEAD, grip_radius_m=radius, style=GripStyle.INTERLOCK
    )
    interlock_trail = build_grip_hand_mesh(
        Hand.TRAIL, grip_radius_m=radius, style=GripStyle.INTERLOCK
    )

    ten_lead = build_grip_hand_mesh(
        Hand.LEAD, grip_radius_m=radius, style=GripStyle.TEN_FINGER
    )
    ten_trail = build_grip_hand_mesh(
        Hand.TRAIL, grip_radius_m=radius, style=GripStyle.TEN_FINGER
    )

    # Trail little finger:
    # Overlap: lifted off grip on top of lead index (on_grip=False)
    # Interlock: on grip in the slot (on_grip=True)
    # Ten-finger: on grip abutting lead index (on_grip=True)
    assert not any(p.on_grip for p in overlap_trail.components if p.finger == "little")
    assert all(p.on_grip for p in interlock_trail.components if p.finger == "little")
    assert all(p.on_grip for p in ten_trail.components if p.finger == "little")

    # Lead index finger:
    # Overlap: on grip (on_grip=True)
    # Interlock: lifted off grip wrapping trail little (on_grip=False)
    # Ten-finger: on grip (on_grip=True)
    assert all(p.on_grip for p in overlap_lead.components if p.finger == "index")
    assert not any(p.on_grip for p in interlock_lead.components if p.finger == "index")
    assert all(p.on_grip for p in ten_lead.components if p.finger == "index")

    # Visibly distinct positions:
    # Overlap trail little radial distance is lifted (> 0.03 m)
    ov_lit_tip = overlap_trail.fingertip_in_grip_m("little")
    il_lit_tip = interlock_trail.fingertip_in_grip_m("little")
    ten_lit_tip = ten_trail.fingertip_in_grip_m("little")

    ov_rho = float(np.hypot(ov_lit_tip[1], ov_lit_tip[2]))
    il_rho = float(np.hypot(il_lit_tip[1], il_lit_tip[2]))
    ten_rho = float(np.hypot(ten_lit_tip[1], ten_lit_tip[2]))

    assert ov_rho > il_rho + 0.010  # > 10 mm radial lift
    assert abs(il_rho - ten_rho) < 2.0e-3  # both on grip
    assert (
        ten_lit_tip[0] > il_lit_tip[0] + 0.010
    )  # ten-finger moved > 10 mm down the shaft

    # Interlock lead index radial distance is lifted
    il_idx_tip = interlock_lead.fingertip_in_grip_m("index")
    ov_idx_tip = overlap_lead.fingertip_in_grip_m("index")
    il_idx_rho = float(np.hypot(il_idx_tip[1], il_idx_tip[2]))
    ov_idx_rho = float(np.hypot(ov_idx_tip[1], ov_idx_tip[2]))
    assert il_idx_rho > ov_idx_rho + 0.010  # > 10 mm radial lift


def test_mesh_summary_per_style_golden() -> None:
    """Finger mesh summary per style matches the golden fixture exactly."""
    if not FIXTURE_PATH.is_file():
        pytest.fail(f"Golden fixture missing: {FIXTURE_PATH}")
    golden = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    radius = 0.014
    for style in GripStyle:
        style_key = style.value
        assert style_key in golden
        lead = build_grip_hand_mesh(Hand.LEAD, grip_radius_m=radius, style=style)
        trail = build_grip_hand_mesh(Hand.TRAIL, grip_radius_m=radius, style=style)
        for hand_key, hand_mesh in [("lead", lead), ("trail", trail)]:
            actual_summary = {}
            for comp in hand_mesh.components:
                v = comp.mesh.vertices
                actual_summary[comp.name] = {
                    "on_grip": comp.on_grip,
                    "n_vertices": len(v),
                    "bbox_min": np.round(v.min(axis=0), 6).tolist(),
                    "bbox_max": np.round(v.max(axis=0), 6).tolist(),
                    "mean": np.round(v.mean(axis=0), 6).tolist(),
                }
            expected_summary = golden[style_key][hand_key]
            assert set(actual_summary) == set(expected_summary)
            for name, expected in expected_summary.items():
                actual = actual_summary[name]
                assert actual["on_grip"] == expected["on_grip"], (
                    f"{style_key} {hand_key} {name} on_grip"
                )
                assert actual["n_vertices"] == expected["n_vertices"], (
                    f"{style_key} {hand_key} {name} n_vertices"
                )
                np.testing.assert_allclose(
                    actual["bbox_min"], expected["bbox_min"], atol=1e-5
                )
                np.testing.assert_allclose(
                    actual["bbox_max"], expected["bbox_max"], atol=1e-5
                )
                np.testing.assert_allclose(actual["mean"], expected["mean"], atol=1e-5)


# =====================================================================
# 4. Per-hand roll follows alpha to <= 0.1 deg (absorbs #13166)
# =====================================================================


@pytest.mark.parametrize("delta_alpha_deg", [-25.0, -12.0, 15.0, 28.0])
def test_per_hand_roll_follows_alpha_within_tenth_degree(
    delta_alpha_deg: float,
) -> None:
    """A shift in alpha rotates the rendered hand about the grip axis by delta_alpha within <= 0.1 deg."""
    radius = 0.014
    base_spec = GripSpec()
    rolled_spec = GripSpec(
        lead=HandAngles(alpha_deg=delta_alpha_deg),
        trail=HandAngles(alpha_deg=delta_alpha_deg),
    )

    base_pose = base_spec.pose()
    rolled_pose = rolled_spec.pose()

    # GripSpec and GripPose visual properties (canonical HRG-A2 fields)
    assert rolled_spec.grip_strength_deg == pytest.approx(delta_alpha_deg, abs=1e-6)
    visual_pose = GripPose(
        grip_strength_deg=delta_alpha_deg,
        lead_roll_offset_deg=delta_alpha_deg,
        trail_roll_offset_deg=delta_alpha_deg,
    )
    assert visual_pose.grip_strength_deg == pytest.approx(delta_alpha_deg, abs=1e-6)
    assert visual_pose.roll_offset_deg(Hand.LEAD) == pytest.approx(
        delta_alpha_deg, abs=1e-6
    )
    assert visual_pose.roll_offset_deg(Hand.TRAIL) == pytest.approx(
        delta_alpha_deg, abs=1e-6
    )

    for hand in (Hand.LEAD, Hand.TRAIL):
        base_hand = build_grip_hand_mesh(hand, grip_radius_m=radius, pose=base_pose)
        rolled_hand = build_grip_hand_mesh(hand, grip_radius_m=radius, pose=rolled_pose)

        # Check fingertip position roll
        base_tip = base_hand.fingertip_position("index")
        rolled_tip = rolled_hand.fingertip_position("index")
        az_base = math.degrees(math.atan2(base_tip[1], base_tip[0]))
        az_rolled = math.degrees(math.atan2(rolled_tip[1], rolled_tip[0]))
        d_az = _wrap_deg(az_rolled - az_base)
        assert abs(d_az - delta_alpha_deg) <= 0.1, (
            f"{hand.value} index tip roll diff={abs(d_az - delta_alpha_deg):.4f} deg"
        )

        # Check palm component roll
        base_palm = next(p for p in base_hand.components if p.name == "palm")
        rolled_palm = next(p for p in rolled_hand.components if p.name == "palm")
        d_az_palm = _wrap_deg(
            _azimuth_deg(rolled_palm.mesh.vertices)
            - _azimuth_deg(base_palm.mesh.vertices)
        )
        assert abs(d_az_palm - delta_alpha_deg) <= 0.1, (
            f"{hand.value} palm roll diff={abs(d_az_palm - delta_alpha_deg):.4f} deg"
        )

        # Check thumb component roll
        base_thumb = next(p for p in base_hand.components if p.name == "thumb_distal")
        rolled_thumb = next(
            p for p in rolled_hand.components if p.name == "thumb_distal"
        )
        d_az_thumb = _wrap_deg(
            _azimuth_deg(rolled_thumb.mesh.vertices)
            - _azimuth_deg(base_thumb.mesh.vertices)
        )
        assert abs(d_az_thumb - delta_alpha_deg) <= 0.1, (
            f"{hand.value} thumb roll diff={abs(d_az_thumb - delta_alpha_deg):.4f} deg"
        )


def test_independent_per_hand_roll() -> None:
    """Lead and trail hands roll independently with their own alpha angles."""
    spec = GripSpec(
        lead=HandAngles(alpha_deg=10.0),
        trail=HandAngles(alpha_deg=-15.0),
    )
    pose = spec.pose()
    visual_pose = GripPose(lead_roll_offset_deg=10.0, trail_roll_offset_deg=-15.0)
    assert visual_pose.roll_offset_deg(Hand.LEAD) == pytest.approx(10.0)
    assert visual_pose.roll_offset_deg(Hand.TRAIL) == pytest.approx(-15.0)
    assert (
        pose.placement(Hand.LEAD).v_azimuth_deg - grip_pose.NEUTRAL_V_AZIMUTH_DEG
        == pytest.approx(10.0)
    )
    assert (
        pose.placement(Hand.TRAIL).v_azimuth_deg - grip_pose.NEUTRAL_V_AZIMUTH_DEG
        == pytest.approx(-15.0)
    )

    base = GripSpec().pose()
    lead_base = build_grip_hand_mesh(Hand.LEAD, grip_radius_m=0.014, pose=base)
    trail_base = build_grip_hand_mesh(Hand.TRAIL, grip_radius_m=0.014, pose=base)

    lead_rolled = build_grip_hand_mesh(Hand.LEAD, grip_radius_m=0.014, pose=pose)
    trail_rolled = build_grip_hand_mesh(Hand.TRAIL, grip_radius_m=0.014, pose=pose)

    palm_lead_base = next(p for p in lead_base.components if p.name == "palm")
    palm_lead_rolled = next(p for p in lead_rolled.components if p.name == "palm")
    d_lead = _wrap_deg(
        _azimuth_deg(palm_lead_rolled.mesh.vertices)
        - _azimuth_deg(palm_lead_base.mesh.vertices)
    )
    assert abs(d_lead - 10.0) <= 0.1

    palm_trail_base = next(p for p in trail_base.components if p.name == "palm")
    palm_trail_rolled = next(p for p in trail_rolled.components if p.name == "palm")
    d_trail = _wrap_deg(
        _azimuth_deg(palm_trail_rolled.mesh.vertices)
        - _azimuth_deg(palm_trail_base.mesh.vertices)
    )
    assert abs(d_trail - (-15.0)) <= 0.1


# =====================================================================
# 5. Solver constants, distal index, non-convergence error & site map
# =====================================================================


def test_solver_constants_and_divergence_error() -> None:
    """_solve_topology_finger uses named constants and raises ValueError if not converged."""
    from src.shared.python.grip_contact.contact_topology import ContactSite
    from src.shared.python.model_appearance import grip_hand_mesh
    from src.shared.python.model_appearance.grip_hand_mesh import (
        _DISTAL_INDEX,
        _SOLVER_MAX_ITERATIONS,
        _SOLVER_MIN_OFF_GRIP_TANGENT_RADIUS_M,
        _SOLVER_ON_GRIP_RADIAL_MARGIN_M,
        _SOLVER_STEP_RELAXATION,
        _SOLVER_TOLERANCE_M,
        _solve_topology_finger,
    )

    assert _DISTAL_INDEX == 2
    assert _SOLVER_ON_GRIP_RADIAL_MARGIN_M == 0.0001
    assert _SOLVER_MIN_OFF_GRIP_TANGENT_RADIUS_M == 0.005
    assert _SOLVER_STEP_RELAXATION == 0.95
    assert _SOLVER_MAX_ITERATIONS == 5
    assert _SOLVER_TOLERANCE_M == 1e-6

    # Test that a site geometrically impossible to converge raises ValueError
    # Target site with rho = 1 mm cannot be reached because d is bounded below by d_min
    impossible_site = ContactSite(
        name="impossible_site",
        hand="lead",
        segment="distal",
        primitive="sphere",
        centre_m=(0.05, 0.001, 0.0),
        radius_m=0.008,
    )
    with pytest.raises(ValueError, match="failed to converge"):
        _solve_topology_finger(
            lengths=(0.040, 0.024, 0.019),
            r=0.008,
            target_site=impossible_site,
            distance_below_butt_m=0.032,
            grip_radius=0.014,
            curl=-1,
            sides=16,
        )


def test_topology_site_map_routing() -> None:
    """_TOPOLOGY_SITE_MAP routes (hand, finger) pairs to canonical site names."""
    from src.shared.python.model_appearance.grip_hand_mesh import _TOPOLOGY_SITE_MAP

    assert _TOPOLOGY_SITE_MAP == {
        (Hand.LEAD, "index"): "lead_index",
        (Hand.TRAIL, "little"): "trail_little_finger",
    }
