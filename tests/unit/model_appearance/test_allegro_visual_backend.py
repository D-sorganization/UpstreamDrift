"""Tests for Wonik Allegro 4-finger visual hand backend (#15152, #15150)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    resolved_shaft_roll_deg,
)
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec, HandSize

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
ALLEGRO_ROOT = (
    REPO_ROOT
    / "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/hand_assets/wonik_allegro"
)
CONFIG_PATH = REPO_ROOT / "scripts/config/hand_backend_allegro.json"


def _radial_xy(vertices: np.ndarray) -> np.ndarray:
    return np.hypot(vertices[:, 0], vertices[:, 1])


def _part_gap_m(a: np.ndarray, b: np.ndarray) -> float:
    return float(cKDTree(a).query(b, k=1)[0].min())


def test_allegro_license_notice_and_provenance_integrity() -> None:
    """Gate 1: BSD-2-Clause license, NOTICE, and every STL SHA256 verify on disk."""
    assert CONFIG_PATH.is_file()
    lic = ALLEGRO_ROOT / "LICENSE"
    notice = ALLEGRO_ROOT / "NOTICE"
    prov_path = ALLEGRO_ROOT / "PROVENANCE.json"
    assert lic.is_file()
    assert notice.is_file()
    assert prov_path.is_file()

    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov["spdx_license"] == "BSD-2-Clause"
    assert hashlib.sha256(lic.read_bytes()).hexdigest() == prov["license_sha256"]
    assert "wonik allegro" in notice.read_text(encoding="utf-8").lower()
    assert "BSD-2-Clause" in notice.read_text(encoding="utf-8")

    stls = sorted((ALLEGRO_ROOT / "assets").glob("*.stl"))
    assert len(stls) == 14
    for stl in stls:
        rel = f"assets/{stl.name}"
        assert rel in prov["stl_files_sha256"]
        assert (
            hashlib.sha256(stl.read_bytes()).hexdigest()
            == prov["stl_files_sha256"][rel]
        )


def test_registry_contract_and_visual_result() -> None:
    """Gate 2: backend registered under 'allegro', honors shared contract."""
    from src.shared.python.model_appearance.hand_backends import (
        HandVisualResult,
        build_hand_visual,
        list_hand_backends,
    )
    from src.shared.python.model_appearance.hand_backends.allegro_settings import (
        visual_settings,
    )

    assert "allegro" in list_hand_backends()
    cfg = visual_settings()
    spec = GripSpec.legacy()
    res = build_hand_visual("allegro", spec, "lead", None)
    assert isinstance(res, HandVisualResult)
    assert res.material_rgba == cfg.skin_tone_rgba
    assert np.isfinite(res.mesh.vertices).all()
    assert len(res.mesh.vertices) > 1000
    assert len(res.mesh.faces) > 1000
    assert abs(float(res.mesh.volume())) > 1e-6

    with pytest.raises(KeyError):
        build_hand_visual("nonexistent_backend_xyz", spec, "lead", None)


def test_hidden_hardware_and_mesh_count() -> None:
    """Gate 3: group-3 collision boxes/capsules are excluded from visual output."""
    from src.shared.python.model_appearance.hand_backends.allegro import (
        extract_allegro_hand_parts,
    )

    spec = GripSpec.legacy()
    parts = extract_allegro_hand_parts(spec, "lead")
    assert "palm" in parts
    for digit in ("index", "middle", "ring"):
        for seg in ("base", "proximal", "medial", "distal", "tip"):
            assert f"{digit}_{seg}" in parts
    for seg in ("base", "proximal", "medial", "distal", "tip"):
        assert f"thumb_{seg}" in parts
    assert not any("little" in k for k in parts)
    assert not any("collision" in k for k in parts)


def test_anthropometric_scale_tracks_hand_size() -> None:
    """Gate 4: native 24.6 cm robot hand scales to HandSize within human bounds."""
    from src.shared.python.model_appearance.hand_backends.allegro import (
        measure_allegro_anthropometry,
    )
    from src.shared.python.model_appearance.hand_backends.allegro_settings import (
        visual_settings,
    )

    cfg = visual_settings()
    base = measure_allegro_anthropometry(GripSpec.legacy(), "lead")
    lo_len, hi_len = cfg.hand_length_range_m
    lo_brd, hi_brd = cfg.palm_breadth_range_m
    lo_d, hi_d = cfg.finger_diameter_range_m

    assert lo_len <= base["hand_length_m"] <= hi_len
    assert lo_brd <= base["palm_breadth_m"] <= hi_brd
    for finger in ("index", "middle", "ring"):
        assert lo_d <= base[f"{finger}_proximal_width_m"] <= hi_d

    small_spec = GripSpec.legacy(
        hand_size=HandSize(length_m=0.175, breadth_m=0.175 * 0.449)
    )
    large_spec = GripSpec.legacy(
        hand_size=HandSize(length_m=0.205, breadth_m=0.205 * 0.449)
    )
    small = measure_allegro_anthropometry(small_spec, "lead")
    large = measure_allegro_anthropometry(large_spec, "lead")
    ratio = small["hand_length_m"] / large["hand_length_m"]
    assert abs(ratio - (0.175 / 0.205)) < 0.03


@pytest.mark.parametrize("side", ("lead", "trail"))
def test_topology_and_monotonic_knuckle_line(side: str) -> None:
    """Gate 5: digits attach to palm within tolerance and form monotonic MCP z line."""
    from src.shared.python.model_appearance.hand_backends.allegro import (
        extract_allegro_hand_parts,
    )
    from src.shared.python.model_appearance.hand_backends.allegro_settings import (
        visual_settings,
    )

    cfg = visual_settings()
    parts = extract_allegro_hand_parts(GripSpec.legacy(), side)  # type: ignore[arg-type]
    palm_v = parts["palm"].vertices
    max_gap = cfg.topology_knuckle_line_max_gap_m

    for digit in ("index", "middle", "ring", "thumb"):
        chain = ("base", "proximal", "medial", "distal", "tip")
        assert _part_gap_m(palm_v, parts[f"{digit}_base"].vertices) <= max_gap
        for a, b in zip(chain[:-1], chain[1:], strict=True):
            assert (
                _part_gap_m(
                    parts[f"{digit}_{a}"].vertices, parts[f"{digit}_{b}"].vertices
                )
                <= max_gap
            )

    # Index toward clubhead (+z), ring toward butt (-z) for both hands
    z_idx = float(parts["index_proximal"].vertices[:, 2].mean())
    z_mid = float(parts["middle_proximal"].vertices[:, 2].mean())
    z_rng = float(parts["ring_proximal"].vertices[:, 2].mean())
    assert z_idx > z_mid > z_rng


@pytest.mark.parametrize("radius_m", (0.0105, 0.012, 0.0135))
@pytest.mark.parametrize("side", ("lead", "trail"))
def test_curl_to_contact_across_grip_radii(side: str, radius_m: float) -> None:
    """Gate 6: zero club penetration, snug pad contact across 0.0105-0.0135 m."""
    from src.shared.python.model_appearance.hand_backends import build_hand_visual
    from src.shared.python.model_appearance.hand_backends.allegro import (
        finger_pad_contact_gaps,
        penetration_metrics,
    )
    from src.shared.python.model_appearance.hand_backends.allegro_settings import (
        visual_settings,
    )

    cfg = visual_settings()
    spec = GripSpec.legacy()
    res = build_hand_visual("allegro", spec, side, None, grip_radius_m=radius_m)  # type: ignore[arg-type]
    min_r = float(_radial_xy(res.mesh.vertices).min())
    assert min_r >= radius_m - cfg.club_penetration_tolerance_m

    gaps = finger_pad_contact_gaps(spec, side, grip_radius_m=radius_m)  # type: ignore[arg-type]
    for digit in ("index", "middle", "ring", "thumb"):
        assert (
            -cfg.club_penetration_tolerance_m
            <= gaps[digit]
            <= cfg.finger_contact_tolerance_m
        )

    pen = penetration_metrics(spec, side, grip_radius_m=radius_m)  # type: ignore[arg-type]
    assert pen["club_penetration_m"] <= cfg.club_penetration_tolerance_m
    assert pen["finger_finger_gap_m"] >= -cfg.finger_finger_penetration_tolerance_m


def test_invalid_grip_radius_rejected() -> None:
    """Gate 6b: DbC rejects out-of-range, non-finite, or bool grip_radius_m."""
    from src.shared.python.model_appearance.hand_backends import build_hand_visual

    spec = GripSpec.legacy()
    for bad in (-0.01, 0.0, 0.005, 0.03, float("nan"), True):
        with pytest.raises((ValueError, TypeError)):
            build_hand_visual("allegro", spec, "lead", None, grip_radius_m=bad)  # type: ignore[arg-type]


def test_shaft_roll_v_azimuth_mirroring_and_joint_angles() -> None:
    """Gate 7: shaft_roll_deg, v_azimuth_deg, handedness mirror, explicit joint_angles."""
    from src.shared.python.model_appearance.hand_backends import build_hand_visual
    from src.shared.python.model_appearance.hand_backends.allegro import (
        allegro_mount_determinant,
        extract_allegro_hand_parts,
    )

    base = GripSpec.legacy()
    res0 = build_hand_visual("allegro", base, "lead", None)

    rolled_spec = replace(base, lead=replace(base.lead, shaft_roll_deg=20.0))
    res_roll = build_hand_visual("allegro", rolled_spec, "lead", None)
    # Rotating about +z shaft axis keeps radial distance invariant
    assert np.allclose(
        _radial_xy(res0.mesh.vertices),
        _radial_xy(res_roll.mesh.vertices),
        atol=1e-9,
    )
    angle_rad = np.deg2rad(resolved_shaft_roll_deg(rolled_spec, "lead"))
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    expected_xy = np.column_stack(
        [
            c * res0.mesh.vertices[:, 0] - s * res0.mesh.vertices[:, 1],
            s * res0.mesh.vertices[:, 0] + c * res0.mesh.vertices[:, 1],
        ]
    )
    assert np.allclose(res_roll.mesh.vertices[:, :2], expected_xy, atol=1e-9)
    assert np.allclose(
        res_roll.mesh.vertices[:, 2], res0.mesh.vertices[:, 2], atol=1e-9
    )

    az_spec = replace(
        base, lead=replace(base.lead, v_azimuth_deg=base.lead.v_azimuth_deg + 30.0)
    )
    res_az = build_hand_visual("allegro", az_spec, "lead", None)
    assert not np.allclose(res0.mesh.vertices[:, :2], res_az.mesh.vertices[:, :2])

    # Proper orientation det > 0 for both handednesses
    left_spec = GripSpec.legacy(handedness="left")
    assert allegro_mount_determinant(base, "lead") > 0.0
    assert allegro_mount_determinant(base, "trail") > 0.0
    assert allegro_mount_determinant(left_spec, "lead") > 0.0
    assert allegro_mount_determinant(left_spec, "trail") > 0.0

    res_rh = build_hand_visual("allegro", base, "lead", None)
    res_lh = build_hand_visual("allegro", left_spec, "lead", None)
    assert np.isfinite(res_lh.mesh.vertices).all()
    assert (
        abs(
            float(res_rh.mesh.vertices[:, 1].mean() + res_lh.mesh.vertices[:, 1].mean())
        )
        < 0.008
    )
    parts_rh = extract_allegro_hand_parts(base, "lead")
    parts_lh = extract_allegro_hand_parts(left_spec, "lead")
    for part in ("palm", "middle_tip", "index_tip", "thumb_tip"):
        assert np.allclose(
            parts_rh[part].vertices[:, 2].mean(),
            parts_lh[part].vertices[:, 2].mean(),
            atol=0.004,
        )

    # Explicit joint_angles override
    override_q = np.zeros(16, dtype=float)
    res_flat = build_hand_visual("allegro", base, "lead", override_q)
    assert float(np.ptp(res_flat.mesh.vertices[:, 2])) != float(
        np.ptp(res0.mesh.vertices[:, 2])
    ) or not np.allclose(res_flat.mesh.vertices, res0.mesh.vertices)

    # Dict override + APPEARANCE_GRIP_SHAFT_AXIS sanity
    res_dict = build_hand_visual("allegro", base, "lead", {"ffj1": 0.6, "thj0": 0.9})
    assert not np.allclose(res_dict.mesh.vertices, res_flat.mesh.vertices)
    assert np.allclose(APPEARANCE_GRIP_SHAFT_AXIS, [0.0, 0.0, 1.0])


def test_evidence_renderer_framing_and_spacing(tmp_path: Path) -> None:
    """Gate 8: >=1024 px render, hand-only fill in [0.62, 0.80], trail z > lead z."""
    from docs.development.full_body_models.evidence.hand_backend_allegro.render_allegro_hand_evidence import (
        compute_hand_fill_fraction,
        render_single_view,
    )

    spec = GripSpec.legacy()
    out_png = tmp_path / "test_both.png"
    stats = render_single_view(
        spec,
        mode="both",
        view="side",
        roll_deg=0.0,
        out_path=out_png,
    )
    assert out_png.is_file()
    assert stats["width_px"] >= 1024
    assert stats["height_px"] >= 1024
    assert 0.62 <= stats["hand_fill_fraction"] <= 0.80
    assert stats["trail_z_mean_m"] > stats["lead_z_mean_m"]
    assert (
        0.0
        <= compute_hand_fill_fraction(
            stats["hand_projected_bounds"], stats["viewport_bounds"]
        )
        <= 1.0
    )


def _wrap_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize("spec_name", ("legacy", "neutral"))
@pytest.mark.parametrize("side", ("lead", "trail"))
def test_thumb_wraps_compactly_without_extending_down_shaft(
    spec_name: str, side: str
) -> None:
    """Thumb must wrap around the grip rather than pointing straight down +Z into trail palm."""
    from src.shared.python.model_appearance.hand_backends.allegro import (
        extract_allegro_hand_parts,
    )
    from src.shared.python.model_appearance.hand_backends.allegro_settings import (
        visual_settings,
    )

    cfg = visual_settings()
    spec = GripSpec.legacy() if spec_name == "legacy" else GripSpec.neutral()
    parts = extract_allegro_hand_parts(spec, side)  # type: ignore[arg-type]
    thumb_pts = np.vstack(
        [
            parts[f"thumb_{seg}"].vertices
            for seg in ("base", "proximal", "medial", "distal", "tip")
        ]
    )
    assert float(thumb_pts[:, 2].max()) <= cfg.thumb_max_z_m + 0.004

    tip_mean = parts["thumb_tip"].vertices.mean(axis=0)
    tip_az_deg = math.degrees(math.atan2(float(tip_mean[1]), float(tip_mean[0])))
    target_az_deg = spec.resolved_thumb_azimuth_deg(side)  # type: ignore[arg-type]
    assert abs(_wrap_deg(tip_az_deg - target_az_deg)) <= 45.0


@pytest.mark.parametrize("spec_name", ("legacy", "neutral"))
def test_bilateral_hands_do_not_collide_at_default_spacing(spec_name: str) -> None:
    """Lead thumb must stay clear of trail palm/ring volume at default hand_spacing_m."""
    from src.shared.python.model_appearance.hand_backends.allegro import (
        extract_allegro_hand_parts,
    )

    spec = GripSpec.legacy() if spec_name == "legacy" else GripSpec.neutral()
    lead_parts = extract_allegro_hand_parts(spec, "lead")
    trail_parts = extract_allegro_hand_parts(spec, "trail")

    lead_thumb = np.vstack(
        [
            lead_parts[f"thumb_{seg}"].vertices
            for seg in ("base", "proximal", "medial", "distal", "tip")
        ]
    )
    trail_all = np.vstack([m.vertices for m in trail_parts.values()])
    trail_all[:, 2] += float(spec.hand_spacing_m)
    assert float(lead_thumb[:, 2].max()) < float(spec.hand_spacing_m) - 0.012
    assert _part_gap_m(lead_thumb[::2], trail_all[::2]) >= 0.0002
