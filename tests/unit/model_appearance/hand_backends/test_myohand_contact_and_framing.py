"""TDD tests for MyoHand curl-to-contact, chirality, export parity & 70% framing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.shared.python.grip_spec import GripSpec, HandSize
from src.shared.python.hand_size import resolve_hand_breadth_m
from src.shared.python.model_appearance.club_assembly import (
    ClubAssembly,
    assembly_meshes,
)
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hands import grip_hand_frame
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.myohand import (
    build_myohand_bone_meshes,
    myohand_assets_ready,
    penetration_depth_m,
)
from src.shared.python.model_appearance.hand_backends.render_evidence import (
    LitRenderSettings,
    render_scene_png,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    default_myohand_settings,
)
from src.shared.python.model_appearance.hand_backends.skeleton_export import (
    export_bilateral_myohand_skeleton,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not myohand_assets_ready(), reason="myo_sim submodule not initialized"
    ),
]


def _signed_outward_volume_score(mesh: Mesh) -> float:
    tris = mesh.vertices[mesh.faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    centroids = tris.mean(axis=1) - mesh.vertices.mean(axis=0)
    return float(np.sum(centroids * normals))


def _min_radial_clearance_m(mesh: Mesh, grip_radius_m: float) -> float:
    radial = np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1])
    return float(radial.min() - grip_radius_m)


def _pair_min_distance_m(a: Mesh, b: Mesh) -> float:
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(b.vertices).query(a.vertices, k=1)
    return float(dist.min())


def _hand_size_spec(length_m: float) -> GripSpec:
    return GripSpec.legacy(
        hand_size=HandSize(
            length_m=length_m,
            breadth_m=resolve_hand_breadth_m(length_m),
        )
    )


def test_anthropometric_scaling_monotonic_on_final_bones() -> None:
    small = build_myohand_bone_meshes(_hand_size_spec(0.165), "trail")
    medium = build_myohand_bone_meshes(_hand_size_spec(0.189), "trail")
    large = build_myohand_bone_meshes(_hand_size_spec(0.215), "trail")

    def _breadth_span(m: dict[str, Mesh]) -> float:
        c2 = m["secondmc"].vertices.mean(axis=0)
        c5 = m["fifthmc"].vertices.mean(axis=0)
        return float(np.linalg.norm(c2 - c5))

    def _middle_chain_span(m: dict[str, Mesh]) -> float:
        pts = [
            m[name].vertices.mean(axis=0)
            for name in ("capitate", "thirdmc", "proxph3", "midph3", "distph3")
        ]
        return float(
            sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1))
        )

    assert _breadth_span(small) < _breadth_span(medium) < _breadth_span(large)
    assert (
        _middle_chain_span(small)
        < _middle_chain_span(medium)
        < _middle_chain_span(large)
    )


def test_mirroring_outward_winding_and_knuckle_attachment() -> None:
    rh_spec = GripSpec.legacy(handedness="right")
    lh_spec = GripSpec.legacy(handedness="left")

    mc_pairs = (
        ("secondmc", "proxph2"),
        ("thirdmc", "proxph3"),
        ("fourthmc", "proxph4"),
        ("fifthmc", "proxph5"),
    )

    for side in ("lead", "trail"):
        rh = build_myohand_bone_meshes(rh_spec, side)  # type: ignore[arg-type]
        lh = build_myohand_bone_meshes(lh_spec, side)  # type: ignore[arg-type]

        for bone_name in ("capitate", "thirdmc", "firstmc", "distph2", "distal_thumb"):
            c_rh = rh[bone_name].vertices.mean(axis=0)
            c_lh = lh[bone_name].vertices.mean(axis=0)
            assert c_lh[0] == pytest.approx(c_rh[0], abs=1e-5)
            assert c_lh[1] == pytest.approx(-c_rh[1], abs=1e-5)
            assert c_lh[2] == pytest.approx(c_rh[2], abs=1e-5)

            assert _signed_outward_volume_score(rh[bone_name]) > 0.0, (side, bone_name)
            assert _signed_outward_volume_score(lh[bone_name]) > 0.0, (side, bone_name)

        for mc_name, ph_name in mc_pairs:
            gap = _pair_min_distance_m(rh[mc_name], rh[ph_name])
            assert gap <= 0.008, (side, mc_name, ph_name, gap)


def test_fingers_and_thumb_wrap_grip_within_contact_tolerances() -> None:
    default_myohand_settings.cache_clear()
    cfg = default_myohand_settings()
    assert cfg.penetration_tolerance_m <= 0.0025

    spec = GripSpec.neutral()
    for side in ("lead", "trail"):
        meshes = build_myohand_bone_meshes(spec, side)  # type: ignore[arg-type]
        pen = penetration_depth_m(meshes, club_radius_m=cfg.club_cylinder_radius_m)
        assert pen <= cfg.penetration_tolerance_m, (side, pen)

        for i in (2, 3, 4, 5):
            mid_gap = _min_radial_clearance_m(meshes[f"midph{i}"], cfg.grip_radius_m)
            dist_gap = _min_radial_clearance_m(meshes[f"distph{i}"], cfg.grip_radius_m)
            assert (
                -cfg.penetration_tolerance_m
                <= mid_gap
                <= cfg.middle_contact_tolerance_m
            ), (
                side,
                i,
                mid_gap,
            )
            assert (
                -cfg.penetration_tolerance_m
                <= dist_gap
                <= cfg.distal_contact_tolerance_m
            ), (
                side,
                i,
                dist_gap,
            )

        thumb_gap = _min_radial_clearance_m(meshes["distal_thumb"], cfg.grip_radius_m)
        assert (
            -cfg.penetration_tolerance_m <= thumb_gap <= cfg.thumb_contact_tolerance_m
        ), (side, thumb_gap)
        thumb_max_z = float(meshes["distal_thumb"].vertices[:, 2].max())
        assert thumb_max_z <= cfg.thumb_max_z_m, (side, thumb_max_z)


def test_skeleton_export_matches_placed_bones_and_keeps_license_gate() -> None:
    spec = GripSpec.neutral()
    payload = export_bilateral_myohand_skeleton(spec)
    for side in ("lead", "trail"):
        side_data = payload[side]
        assert side_data["research_only"] is True
        assert side_data["license_gate_issue"] == 13763
        placed = build_myohand_bone_meshes(spec, side)  # type: ignore[arg-type]
        joints = side_data["joints"]
        assert len(joints) >= 23
        for idx, ph_name in (
            (2, "proxph2"),
            (3, "proxph3"),
            (4, "proxph4"),
            (5, "proxph5"),
        ):
            j_info = joints[f"mcp{idx}_flexion"]
            axis = np.asarray(j_info["axis_unit"], dtype=float)
            assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1e-5)
            origin = np.asarray(j_info["origin_m"], dtype=float)
            nearest = float(
                np.linalg.norm(placed[ph_name].vertices - origin, axis=1).min()
            )
            assert nearest <= 0.015, (side, ph_name, nearest)


def test_render_scene_png_frames_hand_at_seventy_percent_with_club(
    tmp_path: Path,
) -> None:
    spec = GripSpec.neutral()
    club = ClubAssembly("iron7", 0.95, 0.25, 0.0065)
    frame = grip_hand_frame(club, Hand.LEAD, spec.to_grip_pose())
    rot, trans = frame[:3, :3], frame[:3, 3]
    bones = {
        f"lead_{name}": Mesh(mesh.vertices @ rot.T + trans, mesh.faces)
        for name, mesh in build_myohand_bone_meshes(spec, "lead").items()
    }
    club_meshes = assembly_meshes(club)
    lit = LitRenderSettings(image_px=1024, frame_fill=0.70)

    bones_only = tmp_path / "bones.png"
    with_club = tmp_path / "with_club.png"
    render_scene_png(bones_only, bones, view="grip_oblique", settings=lit)
    render_scene_png(
        with_club,
        {"grip": club_meshes["grip"], "shaft": club_meshes["shaft"], **bones},
        view="grip_oblique",
        settings=lit,
    )

    arr_b = np.asarray(Image.open(bones_only).convert("RGB"))
    assert arr_b.shape == (1024, 1024, 3)
    bg_b = arr_b[0, 0]
    fg_b = np.max(np.abs(arr_b.astype(int) - bg_b.astype(int)), axis=2) > 12
    ys_b, xs_b = np.where(fg_b)
    fill_b = max(
        (xs_b.max() - xs_b.min() + 1) / 1024.0,
        (ys_b.max() - ys_b.min() + 1) / 1024.0,
    )
    assert 0.62 <= fill_b <= 0.78, fill_b

    arr_c = np.asarray(Image.open(with_club).convert("RGB"))
    assert arr_c.shape == (1024, 1024, 3)
    # Dark grip pixels must be visible and bone region must remain identical scale.
    luma_c = arr_c.mean(axis=2)
    assert int(np.sum(luma_c < 140)) > 2000
    bone_region = arr_b.mean(axis=2) < 235
    overlap_changed = int(
        np.sum(
            np.abs(arr_c[bone_region].mean(axis=1) - arr_b[bone_region].mean(axis=1))
            < 5
        )
    )
    assert overlap_changed > int(0.45 * np.sum(bone_region))
